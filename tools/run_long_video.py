"""
Memory-efficient SAM3 video inference for thousands of frames.

Outputs a visualization video with mask overlays rendered on-the-fly --
nothing accumulates in memory across frames.

Key optimizations:
  1. LazyFrameLoader -- loads one frame at a time from disk (image folder or
     video file).  Nothing is kept in CPU or GPU memory after the frame is
     consumed by the model.  ``init_state`` is bypassed entirely so that
     ``load_resource_as_video_frames`` is never called.
  2. After every frame the tracker's per-frame output_dict is trimmed so that
     only the three keys the tracker reads from past frames survive:
     ``maskmem_features`` (spatial memory), ``maskmem_pos_enc`` (its
     positional encoding), and ``obj_ptr`` (object pointer).  Everything
     else -- pred_masks, pred_masks_high_res, object_score_logits, etc. --
     is deleted immediately.
  3. cached_frame_outputs is cleared after each frame.

Usage:
    python tools/run_long_video.py \
        --video_path /path/to/frames_folder_or_video.mp4 \
        --text "person" \
        --output /path/to/output.mp4
"""

import argparse
import gc
import os
import subprocess
import time

import cv2
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Lazy frame loader -- loads exactly one frame on each __getitem__ call.
# No frame is ever cached; after the model processes it, Python's refcount
# drops to zero and the tensor is freed.
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


class LazyFrameLoader:
    """Drop-in replacement for the pre-loaded image tensor.

    Supports ``__getitem__(int)`` and ``__len__`` so it can be used wherever
    ``input_batch.img_batch`` is expected.  Each access loads, resizes, and
    normalises exactly one frame, returning a ``(3, H, W)`` float16 tensor on
    CPU.  The tensor is *not* cached -- the caller (backbone) will move it to
    GPU and consume it; after that the memory is freed.
    """

    def __init__(
        self,
        resource_path: str,
        image_size: int = 1008,
        img_mean=(0.5, 0.5, 0.5),
        img_std=(0.5, 0.5, 0.5),
    ):
        self.image_size = image_size
        self.img_mean = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
        self.img_std = torch.tensor(img_std, dtype=torch.float16)[:, None, None]

        if os.path.isdir(resource_path):
            self._init_from_image_folder(resource_path)
        elif os.path.splitext(resource_path)[-1].lower() in VIDEO_EXTS:
            self._init_from_video_file(resource_path)
        else:
            raise ValueError(
                f"resource_path must be an image folder or video file, got: {resource_path}"
            )

    # -- image-folder backend ------------------------------------------------

    def _init_from_image_folder(self, folder: str):
        self._backend = "folder"
        frame_names = [
            p for p in os.listdir(folder)
            if os.path.splitext(p)[-1].lower() in IMAGE_EXTS
        ]
        try:
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        except ValueError:
            frame_names.sort()
        if not frame_names:
            raise RuntimeError(f"No images found in {folder}")
        self._img_paths = [os.path.join(folder, n) for n in frame_names]
        self._num_frames = len(self._img_paths)
        first = Image.open(self._img_paths[0])
        self.orig_width, self.orig_height = first.size

    def _load_from_folder(self, index: int) -> torch.Tensor:
        img = Image.open(self._img_paths[index]).convert("RGB")
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(img_np).permute(2, 0, 1).to(dtype=torch.float16)
        t = (t - self.img_mean) / self.img_std
        return t

    # -- video-file backend --------------------------------------------------

    def _init_from_video_file(self, path: str):
        self._backend = "video"
        self._video_path = path
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        self.orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        if self._num_frames <= 0:
            raise RuntimeError(f"Could not determine frame count for {path}")
        self._cap = None
        self._cap_pos = -1

    def _load_from_video(self, index: int) -> torch.Tensor:
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self._video_path)
            self._cap_pos = 0
        if self._cap_pos != index:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self._cap_pos = index
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {index} from {self._video_path}")
        self._cap_pos = index + 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(
            frame_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC
        )
        img_np = frame_resized.astype(np.float32) / 255.0
        t = torch.from_numpy(img_np).permute(2, 0, 1).to(dtype=torch.float16)
        t = (t - self.img_mean) / self.img_std
        return t

    # -- common interface ----------------------------------------------------

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        if index < 0 or index >= self._num_frames:
            raise IndexError(f"Frame index {index} out of range [0, {self._num_frames})")
        if self._backend == "folder":
            return self._load_from_folder(index)
        else:
            return self._load_from_video(index)

    def __len__(self):
        return self._num_frames

    @property
    def fps(self):
        if self._backend == "video":
            return self._fps
        return 30.0


# ---------------------------------------------------------------------------
# Raw frame reader -- reads the original (un-normalised) RGB frame for
# visualization overlay.
# ---------------------------------------------------------------------------

class RawFrameReader:
    """Read original-resolution RGB uint8 frames for visualization overlay."""

    def __init__(self, resource_path: str):
        if os.path.isdir(resource_path):
            self._backend = "folder"
            frame_names = [
                p for p in os.listdir(resource_path)
                if os.path.splitext(p)[-1].lower() in IMAGE_EXTS
            ]
            try:
                frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            except ValueError:
                frame_names.sort()
            self._img_paths = [os.path.join(resource_path, n) for n in frame_names]
        elif os.path.splitext(resource_path)[-1].lower() in VIDEO_EXTS:
            self._backend = "video"
            self._video_path = resource_path
            self._cap = None
            self._cap_pos = -1
        else:
            raise ValueError(f"Unsupported resource_path: {resource_path}")

    def read(self, index: int) -> np.ndarray:
        """Return an (H, W, 3) uint8 RGB numpy array for frame *index*."""
        if self._backend == "folder":
            img = cv2.imread(self._img_paths[index])
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            if self._cap is None or not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self._video_path)
                self._cap_pos = 0
            if self._cap_pos != index:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                self._cap_pos = index
            ret, frame = self._cap.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {index}")
            self._cap_pos = index + 1
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close(self):
        if self._backend == "video" and self._cap is not None:
            self._cap.release()
            self._cap = None


# ---------------------------------------------------------------------------
# Build inference_state WITHOUT loading all frames into memory.
#
# This replicates Sam3VideoInference.init_state() +
# _construct_initial_input_batch() but passes the LazyFrameLoader directly
# as img_batch, so load_resource_as_video_frames() is never called.
# ---------------------------------------------------------------------------

def _build_inference_state(model, lazy_loader: LazyFrameLoader):
    """Construct the inference_state dict that Sam3VideoInference normally
    builds inside ``init_state``, but use *lazy_loader* as the image source
    instead of loading every frame into a tensor.
    """
    from sam3.model.data_misc import BatchedDatapoint, FindStage, convert_my_tensors
    from sam3.model.geometry_encoders import Prompt
    from sam3.model.utils.misc import copy_data_to_device

    num_frames = len(lazy_loader)
    device = model.device

    # --- FindStage per frame (tiny tensors, negligible memory) ---------------
    input_box_embedding_dim = 258
    input_points_embedding_dim = 257
    stages = [
        FindStage(
            img_ids=[stage_id],
            text_ids=[0],
            input_boxes=[torch.zeros(input_box_embedding_dim)],
            input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
            input_boxes_label=[torch.empty(0, dtype=torch.long)],
            input_points=[torch.empty(0, input_points_embedding_dim)],
            input_points_mask=[torch.empty(0)],
            object_ids=[],
        )
        for stage_id in range(num_frames)
    ]
    for i in range(len(stages)):
        stages[i] = convert_my_tensors(stages[i])

    # --- BatchedDatapoint with lazy_loader as img_batch ----------------------
    # copy_data_to_device recurses into the dataclass.  For the LazyFrameLoader
    # it falls through to ``return data`` (not a tensor/list/dict/dataclass),
    # so it stays on CPU -- exactly what we want.  The backbone will call
    # ``img_batch[frame_idx]`` and ``.to(device)`` on a per-frame basis.
    input_batch = BatchedDatapoint(
        img_batch=lazy_loader,
        find_text_batch=["<text placeholder>", "visual"],
        find_inputs=stages,
        find_targets=[None] * num_frames,
        find_metadatas=[None] * num_frames,
    )
    input_batch = copy_data_to_device(input_batch, device, non_blocking=True)

    # --- Assemble the full inference_state ----------------------------------
    bs = 1
    inference_state = {
        "image_size": model.image_size,
        "num_frames": num_frames,
        "orig_height": lazy_loader.orig_height,
        "orig_width": lazy_loader.orig_width,
        "constants": {
            "empty_geometric_prompt": Prompt(
                box_embeddings=torch.zeros(0, bs, 4, device=device),
                box_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
                box_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
                point_embeddings=torch.zeros(0, bs, 2, device=device),
                point_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
                point_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
            ),
        },
        "input_batch": input_batch,
        "previous_stages_out": [None] * num_frames,
        "text_prompt": None,
        "per_frame_raw_point_input": [None] * num_frames,
        "per_frame_raw_box_input": [None] * num_frames,
        "per_frame_visual_prompt": [None] * num_frames,
        "per_frame_geometric_prompt": [None] * num_frames,
        "per_frame_cur_step": [0] * num_frames,
        "visual_prompt_embed": None,
        "visual_prompt_mask": None,
        "tracker_inference_states": [],
        "tracker_metadata": {},
        "feature_cache": {},
        "cached_frame_outputs": {},
        "action_history": [],
        "is_image_only": False,
    }
    return inference_state


# ---------------------------------------------------------------------------
# Render a single overlay frame -- masks + bounding boxes + labels
# ---------------------------------------------------------------------------

_COLOR_CACHE = None


def _get_colors():
    global _COLOR_CACHE
    if _COLOR_CACHE is not None:
        return _COLOR_CACHE
    try:
        from sam3.visualization_utils import COLORS
        _COLOR_CACHE = COLORS
    except ImportError:
        rng = np.random.RandomState(42)
        _COLOR_CACHE = rng.rand(128, 3).astype(np.float64)
    return _COLOR_CACHE


def render_overlay(img_rgb: np.ndarray, outputs: dict, frame_idx: int,
                   alpha: float = 0.45) -> np.ndarray:
    """Overlay masks, boxes and labels on a raw RGB frame."""
    colors = _get_colors()
    overlay = img_rgb.copy()
    h, w = overlay.shape[:2]

    obj_ids = outputs.get("out_obj_ids")
    masks = outputs.get("out_binary_masks")
    boxes = outputs.get("out_boxes_xywh")
    probs = outputs.get("out_probs")

    if obj_ids is not None and len(obj_ids) > 0:
        for i, oid in enumerate(obj_ids):
            color = colors[int(oid) % len(colors)]
            c_uint8 = (color * 255).astype(np.uint8)

            mask = masks[i]
            if mask.shape != (h, w):
                mask = cv2.resize(
                    mask.astype(np.float32), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0.5
            m = mask > 0
            for c in range(3):
                overlay[..., c][m] = (
                    alpha * c_uint8[c] + (1 - alpha) * overlay[..., c][m]
                ).astype(np.uint8)

            if boxes is not None:
                bx, by, bw, bh = boxes[i]
                x1, y1 = int(bx * w), int(by * h)
                x2, y2 = int((bx + bw) * w), int((by + bh) * h)
                c_bgr = tuple(int(x) for x in c_uint8)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), c_bgr, 2)
                prob = probs[i] if probs is not None else None
                label = f"id={int(oid)}"
                if prob is not None:
                    label += f" {prob:.2f}"
                cv2.putText(
                    overlay, label, (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_bgr, 2, cv2.LINE_AA,
                )

    cv2.putText(
        overlay, f"Frame {frame_idx}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return overlay


# ---------------------------------------------------------------------------
# Trim tracker output_dict to keep only maskmem_features + obj_ptr
# ---------------------------------------------------------------------------

def _trim_tracker_output_dict(tracker_state):
    """Remove all heavy tensors from the tracker's output_dict except the
    keys the tracker reads from past frames during propagation:

    - ``maskmem_features``  -- spatial memory fed into memory attention
    - ``maskmem_pos_enc``   -- positional encoding for the spatial memory
    - ``obj_ptr``           -- object pointer token for cross-attention
    - ``eff_iou_score``     -- memory-selection score (used by frame_filter
                               when use_memory_selection=True to pick the
                               most informative pointers from a large window)

    Also trims ``output_dict_per_obj`` which stores per-object slices of
    the same data -- without trimming it accumulates unboundedly.
    """
    keys_to_keep = {"maskmem_features", "maskmem_pos_enc", "obj_ptr", "eff_iou_score"}

    # --- main output_dict ---
    output_dict = tracker_state.get("output_dict")
    if output_dict is not None:
        for bucket_name in ("cond_frame_outputs", "non_cond_frame_outputs"):
            bucket = output_dict.get(bucket_name, {})
            for frame_out in bucket.values():
                if frame_out is None:
                    continue
                for k in [k for k in frame_out if k not in keys_to_keep]:
                    del frame_out[k]

    # --- per-object sliced output dicts ---
    # _add_output_per_object stores per-object slices of every frame's output
    # in output_dict_per_obj.  Without trimming these grow without bound.
    output_dict_per_obj = tracker_state.get("output_dict_per_obj", {})
    for obj_output_dict in output_dict_per_obj.values():
        for bucket_name in ("cond_frame_outputs", "non_cond_frame_outputs"):
            bucket = obj_output_dict.get(bucket_name, {})
            for frame_out in bucket.values():
                if frame_out is None:
                    continue
                for k in [k for k in frame_out if k not in keys_to_keep]:
                    del frame_out[k]


# ---------------------------------------------------------------------------
# Consolidate multiple tracker states into a single state.
#
# When objects are detected on different frames, SAM3 creates separate
# tracker states.  Each state runs a full propagation per frame, so
# N states = N× tracker cost.  Consolidation merges all objects into one
# state so the tracker runs only once (batched) per frame.
#
# The spatial memories (maskmem_features) from each old state are merged
# into the new state's output_dict so the tracker has full memory context.
# For objects that didn't exist at a past frame, no_obj_embed_spatial is
# used as a placeholder (same as what SAM3 uses for occluded objects).
# ---------------------------------------------------------------------------

def _consolidate_tracker_states(model, inference_state, frame_idx):
    """Merge all tracker states into a single state, preserving spatial memories.

    For each past frame in the memory window:
      - Collects maskmem_features, maskmem_pos_enc, obj_ptr from each old state
      - Builds merged tensors with batch_dim = total_objects
      - Fills no_obj_embed_spatial / no_obj_ptr for objects not yet present

    This turns N sequential tracker propagations into 1 batched one.
    """
    import torch.nn.functional as F

    tracker_states = inference_state["tracker_inference_states"]
    if len(tracker_states) <= 1:
        return  # nothing to consolidate

    # --- collect all obj_ids, index mapping, and first-appeared frames ---
    # obj_order[i] = (state_idx, position_within_that_state)
    all_obj_ids = []
    obj_order = []
    obj_first_appeared = {}  # obj_id -> earliest frame in its state's output_dict
    for state_idx, state in enumerate(tracker_states):
        state_obj_ids = state.get("obj_ids", [])
        # Find the earliest frame in this state's output_dict
        od = state.get("output_dict", {})
        earliest = frame_idx
        for bucket in ("cond_frame_outputs", "non_cond_frame_outputs"):
            for fidx in od.get(bucket, {}).keys():
                earliest = min(earliest, fidx)
        # Also check if tracker already has _obj_first_appeared_frame from
        # a prior consolidation
        prior_first = getattr(model.tracker, "_obj_first_appeared_frame", {})
        for pos, oid in enumerate(state_obj_ids):
            all_obj_ids.append(oid)
            obj_order.append((state_idx, pos))
            obj_first_appeared[oid] = prior_first.get(oid, earliest)
    total_objs = len(all_obj_ids)
    if total_objs == 0:
        return

    # --- get current masks from cached_frame_outputs ---
    cached = inference_state["cached_frame_outputs"].get(frame_idx, {})
    if len(cached) == 0:
        return

    orig_h = inference_state["orig_height"]
    orig_w = inference_state["orig_width"]
    feature_cache = inference_state["feature_cache"]
    num_frames = inference_state["num_frames"]

    # --- create a fresh tracker state ---
    # We DON'T use add_new_mask (which needs backbone features that may be
    # evicted).  Instead we manually register objects and directly merge
    # all old states' output_dicts into the new state.
    new_state = model.tracker.init_state(
        cached_features=feature_cache,
        video_height=orig_h,
        video_width=orig_w,
        num_frames=num_frames,
    )
    new_state["backbone_out"] = tracker_states[0].get("backbone_out", None)

    # --- manually register all objects (before tracking_has_started) ---
    from collections import OrderedDict
    new_state["obj_id_to_idx"] = OrderedDict()
    new_state["obj_idx_to_id"] = OrderedDict()
    new_state["obj_ids"] = []
    new_state["point_inputs_per_obj"] = {}
    new_state["mask_inputs_per_obj"] = {}
    new_state["output_dict_per_obj"] = {}
    new_state["temp_output_dict_per_obj"] = {}
    for idx, obj_id in enumerate(all_obj_ids):
        new_state["obj_id_to_idx"][obj_id] = idx
        new_state["obj_idx_to_id"][idx] = obj_id
        new_state["point_inputs_per_obj"][idx] = {}
        new_state["mask_inputs_per_obj"][idx] = {}
        new_state["output_dict_per_obj"][idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        new_state["temp_output_dict_per_obj"][idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
    new_state["obj_ids"] = list(all_obj_ids)
    new_state["tracking_has_started"] = True

    # --- merge ALL frame outputs from old states into the new state ---------
    device = model.device
    no_obj_spatial = model.tracker.no_obj_embed_spatial  # (1, mem_dim)
    no_obj_ptr = model.tracker.no_obj_ptr                # (1, hidden_dim)
    new_output_dict = new_state["output_dict"]

    all_frame_indices = set()
    for state in tracker_states:
        od = state.get("output_dict", {})
        for bucket in ("cond_frame_outputs", "non_cond_frame_outputs"):
            all_frame_indices.update(od.get(bucket, {}).keys())

    n_merged = 0
    for past_fidx in sorted(all_frame_indices):
        # For each past frame, build merged tensors across all old states
        # First, find one state that has maskmem_features to get the shape
        ref_feats = None
        ref_ptr = None
        for state in tracker_states:
            od = state.get("output_dict", {})
            for bucket in ("cond_frame_outputs", "non_cond_frame_outputs"):
                out = od.get(bucket, {}).get(past_fidx)
                if out is not None:
                    if ref_feats is None and out.get("maskmem_features") is not None:
                        ref_feats = out["maskmem_features"]
                    if ref_ptr is None and out.get("obj_ptr") is not None:
                        ref_ptr = out["obj_ptr"]
            if ref_feats is not None and ref_ptr is not None:
                break

        if ref_feats is None:
            continue  # no spatial memory at this frame, skip

        # Build merged maskmem_features: (total_objs, mem_dim, H, W)
        mem_dim = ref_feats.shape[1]
        H_mem, W_mem = ref_feats.shape[2], ref_feats.shape[3]
        merged_feats = no_obj_spatial[..., None, None].expand(
            total_objs, mem_dim, H_mem, W_mem
        ).clone().to(device=ref_feats.device, dtype=ref_feats.dtype)

        # Build merged obj_ptr: (total_objs, hidden_dim)
        hidden_dim = ref_ptr.shape[1] if ref_ptr is not None else model.tracker.hidden_dim
        merged_ptr = no_obj_ptr.expand(total_objs, hidden_dim).clone().to(
            device=device
        )

        # Fill in each object's slice from its original state
        for merged_idx, (state_idx, pos_in_state) in enumerate(obj_order):
            state = tracker_states[state_idx]
            od = state.get("output_dict", {})
            out = None
            for bucket in ("cond_frame_outputs", "non_cond_frame_outputs"):
                out = od.get(bucket, {}).get(past_fidx)
                if out is not None:
                    break
            if out is None:
                continue  # this state had no output at this frame

            feats = out.get("maskmem_features")
            if feats is not None and pos_in_state < feats.shape[0]:
                merged_feats[merged_idx] = feats[pos_in_state].to(
                    device=merged_feats.device
                )

            ptr = out.get("obj_ptr")
            if ptr is not None and pos_in_state < ptr.shape[0]:
                merged_ptr[merged_idx] = ptr[pos_in_state].to(device=device)

        # Get maskmem_pos_enc -- it's the same across all frames/objects.
        # Try the new state's constants first, then old states' constants.
        pos_enc = new_state["constants"].get("maskmem_pos_enc")
        if pos_enc is None:
            for state in tracker_states:
                pos_enc = state.get("constants", {}).get("maskmem_pos_enc")
                if pos_enc is not None:
                    new_state["constants"]["maskmem_pos_enc"] = pos_enc
                    break
        if pos_enc is not None:
            merged_pos_enc = [x.expand(total_objs, -1, -1, -1) for x in pos_enc]
        else:
            merged_pos_enc = None

        # Determine if this frame was a conditioning frame in any old state
        is_cond = any(
            past_fidx in state.get("output_dict", {}).get("cond_frame_outputs", {})
            for state in tracker_states
        )
        bucket_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Also merge pred_masks and object_score_logits (needed by tracker
        # for conditioning frame lookups and _run_single_frame_inference)
        merged_pred_masks = None
        merged_obj_score = None
        for merged_idx2, (state_idx2, pos2) in enumerate(obj_order):
            state2 = tracker_states[state_idx2]
            od2 = state2.get("output_dict", {})
            out2 = None
            for b2 in ("cond_frame_outputs", "non_cond_frame_outputs"):
                out2 = od2.get(b2, {}).get(past_fidx)
                if out2 is not None:
                    break
            if out2 is None:
                continue
            pm = out2.get("pred_masks")
            if pm is not None and pos2 < pm.shape[0]:
                if merged_pred_masks is None:
                    # Initialize with zeros
                    merged_pred_masks = torch.zeros(
                        total_objs, *pm.shape[1:],
                        device=pm.device, dtype=pm.dtype
                    )
                merged_pred_masks[merged_idx2] = pm[pos2]
            osl = out2.get("object_score_logits")
            if osl is not None and pos2 < osl.shape[0]:
                if merged_obj_score is None:
                    merged_obj_score = torch.zeros(
                        total_objs, *osl.shape[1:],
                        device=osl.device, dtype=osl.dtype
                    )
                merged_obj_score[merged_idx2] = osl[pos2]

        # Store the merged output
        merged_out = {
            "maskmem_features": merged_feats,
            "maskmem_pos_enc": merged_pos_enc,
            "obj_ptr": merged_ptr,
        }
        if merged_pred_masks is not None:
            merged_out["pred_masks"] = merged_pred_masks
        if merged_obj_score is not None:
            merged_out["object_score_logits"] = merged_obj_score
        # Collect eff_iou_score if available (take max across states)
        eff_scores = []
        for state in tracker_states:
            od = state.get("output_dict", {})
            for bucket in ("cond_frame_outputs", "non_cond_frame_outputs"):
                out = od.get(bucket, {}).get(past_fidx)
                if out is not None and "eff_iou_score" in out:
                    eff_scores.append(out["eff_iou_score"])
        if eff_scores:
            merged_out["eff_iou_score"] = max(
                eff_scores, key=lambda x: x.item() if hasattr(x, "item") else x
            )

        new_output_dict[bucket_key][past_fidx] = merged_out
        if is_cond:
            new_state["consolidated_frame_inds"]["cond_frame_outputs"].add(past_fidx)
        n_merged += 1

    # --- also build merged output_dict_per_obj for the new state ---
    # The per-obj dicts reference slices of the main output_dict, so after
    # merging we need to rebuild them.
    new_per_obj = new_state.get("output_dict_per_obj", {})
    for merged_idx, obj_id in enumerate(all_obj_ids):
        obj_idx = new_state["obj_id_to_idx"].get(obj_id)
        if obj_idx is None:
            continue
        obj_out_dict = new_per_obj.get(obj_idx, {})
        for bucket in ("cond_frame_outputs", "non_cond_frame_outputs"):
            main_bucket = new_output_dict.get(bucket, {})
            obj_bucket = obj_out_dict.get(bucket, {})
            for fidx, main_out in main_bucket.items():
                obj_slice = slice(merged_idx, merged_idx + 1)
                obj_out = {}
                for k, v in main_out.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 1:
                        obj_out[k] = v[obj_slice]
                    elif isinstance(v, list):
                        obj_out[k] = [x[obj_slice] if isinstance(x, torch.Tensor) else x for x in v]
                    else:
                        obj_out[k] = v
                obj_bucket[fidx] = obj_out

    # --- store first-appeared info on the tracker for attention masking ---
    model.tracker._obj_first_appeared_frame = obj_first_appeared
    model.tracker._current_inference_state = new_state

    # --- mark all merged frames as tracked ---
    for fidx in all_frame_indices:
        new_state["frames_already_tracked"][fidx] = {"reverse": False}

    # --- replace old states ---
    n_old = len(tracker_states)
    n_objs = len(new_state.get("obj_ids", []))
    tracker_states.clear()
    tracker_states.append(new_state)
    print(
        f"    [consolidate] frame {frame_idx}: "
        f"merged {n_old} states -> 1 ({n_objs} objs, {n_merged} memory frames transferred)"
    )


# ---------------------------------------------------------------------------
# Option B: per-object attention masking for non-existent frames.
#
# Instead of filling no_obj_embed_spatial for objects that didn't exist at a
# past frame, set the attention mask to True (= exclude) for those objects on
# that frame's spatial tokens.  This way the cross-attention never sees dummy
# data -- it just skips those tokens entirely for that object.
#
# Enabled with --use_obj_mask.  When disabled, the original no_obj_embed_spatial
# approach is used (Option A).
# ---------------------------------------------------------------------------

def _install_obj_attention_mask(model):
    """Monkey-patch the tracker's _prepare_memory_conditioned_features to build
    a per-object attention mask based on ``_obj_first_appeared_frame``.

    After patching, the tracker will:
    1. Check each object's first-appeared frame from the consolidated state
    2. For spatial memory frames before an object appeared, set that object's
       mask to True (excluded from attention)
    3. Same for obj_ptr tokens from frames before the object appeared
    4. Pass the real mask to the encoder instead of None
    """
    import types

    tracker = model.tracker
    _orig_fn = tracker._prepare_memory_conditioned_features

    @torch.inference_mode()
    def _patched_prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
        use_prev_mem_frame=True,
    ):
        # Get the inference state that holds _obj_first_appeared_frame.
        # We stash it on self during consolidation.
        first_appeared = getattr(self, "_obj_first_appeared_frame", None)
        if first_appeared is None:
            # No consolidation happened yet -- use original path
            return _orig_fn(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                use_prev_mem_frame=use_prev_mem_frame,
            )

        # --- Run the original function up to the point where it builds
        #     to_cat_prompt / to_cat_prompt_mask, then fix the mask. ---
        # We call the original which sets prompt_mask=None, then re-run
        # just the mask construction.  Actually, the cleanest approach is
        # to let the original build everything, then patch the mask before
        # it's passed to the encoder.  But since the original has the
        # encoder call inside it, we need to intercept at the right spot.
        #
        # The simplest reliable approach: temporarily replace
        # self.transformer.encoder with a wrapper that intercepts the mask.

        B = current_vision_feats[-1].size(1)  # batch size = num objects
        device = current_vision_feats[-1].device

        # Build the first_appeared lookup for batch indices
        # obj_ids in the current state tell us the order
        _current_state = getattr(self, "_current_inference_state", None)
        if _current_state is None:
            return _orig_fn(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                use_prev_mem_frame=use_prev_mem_frame,
            )

        obj_ids = _current_state.get("obj_ids", [])
        # first_frame_per_batch[i] = first appeared frame for batch index i
        first_frame_per_batch = []
        for oid in obj_ids:
            first_frame_per_batch.append(first_appeared.get(oid, 0))

        # Intercept the encoder call to inject the mask
        real_encoder = self.transformer.encoder
        spatial_frames_info = []  # will be filled by the interceptor

        class _EncoderWithMask:
            """Wrapper that intercepts the encoder call to inject a real
            prompt_key_padding_mask based on per-object first-appeared frame."""

            def __init__(self, real_enc, first_frames, batch_size, device):
                self.real_enc = real_enc
                self.first_frames = first_frames  # list[int], len=B
                self.B = batch_size
                self.device = device
                # Forward any attribute access to the real encoder
                for attr in dir(real_enc):
                    if not attr.startswith("_") and not hasattr(self, attr):
                        try:
                            setattr(self, attr, getattr(real_enc, attr))
                        except Exception:
                            pass

            def __call__(self, *args, **kwargs):
                # Build the per-object mask from spatial_frames_info
                # spatial_frames_info is populated by the patched
                # _prepare_memory_conditioned_features above via the
                # to_cat_prompt_mask list.  But we actually need the frame
                # indices for each spatial chunk.
                #
                # The prompt has structure:
                #   [spatial_chunk_0, spatial_chunk_1, ..., obj_ptr_chunk]
                # We need to know which frame each spatial chunk came from.
                #
                # Since we can't easily extract this from the original fn,
                # we take a different approach: read the frame info that
                # was stashed on the tracker by the consolidation code.
                spatial_info = getattr(self.real_enc, "_spatial_frames_info", None)
                if spatial_info is not None and len(spatial_info) > 0:
                    prompt = kwargs.get("prompt", args[5] if len(args) > 5 else None)
                    if prompt is not None:
                        total_len = prompt.shape[0]
                        # Build mask: (B, total_len), True = exclude
                        mask = torch.zeros(self.B, total_len, device=self.device, dtype=torch.bool)
                        offset = 0
                        for (mem_frame_idx, seq_len) in spatial_info:
                            for obj_batch_idx, first_f in enumerate(self.first_frames):
                                if mem_frame_idx < first_f:
                                    mask[obj_batch_idx, offset:offset + seq_len] = True
                            offset += seq_len
                        # Also mask obj_ptr tokens (they come after spatial tokens)
                        ptr_info = getattr(self.real_enc, "_ptr_frames_info", None)
                        if ptr_info is not None:
                            for (ptr_frame_idx, ptr_len) in ptr_info:
                                for obj_batch_idx, first_f in enumerate(self.first_frames):
                                    if ptr_frame_idx < first_f:
                                        mask[obj_batch_idx, offset:offset + ptr_len] = True
                                offset += ptr_len
                        # Only use mask if any entries are True
                        if mask.any():
                            kwargs["prompt_key_padding_mask"] = mask
                # Clean up
                self.real_enc._spatial_frames_info = None
                self.real_enc._ptr_frames_info = None
                return self.real_enc(*args, **kwargs)

        # Now we need to stash frame info during the original fn's execution.
        # We do this by patching to_cat_prompt_mask collection.
        # But that's deep inside _prepare_memory_conditioned_features...
        #
        # Cleaner approach: we know the structure of the original function.
        # After it builds to_cat_prompt, each spatial entry corresponds to a
        # memory frame.  The frame indices follow a known pattern from
        # selected_cond_outputs + the t_pos loop.
        #
        # Let's just record which frames are accessed by wrapping
        # output_dict access.

        class _OutputDictTracker:
            """Wraps output_dict to record which frame indices are accessed
            for spatial memory and obj_ptr."""
            def __init__(self, real_dict):
                self._real = real_dict
                self.spatial_accessed = []  # (frame_idx, is_cond)
                self.ptr_accessed = []      # (frame_idx,)

            def __getitem__(self, key):
                return self._real[key]

            def __contains__(self, key):
                return key in self._real

            def get(self, key, default=None):
                return self._real.get(key, default)

            def keys(self):
                return self._real.keys()

            def items(self):
                return self._real.items()

            def values(self):
                return self._real.values()

            def __len__(self):
                return len(self._real)

        # This approach is getting too complex with monkey-patching internals.
        # Let's use a simpler and more robust approach: wrap the encoder.
        # Since the original _prepare_memory_conditioned_features already
        # builds to_cat_prompt with known structure, we can compute the mask
        # from the output_dict frame indices BEFORE calling the original fn.

        # Gather which frames have spatial memory in output_dict
        spatial_frame_indices = []
        H_mem, W_mem = feat_sizes[-1]
        seq_len_per_frame = H_mem * W_mem

        # Replicate the frame selection logic from the original function
        from sam3.model.sam3_tracker_utils import select_closest_cond_frames
        cond_outputs = output_dict["cond_frame_outputs"]
        selected_cond, unselected_cond = select_closest_cond_frames(
            frame_idx, cond_outputs,
            self.max_cond_frames_in_attn,
            keep_first_cond_frame=self.keep_first_cond_frame,
        )
        # Conditioning frames (spatial memory)
        for t, out in selected_cond.items():
            if out is not None and out.get("maskmem_features") is not None:
                spatial_frame_indices.append(t)

        # Non-conditioning spatial frames
        r = self.memory_temporal_stride_for_eval
        if self.use_memory_selection:
            valid_indices = self.frame_filter(
                output_dict, track_in_reverse, frame_idx, num_frames, r
            )
        for t_pos in range(1, self.num_maskmem):
            t_rel = self.num_maskmem - t_pos
            if self.use_memory_selection:
                if t_rel > len(valid_indices):
                    continue
                prev_fidx = valid_indices[-t_rel]
            else:
                if t_rel == 1:
                    prev_fidx = frame_idx - t_rel if not track_in_reverse else frame_idx + t_rel
                else:
                    if not track_in_reverse:
                        prev_fidx = ((frame_idx - 2) // r) * r - (t_rel - 2) * r
                    else:
                        prev_fidx = -(-(frame_idx + 2) // r) * r + (t_rel - 2) * r
            out = output_dict["non_cond_frame_outputs"].get(prev_fidx)
            if out is None:
                out = unselected_cond.get(prev_fidx)
            if out is not None and out.get("maskmem_features") is not None:
                spatial_frame_indices.append(prev_fidx)

        # Obj_ptr frame indices
        ptr_frame_indices = []
        tpos_sign_mul = -1 if track_in_reverse else 1
        max_obj_ptrs = min(num_frames, self.max_obj_ptrs_in_encoder)
        # conditioning frame pointers
        if not self.training:
            ptr_cond = {t: out for t, out in selected_cond.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)}
        else:
            ptr_cond = selected_cond
        for t in ptr_cond:
            ptr_frame_indices.append(t)
        # non-conditioning pointers
        for t_diff in range(1, max_obj_ptrs):
            if not self.use_memory_selection:
                t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                if t < 0 or (num_frames is not None and t >= num_frames):
                    break
            else:
                if -t_diff <= -len(valid_indices):
                    break
                t = valid_indices[-t_diff]
            out = output_dict["non_cond_frame_outputs"].get(t, unselected_cond.get(t))
            if out is not None:
                ptr_frame_indices.append(t)

        # Stash frame info on the encoder so the wrapper can build the mask
        real_encoder._spatial_frames_info = [
            (fidx, seq_len_per_frame) for fidx in spatial_frame_indices
        ]
        # Each obj_ptr token has 1 token per frame (or C//mem_dim if split)
        ptr_tokens_per_frame = 1
        if self.mem_dim < self.hidden_dim:
            ptr_tokens_per_frame = self.hidden_dim // self.mem_dim
        real_encoder._ptr_frames_info = [
            (fidx, ptr_tokens_per_frame) for fidx in ptr_frame_indices
        ]

        # Temporarily swap encoder with mask-aware wrapper
        self.transformer.encoder = _EncoderWithMask(
            real_encoder, first_frame_per_batch, B, device
        )
        try:
            result = _orig_fn(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                use_prev_mem_frame=use_prev_mem_frame,
            )
        finally:
            # Restore the real encoder
            self.transformer.encoder = real_encoder
        return result

    tracker._prepare_memory_conditioned_features = types.MethodType(
        _patched_prepare_memory_conditioned_features, tracker
    )
    print("  [obj_mask] Installed per-object attention masking on tracker")


# ---------------------------------------------------------------------------
# Main: memory-efficient propagation -> visualization video
# ---------------------------------------------------------------------------

def run_long_video(
    video_path: str,
    text_prompt: str,
    output_path: str = "output.mp4",
    gpus_to_use=None,
    fps: float | None = None,
    gc_every: int = 50,
    max_obj_ptrs: int = 32,
    use_obj_mask: bool = False,
):
    """Run SAM3 on a long video with constant memory, write a visualization MP4.

    Args:
        video_path:   Path to a folder of JPEG frames or a video file.
        text_prompt:  Text describing the objects to track (e.g. "person").
        output_path:  Path for the output visualization video (.mp4).
        gpus_to_use:  List of GPU ids; defaults to current device only.
        fps:          Output video FPS.  Defaults to the source video's FPS
                      (or 30 for image folders).
        gc_every:     Run gc.collect + cuda.empty_cache every N frames.
        max_obj_ptrs: Maximum number of past-frame object pointers the tracker
                      attends to.  SAM3 default is 16 (~0.5 s at 30 fps).
                      32 ≈ 1 s, 128 ≈ 4 s, 900 ≈ 30 s.  Each pointer is a
                      small vector (~2 KB) so the memory cost is negligible,
                      but higher values make the Python-level frame_filter
                      loop iterate over more frames.
        use_obj_mask: If True, use per-object attention masking (Option B):
                      exclude spatial memory tokens for frames where an object
                      didn't exist yet, instead of feeding no_obj_embed_spatial.
                      If False (default), use the original no_obj_embed_spatial
                      placeholder approach (Option A).
    """
    from sam3.model_builder import build_sam3_video_predictor

    if gpus_to_use is None:
        gpus_to_use = [torch.cuda.current_device()]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # ---- 1. Build predictor (loads model weights) --------------------------
    print("Loading SAM3 model ...")
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
    model = predictor.model  # Sam3VideoInference

    # ---- 1b. Increase obj_ptr memory window for long-range re-id -----------
    #  Default max_obj_ptrs_in_encoder=16 only looks ~0.5 s back at 30 fps.
    #  Each obj_ptr is a single vector (hidden_dim floats ≈ 2 KB) so even
    #  hundreds of pointers add negligible memory.  The temporal positional
    #  encoding is sinusoidal and generalises to any range.
    prev_ptrs = model.tracker.max_obj_ptrs_in_encoder
    model.tracker.max_obj_ptrs_in_encoder = max_obj_ptrs
    print(
        f"obj_ptr window: {prev_ptrs} -> {max_obj_ptrs} frames "
        f"(~{max_obj_ptrs / 30:.1f}s at 30 fps)"
    )

    # ---- 1c. Optionally install per-object attention masking ---------------
    if use_obj_mask:
        _install_obj_attention_mask(model)

    # ---- 2. Build lazy frame loader ----------------------------------------
    image_size = model.image_size
    loader = LazyFrameLoader(
        video_path,
        image_size=image_size,
        img_mean=model.image_mean,
        img_std=model.image_std,
    )
    num_frames = len(loader)
    orig_h, orig_w = loader.orig_height, loader.orig_width
    if fps is None:
        fps = loader.fps
    print(f"Video: {num_frames} frames, {orig_w}x{orig_h} @ {fps:.1f} fps")

    # ---- 3. Raw frame reader for visualization -----------------------------
    raw_reader = RawFrameReader(video_path)

    # ---- 4. Build inference_state directly (NO init_state call) -------------
    #  This avoids load_resource_as_video_frames which would load every frame
    #  into a CPU tensor and explode memory.
    print("Building inference state (lazy -- no frames loaded) ...")
    inference_state = _build_inference_state(model, loader)

    # ---- 5. Add text prompt on frame 0 -------------------------------------
    print(f"Adding text prompt: '{text_prompt}' on frame 0 ...")
    session_id = "lazy_session"
    predictor._all_inference_states[session_id] = {
        "state": inference_state,
        "session_id": session_id,
        "start_time": time.time(),
        "last_use_time": time.time(),
    }
    predictor.handle_request(
        dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=text_prompt,
        )
    )

    # ---- 6. Open video writer -----------------------------------------------
    tmp_path = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {tmp_path}")

    # ---- 7. Propagate, render each frame, write to video -------------------
    print("Propagating and rendering ...")
    t0 = time.time()
    t_model_total = 0.0  # time inside model (between yields)
    t_vis_total = 0.0    # time for reading raw frame + rendering overlay
    t_trim_total = 0.0   # time for output trimming + GC
    t_yield_start = time.time()

    for response in predictor.handle_stream_request(
        dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction="forward",
        )
    ):
        t_model_end = time.time()
        t_model_total += t_model_end - t_yield_start

        frame_idx = response["frame_index"]
        outputs = response["outputs"]

        # -- read raw frame & render overlay ---------------------------------
        t_vis_start = time.time()
        raw_frame = raw_reader.read(frame_idx)  # (H, W, 3) uint8 RGB
        if outputs is not None:
            overlay = render_overlay(raw_frame, outputs, frame_idx)
        else:
            overlay = raw_frame.copy()
            cv2.putText(
                overlay, f"Frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
            )
        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        t_vis_total += time.time() - t_vis_start

        # -- consolidate tracker states if >1 (before trimming clears masks) --
        t_trim_start = time.time()
        if len(inference_state["tracker_inference_states"]) > 1:
            _consolidate_tracker_states(model, inference_state, frame_idx)

        # -- trim tracker memory to keep only spatial + obj_ptr --------------
        for tracker_state in inference_state["tracker_inference_states"]:
            _trim_tracker_output_dict(tracker_state)

        # -- clear cached frame outputs (full-res masks we no longer need) ---
        inference_state["cached_frame_outputs"].pop(frame_idx, None)

        # -- periodic GC + CUDA cache flush ----------------------------------
        if frame_idx % gc_every == 0 and frame_idx > 0:
            gc.collect()
            torch.cuda.empty_cache()
        t_trim_total += time.time() - t_trim_start

        # -- per-frame timing report -----------------------------------------
        if frame_idx > 0 and frame_idx % 50 == 0:
            n = frame_idx + 1
            mem_mb = torch.cuda.memory_allocated() // (1024 * 1024)
            n_states = len(inference_state["tracker_inference_states"])
            n_objs = sum(
                len(s.get("obj_ids", []))
                for s in inference_state["tracker_inference_states"]
            )
            elapsed = time.time() - t0
            print(
                f"  frame {frame_idx}/{num_frames}  |  "
                f"model {t_model_total/n*1000:.0f}ms  "
                f"vis {t_vis_total/n*1000:.0f}ms  "
                f"trim {t_trim_total/n*1000:.0f}ms  |  "
                f"{n/elapsed:.1f} fps  |  "
                f"GPU {mem_mb}MiB  "
                f"states={n_states} objs={n_objs}"
            )

        t_yield_start = time.time()

    writer.release()
    raw_reader.close()
    elapsed = time.time() - t0
    print(
        f"Inference done: {num_frames} frames in {elapsed:.1f}s "
        f"({num_frames / elapsed:.1f} fps)"
    )

    # ---- 8. Re-encode with ffmpeg for broad compatibility -------------------
    print("Re-encoding video with ffmpeg ...")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", output_path],
            check=True, capture_output=True,
        )
        os.remove(tmp_path)
    except (subprocess.CalledProcessError, FileNotFoundError):
        os.rename(tmp_path, output_path)
        print("  (ffmpeg not available, using raw mp4v output)")

    print(f"Saved visualization video: {output_path}")

    # ---- 9. Cleanup --------------------------------------------------------
    predictor.handle_request(
        dict(type="close_session", session_id=session_id)
    )
    predictor.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3 on a long video with constant memory. "
                    "Outputs a visualization MP4 with mask overlays."
    )
    parser.add_argument(
        "--video_path", required=True,
        help="Path to a folder of JPEG frames or a video file.",
    )
    parser.add_argument(
        "--text", required=True,
        help="Text prompt describing the objects to track (e.g. 'person').",
    )
    parser.add_argument(
        "--output", default="output.mp4",
        help="Output visualization video path (default: output.mp4).",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Output video FPS. Defaults to source video's FPS (or 30).",
    )
    parser.add_argument(
        "--gc_every", type=int, default=50,
        help="Run gc.collect + cuda.empty_cache every N frames (default: 50).",
    )
    parser.add_argument(
        "--gpus", type=int, nargs="*", default=None,
        help="GPU IDs to use (default: current device only).",
    )
    parser.add_argument(
        "--max_obj_ptrs", type=int, default=32,
        help="Max object-pointer memory window (default: 32 ≈ 1s at 30fps). "
             "Controls how far back the tracker looks for re-identification. "
             "SAM3 default is 16. Higher = better re-id but slightly more "
             "Python overhead per frame. 128 ≈ 4s, 900 ≈ 30s.",
    )
    parser.add_argument(
        "--use_obj_mask", action="store_true", default=False,
        help="Use per-object attention masking (Option B). Excludes spatial "
             "memory tokens from cross-attention for frames where an object "
             "didn't exist yet, instead of feeding no_obj_embed_spatial "
             "placeholders. Off by default (Option A).",
    )
    args = parser.parse_args()

    run_long_video(
        video_path=args.video_path,
        text_prompt=args.text,
        output_path=args.output,
        gpus_to_use=args.gpus,
        fps=args.fps,
        gc_every=args.gc_every,
        max_obj_ptrs=args.max_obj_ptrs,
        use_obj_mask=args.use_obj_mask,
    )


if __name__ == "__main__":
    main()
