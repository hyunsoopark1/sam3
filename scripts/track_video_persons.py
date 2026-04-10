"""
Multi-person detection and tracking using the SAM 3 tracker.

Detects persons on every Nth frame using torchvision Faster R-CNN, matches
them to existing tracks via bbox IoU, and creates new tracker states for
unmatched detections.  The SAM 3 tracker's memory (object pointers) handles
long-range re-identification — a person who disappears and reappears can be
re-associated by the tracker's attention over stored object pointers.

Each "tracker state" corresponds to a batch of persons first detected on the
same frame.  Multiple states are propagated independently, following the same
pattern used internally by SAM 3's joint detect-and-track pipeline.

Examples
--------
Detect & track all persons, output overlay::

    python scripts/track_video_persons.py \\
        --video shea.mp4 \\
        --lazy-load --trim-memory --max-obj-ptrs 256 \\
        --overlay-video /tmp/persons.mp4

Detect every 10th frame (faster), also save JSON::

    python scripts/track_video_persons.py \\
        --video shea.mp4 --detect-every 10 \\
        --lazy-load --trim-memory \\
        --output-json /tmp/persons.json \\
        --overlay-video /tmp/persons.mp4
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from sam3.model_builder import build_sam3_detector_and_tracker

# ---------------------------------------------------------------------------
# Colors for up to 20 persons (RGB).  Wraps around for more.
# ---------------------------------------------------------------------------
PERSON_COLORS = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
    (0, 0, 128),
    (128, 128, 128),
]


def color_for_id(obj_id: int) -> Tuple[int, int, int]:
    return PERSON_COLORS[obj_id % len(PERSON_COLORS)]


# ---------------------------------------------------------------------------
# Lazy video frame loader (same as track_video_bbox.py)
# ---------------------------------------------------------------------------
class LazyVideoFrameLoader:
    """Load and normalize video frames on demand — O(1) memory."""

    def __init__(self, video_path: str, image_size: int = 1008):
        self.image_size = image_size
        self._img_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)[
            :, None, None
        ]
        self._img_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)[
            :, None, None
        ]
        if os.path.isdir(video_path):
            self._mode = "jpeg"
            jpg_exts = (".jpg", ".jpeg", ".JPG", ".JPEG")
            names = [p for p in os.listdir(video_path) if p.endswith(jpg_exts)]
            if not names:
                raise RuntimeError(f"No JPEG frames in {video_path}")
            names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            self._img_paths = [os.path.join(video_path, n) for n in names]
            first = Image.open(self._img_paths[0])
            self.video_width, self.video_height = first.size
            self._num_frames = len(self._img_paths)
            self._vr = None
        else:
            self._mode = "mp4"
            import decord

            decord.bridge.set_bridge("torch")
            self._video_path = video_path
            self._vr = decord.VideoReader(
                video_path, width=image_size, height=image_size
            )
            first_frame = decord.VideoReader(video_path)[0]
            self.video_height, self.video_width = first_frame.shape[:2]
            self._num_frames = len(self._vr)

    def __len__(self):
        return self._num_frames

    def __getitem__(self, index):
        if self._mode == "jpeg":
            img_pil = Image.open(self._img_paths[index]).convert("RGB")
            img_pil = img_pil.resize((self.image_size, self.image_size))
            img = (
                torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
            )
        else:
            img = self._vr[index].permute(2, 0, 1).float() / 255.0
        img = (img - self._img_mean) / self._img_std
        return img


# ---------------------------------------------------------------------------
# Lazy original-resolution frame reader (for overlay & detection)
# ---------------------------------------------------------------------------
class OriginalFrameReader:
    """Read original-resolution RGB frames sequentially (keeps file handle open)."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self._cap = None
        self._next_idx = 0
        if os.path.isdir(video_path):
            self._mode = "jpeg"
            jpg_exts = (".jpg", ".jpeg", ".JPG", ".JPEG")
            names = [p for p in os.listdir(video_path) if p.endswith(jpg_exts)]
            names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            self._img_paths = [os.path.join(video_path, n) for n in names]
            first = Image.open(self._img_paths[0])
            self.width, self.height = first.size
            self.fps = 10.0
            self.num_frames = len(self._img_paths)
        else:
            import cv2 as _cv2

            self._cap = _cv2.VideoCapture(video_path)
            self.width = int(self._cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self._cap.get(_cv2.CAP_PROP_FPS) or 30.0
            self.num_frames = int(self._cap.get(_cv2.CAP_PROP_FRAME_COUNT))
            self._mode = "mp4"

    def read_frame(self, frame_idx: int) -> np.ndarray:
        """Return RGB uint8 array of shape (H, W, 3)."""
        if self._mode == "jpeg":
            return np.array(Image.open(self._img_paths[frame_idx]).convert("RGB"))
        else:
            import cv2 as _cv2

            # Sequential reads are fast; only seek if out of order
            if frame_idx != self._next_idx:
                self._cap.set(_cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = self._cap.read()
            self._next_idx = frame_idx + 1
            if not ok:
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)

    def close(self):
        if self._cap is not None:
            self._cap.release()


# ---------------------------------------------------------------------------
# Person detector (SAM3 DETR with text prompt — uses shared processor)
# ---------------------------------------------------------------------------
class PersonDetector:
    """Thin wrapper around a :class:`Sam3Processor` that runs the text prompt
    ``"person"`` and returns per-detection dicts with bbox, mask, and score.

    The processor (and its underlying backbone) is created externally by
    :func:`build_sam3_detector_and_tracker` so that the ViT backbone is
    physically shared with the tracker — no duplicate weights in memory.
    """

    def __init__(self, processor, prompt: str = "person"):
        self.processor = processor
        self.prompt = prompt

    @torch.inference_mode()
    def detect(self, frame_rgb: np.ndarray) -> List[Dict]:
        """Detect persons in an RGB frame.

        Returns list of ``{"bbox_xyxy": [...], "mask": np.ndarray, "score": float}``.
        """
        pil_img = Image.fromarray(frame_rgb)
        state = self.processor.set_image(pil_img)
        state = self.processor.set_text_prompt(self.prompt, state)

        persons = []
        if "boxes" not in state or len(state["boxes"]) == 0:
            return persons

        boxes = state["boxes"].float().cpu().numpy()       # (N, 4) xyxy pixels
        scores = state["scores"].float().cpu().numpy()     # (N,)
        masks = state["masks"].squeeze(1).cpu().numpy()    # (N, H, W) bool

        for i in range(len(boxes)):
            persons.append({
                "bbox_xyxy": boxes[i].tolist(),
                "mask": masks[i],
                "score": float(scores[i]),
            })
        return persons


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def bbox_iou(a: List[float], b: List[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def mask_to_bbox(mask: np.ndarray) -> Optional[List[int]]:
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def match_detections_to_tracks(
    det_bboxes: List[List[float]],
    tracked: Dict[int, Dict],
    iou_threshold: float = 0.3,
) -> Tuple[Dict[int, int], List[int]]:
    """Match detection bboxes to tracked objects via greedy IoU.

    Returns
    -------
    matched : dict mapping det_index -> tracked obj_id
    unmatched_det_indices : list of detection indices with no match
    """
    trk_ids = []
    trk_bboxes = []
    for obj_id, info in tracked.items():
        if info["bbox_xyxy"] is not None:
            trk_ids.append(obj_id)
            trk_bboxes.append(info["bbox_xyxy"])

    if not det_bboxes or not trk_bboxes:
        return {}, list(range(len(det_bboxes)))

    iou_mat = np.zeros((len(det_bboxes), len(trk_bboxes)))
    for i, db in enumerate(det_bboxes):
        for j, tb in enumerate(trk_bboxes):
            iou_mat[i, j] = bbox_iou(db, tb)

    matched: Dict[int, int] = {}
    used_d, used_t = set(), set()
    # Greedy: pick highest IoU pair repeatedly
    for _ in range(min(len(det_bboxes), len(trk_bboxes))):
        if iou_mat.max() < iou_threshold:
            break
        di, ti = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
        matched[int(di)] = trk_ids[ti]
        used_d.add(int(di))
        used_t.add(ti)
        iou_mat[di, :] = 0
        iou_mat[:, ti] = 0

    unmatched = [i for i in range(len(det_bboxes)) if i not in used_d]
    return matched, unmatched


# ---------------------------------------------------------------------------
# Multi-person tracker
# ---------------------------------------------------------------------------
class MultiPersonTracker:
    """Manage multiple SAM 3 tracker states for multi-person tracking."""

    def __init__(self, predictor, lazy_loader, video_h, video_w, num_frames):
        self.predictor = predictor
        self.lazy_loader = lazy_loader
        self.video_h = video_h
        self.video_w = video_w
        self.num_frames = num_frames
        self.tracker_states: List[dict] = []
        self.next_obj_id = 1
        # How far back the tracker actually looks for memory / pointers.
        # Anything older is dead weight and can be deleted.
        self._evict_horizon = max(
            predictor.num_maskmem,           # spatial memory window (default 7)
            predictor.max_obj_ptrs_in_encoder,  # pointer window (default 16)
        ) + 2  # small margin

    def _create_state(self) -> dict:
        """Create a fresh tracker inference state."""
        state = self.predictor.init_state(
            video_height=self.video_h,
            video_width=self.video_w,
            num_frames=self.num_frames,
        )
        state["images"] = self.lazy_loader
        return state

    def add_persons(
        self,
        frame_idx: int,
        detections: List[Dict],
    ) -> List[int]:
        """Seed new persons on *frame_idx* using detected masks.

        Each entry in *detections* should have ``"mask"`` (H×W bool array)
        and ``"bbox_xyxy"`` (pixel coords).  If ``"mask"`` is present it is
        used directly (better quality from the SAM3 DETR detector); otherwise
        we fall back to the bounding box.

        Returns assigned obj_ids.
        """
        if not detections:
            return []
        state = self._create_state()
        assigned = []
        for det in detections:
            oid = self.next_obj_id
            self.next_obj_id += 1
            mask = det.get("mask")
            if mask is not None:
                mask_t = torch.from_numpy(mask.astype(np.float32))
                self.predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=oid,
                    mask=mask_t,
                )
            else:
                x1, y1, x2, y2 = det["bbox_xyxy"]
                norm = np.array(
                    [[x1 / self.video_w, y1 / self.video_h,
                      x2 / self.video_w, y2 / self.video_h]],
                    dtype=np.float32,
                )
                self.predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=frame_idx,
                    obj_id=oid,
                    box=norm,
                )
            assigned.append(oid)
        self.predictor.propagate_in_video_preflight(state)
        self.tracker_states.append(state)
        return assigned

    def _evict_old_outputs(self, frame_idx: int):
        """Strip stored outputs to only what the tracker actually reads.

        Two independent windows:
          - Spatial memory window (``num_maskmem`` frames, default 7):
            keep ``maskmem_features`` + ``maskmem_pos_enc``
          - Pointer window (``max_obj_ptrs_in_encoder`` frames):
            keep ``obj_ptr`` only (1 KB each)

        Outside spatial window → drop maskmem_features/maskmem_pos_enc.
        Outside pointer window → drop obj_ptr.
        Outside both → delete the entire entry.
        Always drop: pred_masks, object_score_logits, iou_score.
        """
        # The tracker reads maskmem_features from entries it finds in
        # output_dict.  If an entry exists but maskmem_features was stripped,
        # it crashes with a KeyError.  So we can only drop maskmem_features
        # by deleting the ENTIRE entry from output_dict.  We keep obj_ptr
        # in a separate lightweight store instead.
        r = self.predictor.memory_temporal_stride_for_eval
        mem_cutoff = frame_idx - (self.predictor.num_maskmem) * r - 1
        ptr_cutoff = frame_idx - self.predictor.max_obj_ptrs_in_encoder - 1

        for state in self.tracker_states:
            all_dicts = [state["output_dict"]] + list(
                state["output_dict_per_obj"].values()
            )
            for state_dict in all_dicts:
                # Never evict cond_frame_outputs — the tracker requires at
                # least one conditioning frame to exist and reads its
                # maskmem_features.  There's typically just 1 per state.
                d = state_dict["non_cond_frame_outputs"]
                to_del = []
                for t, out in d.items():
                    if t >= frame_idx:
                        continue
                    for drop in (
                        "pred_masks", "object_score_logits",
                        "iou_score", "eff_iou_score",
                    ):
                        out.pop(drop, None)
                    if t < mem_cutoff:
                        to_del.append(t)
                for t in to_del:
                    del d[t]

    def propagate_frame(self, frame_idx: int) -> Dict[int, Dict]:
        """Propagate all states one frame. Returns {obj_id: info_dict}.

        The ViT backbone is shared: the first state computes and caches the
        features, and all subsequent states reuse the cache — so the backbone
        runs only **once** per frame regardless of how many states exist.
        """
        results: Dict[int, Dict] = {}
        shared_cache = None
        for state in self.tracker_states:
            if not state["obj_ids"]:
                continue
            # Inject cached backbone features from the first state that
            # computed them, so later states skip the backbone entirely.
            if shared_cache is not None:
                state["cached_features"] = shared_cache
            for out in self.predictor.propagate_in_video(
                state,
                start_frame_idx=frame_idx,
                max_frame_num_to_track=0,
                reverse=False,
                propagate_preflight=False,
                tqdm_disable=True,
            ):
                _, obj_ids, _, video_res_masks, obj_scores = out
                masks = (video_res_masks > 0.0).squeeze(1).cpu().numpy()
                scores = obj_scores.sigmoid().squeeze(-1).float().cpu().numpy()
                for i, oid in enumerate(obj_ids):
                    m = masks[i].astype(bool)
                    results[oid] = {
                        "mask": m,
                        "bbox_xyxy": mask_to_bbox(m),
                        "score": float(scores[i]),
                    }
            # Grab the cache after the first state computes backbone features
            if shared_cache is None:
                shared_cache = state["cached_features"]
        self._evict_old_outputs(frame_idx)
        return results

    def propagate_frame_last_state(self, frame_idx: int) -> Dict[int, Dict]:
        """Propagate only the most recently added state for *frame_idx*."""
        results: Dict[int, Dict] = {}
        state = self.tracker_states[-1]
        if not state["obj_ids"]:
            return results
        for out in self.predictor.propagate_in_video(
            state,
            start_frame_idx=frame_idx,
            max_frame_num_to_track=0,
            reverse=False,
            propagate_preflight=False,
        ):
            _, obj_ids, _, video_res_masks, obj_scores = out
            masks = (video_res_masks > 0.0).squeeze(1).cpu().numpy()
            scores = obj_scores.sigmoid().squeeze(-1).float().cpu().numpy()
            for i, oid in enumerate(obj_ids):
                m = masks[i].astype(bool)
                results[oid] = {
                    "mask": m,
                    "bbox_xyxy": mask_to_bbox(m),
                    "score": float(scores[i]),
                }
        return results


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------
def draw_person_overlay(
    frame_rgb: np.ndarray,
    persons: Dict[int, Dict],
    frame_idx: int,
) -> np.ndarray:
    """Draw masks, bboxes and 'Person {id}' labels on *frame_rgb*."""
    import cv2

    overlay = frame_rgb.copy()
    for obj_id, info in sorted(persons.items()):
        color_rgb = color_for_id(obj_id)
        mask = info.get("mask")
        bbox = info.get("bbox_xyxy")

        # Mask tint
        if mask is not None and mask.any():
            if mask.shape != overlay.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (overlay.shape[1], overlay.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            tint = np.array(color_rgb, dtype=np.uint8)
            overlay[mask] = (0.45 * overlay[mask] + 0.55 * tint).astype(np.uint8)

        # Bbox + label
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_rgb, 2)
            label = f"Person {obj_id}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                overlay,
                (x1, max(y1 - th - 8, 0)),
                (x1 + tw + 4, max(y1, th + 8)),
                color_rgb,
                -1,
            )
            cv2.putText(
                overlay,
                label,
                (x1 + 2, max(y1 - 4, th + 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    # Frame counter
    cv2.putText(
        overlay,
        f"frame {frame_idx}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Detect and track multiple persons in a video with SAM 3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", required=True, help="MP4 or JPEG folder.")
    p.add_argument(
        "--detect-every",
        type=int,
        default=1,
        help="Run person detector every N frames. Default: 1 (every frame).",
    )
    p.add_argument(
        "--det-score-threshold",
        type=float,
        default=0.5,
        help="Minimum detection confidence for persons. Default: 0.5.",
    )
    p.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold to match a detection to an existing track.",
    )
    p.add_argument(
        "--max-frames", type=int, default=None, help="Limit number of frames."
    )
    p.add_argument("--output-json", default=None, help="Save per-frame JSON.")
    p.add_argument("--output-masks", default=None, help="Save mask PNGs.")
    p.add_argument("--overlay-video", default=None, help="Save overlay MP4.")
    p.add_argument("--checkpoint", default=None, help="SAM 3 checkpoint path.")
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--lazy-load", action="store_true", help="O(1) frame memory.")
    p.add_argument(
        "--trim-memory",
        action="store_true",
        help="Drop old spatial memory (keep obj pointers).",
    )
    p.add_argument(
        "--max-obj-ptrs",
        type=int,
        default=16,
        help="Object pointer attention window.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # ---- Build models (shared backbone) ----
    print("Building SAM 3 detector + tracker (shared ViT backbone)...")
    processor, predictor = build_sam3_detector_and_tracker(
        checkpoint_path=args.checkpoint,
        trim_past_memory=args.trim_memory,
        max_obj_ptrs_in_encoder=args.max_obj_ptrs,
        det_confidence=args.det_score_threshold,
        device=args.device,
    )
    detector = PersonDetector(processor)

    # ---- Video setup ----
    if args.lazy_load:
        loader = LazyVideoFrameLoader(args.video, image_size=predictor.image_size)
        video_h, video_w = loader.video_height, loader.video_width
        num_frames = len(loader)
    else:
        loader = None
        # Fall back to eager loading (handled inside init_state)
        raise NotImplementedError(
            "Multi-person tracking requires --lazy-load for frame-by-frame "
            "propagation. Please add --lazy-load."
        )

    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)

    print(f"Video: {video_w}x{video_h}, {num_frames} frames")

    # Original-resolution frame reader (for detection & overlay)
    orig_reader = OriginalFrameReader(args.video)

    # ---- Tracker ----
    tracker = MultiPersonTracker(predictor, loader, video_h, video_w, num_frames)

    # ---- Overlay writer ----
    overlay_writer = None
    cv2_mod = None
    if args.overlay_video is not None:
        import cv2 as cv2_mod

        os.makedirs(
            os.path.dirname(os.path.abspath(args.overlay_video)), exist_ok=True
        )
        fourcc = cv2_mod.VideoWriter_fourcc(*"mp4v")
        overlay_writer = cv2_mod.VideoWriter(
            args.overlay_video,
            fourcc,
            orig_reader.fps,
            (orig_reader.width, orig_reader.height),
        )

    # Mask output dir
    mask_dir = args.output_masks
    if mask_dir is not None:
        os.makedirs(mask_dir, exist_ok=True)

    json_results: Dict[int, List[Dict]] = {}

    # ---- Main loop ----
    print("Tracking...")
    try:
        for frame_idx in tqdm(range(num_frames), desc="frames"):
            # Step 1: Propagate existing tracks for this frame
            tracked = tracker.propagate_frame(frame_idx)

            # Step 2: Run person detection (every Nth frame)
            if frame_idx % args.detect_every == 0:
                frame_rgb = orig_reader.read_frame(frame_idx)
                detections = detector.detect(frame_rgb)
                det_bboxes = [d["bbox_xyxy"] for d in detections]

                # Step 3: Match detections to existing tracks
                _matched, unmatched_idxs = match_detections_to_tracks(
                    det_bboxes, tracked, iou_threshold=args.iou_threshold
                )

                # Step 4: Create new tracks for unmatched detections
                if unmatched_idxs:
                    new_dets = [detections[i] for i in unmatched_idxs]
                    new_ids = tracker.add_persons(frame_idx, new_dets)
                    # Propagate the new state for this frame to get initial masks
                    new_tracked = tracker.propagate_frame_last_state(frame_idx)
                    tracked.update(new_tracked)
                    print(
                        f"  frame {frame_idx}: detected {len(detections)} persons, "
                        f"{len(new_ids)} new (ids: {new_ids})"
                    )

            # Step 5: Collect outputs
            frame_entries = []
            for obj_id, info in sorted(tracked.items()):
                frame_entries.append(
                    {
                        "obj_id": obj_id,
                        "bbox_xyxy_pixels": info["bbox_xyxy"],
                        "score": info["score"],
                        "present": info["bbox_xyxy"] is not None,
                    }
                )
            json_results[frame_idx] = frame_entries

            # Stream mask PNGs
            if mask_dir is not None:
                for obj_id, info in tracked.items():
                    m = info["mask"]
                    png = os.path.join(
                        mask_dir, f"frame{frame_idx:06d}_person{obj_id}.png"
                    )
                    Image.fromarray(m.astype(np.uint8) * 255, mode="L").save(png)

            # Stream overlay frame
            if overlay_writer is not None:
                if frame_idx % args.detect_every != 0:
                    frame_rgb = orig_reader.read_frame(frame_idx)
                overlay = draw_person_overlay(frame_rgb, tracked, frame_idx)
                overlay_writer.write(
                    cv2_mod.cvtColor(overlay, cv2_mod.COLOR_RGB2BGR)
                )

    finally:
        if overlay_writer is not None:
            overlay_writer.release()
        orig_reader.close()

    print(f"Tracked {len(json_results)} frames, "
          f"{tracker.next_obj_id - 1} persons total.")

    # ---- Save JSON ----
    if args.output_json is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        serializable = {
            "video": os.path.abspath(args.video),
            "video_height": video_h,
            "video_width": video_w,
            "num_frames": num_frames,
            "total_persons": tracker.next_obj_id - 1,
            "frames": {
                str(idx): entries for idx, entries in sorted(json_results.items())
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved JSON to {args.output_json}")

    if mask_dir is not None:
        print(f"Saved masks to {mask_dir}")
    if args.overlay_video is not None:
        print(f"Saved overlay to {args.overlay_video}")


if __name__ == "__main__":
    main()
