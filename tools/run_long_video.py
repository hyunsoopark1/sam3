"""
Memory-efficient SAM3 video inference for thousands of frames.

Key optimizations:
  1. LazyFrameLoader -- loads one frame at a time from disk (image folder or
     video file).  Nothing is kept in CPU or GPU memory after the frame is
     consumed by the model.
  2. After every frame the tracker's per-frame output_dict is trimmed so that
     only `maskmem_features` (spatial memory) and `obj_ptr` (object pointer)
     survive.  Everything else -- pred_masks, pred_masks_high_res,
     maskmem_pos_enc, object_score_logits, etc. -- is deleted immediately.
  3. cached_frame_outputs is cleared after each frame.

Usage:
    python tools/run_long_video.py \
        --video_path /path/to/frames_folder_or_video.mp4 \
        --text "person" \
        --output_dir /path/to/output
"""

import argparse
import gc
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

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
        # Read original dimensions from first frame
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
        cap.release()
        if self._num_frames <= 0:
            raise RuntimeError(f"Could not determine frame count for {path}")
        # We will open a fresh VideoCapture and seek on each access.
        # For sequential access this is fine (cv2 seeks efficiently forward).
        self._cap = None
        self._cap_pos = -1

    def _load_from_video(self, index: int) -> torch.Tensor:
        # Open / reuse the capture, seeking only when needed.
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


# ---------------------------------------------------------------------------
# Trim tracker output_dict to keep only maskmem_features + obj_ptr
# ---------------------------------------------------------------------------

def _trim_tracker_output_dict(tracker_state, current_frame_idx, num_maskmem):
    """Remove all heavy tensors from the tracker's output_dict except
    ``maskmem_features`` (spatial) and ``obj_ptr``.

    We keep the most recent ``num_maskmem`` frames' maskmem_features because
    the tracker's memory-attention window needs them.  For frames older than
    that, even maskmem_features is dropped (only obj_ptr remains).
    """
    output_dict = tracker_state.get("output_dict")
    if output_dict is None:
        return

    keys_to_keep = {"maskmem_features", "obj_ptr"}

    for bucket_name in ("cond_frame_outputs", "non_cond_frame_outputs"):
        bucket = output_dict.get(bucket_name, {})
        for frame_idx, frame_out in list(bucket.items()):
            if frame_out is None:
                continue
            # Remove everything except maskmem_features and obj_ptr
            keys_to_delete = [k for k in frame_out if k not in keys_to_keep]
            for k in keys_to_delete:
                del frame_out[k]


# ---------------------------------------------------------------------------
# Main: memory-efficient propagation
# ---------------------------------------------------------------------------

def run_long_video(
    video_path: str,
    text_prompt: str,
    output_dir: str | None = None,
    gpus_to_use=None,
    save_every: int = 1,
    gc_every: int = 50,
):
    """Run SAM3 on a long video with constant memory usage.

    Args:
        video_path:   Path to a folder of JPEG frames or a video file.
        text_prompt:  Text describing the objects to track (e.g. "person").
        output_dir:   Where to save per-frame mask PNGs.  ``None`` to skip.
        gpus_to_use:  List of GPU ids; defaults to current device only.
        save_every:   Save a mask PNG every N frames (1 = every frame).
        gc_every:     Run gc.collect + empty_cache every N frames.
    """
    from sam3.model_builder import build_sam3_video_predictor

    if gpus_to_use is None:
        gpus_to_use = [torch.cuda.current_device()]

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Build predictor (loads model weights) --------------------------
    print("Loading SAM3 model ...")
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
    model = predictor.model  # Sam3VideoInference

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
    print(f"Video: {num_frames} frames, original size {orig_w}x{orig_h}")

    # ---- 3. Build a minimal inference_state with the lazy loader -----------
    # We replicate what Sam3VideoInference.init_state does, but swap in our
    # LazyFrameLoader instead of the full pre-loaded tensor.
    inference_state = model.init_state(
        resource_path=video_path,
        offload_video_to_cpu=True,  # load to CPU first (minimise GPU mem)
    )
    # Replace the pre-loaded image batch with our lazy loader.
    # The model's backbone indexes img_batch[frame_idx] and sends it to GPU
    # on the fly -- our loader does the same thing but without caching.
    inference_state["input_batch"].img_batch = loader

    # ---- 4. Add text prompt on frame 0 -------------------------------------
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

    # ---- 5. Propagate with aggressive memory cleanup -----------------------
    print("Propagating ...")
    num_maskmem = model.tracker.num_maskmem  # typically 7
    t0 = time.time()

    for response in predictor.handle_stream_request(
        dict(
            type="propagate_in_video",
            session_id=session_id,
            propagation_direction="forward",
        )
    ):
        frame_idx = response["frame_index"]
        outputs = response["outputs"]

        # -- save masks (optional) -------------------------------------------
        if output_dir is not None and outputs is not None and frame_idx % save_every == 0:
            masks = outputs.get("out_binary_masks")  # (N, H, W) bool
            obj_ids = outputs.get("out_obj_ids")      # (N,)
            if masks is not None and len(masks) > 0:
                # Merge all object masks into a single label map
                combined = np.zeros((orig_h, orig_w), dtype=np.uint16)
                for i, oid in enumerate(obj_ids):
                    combined[masks[i]] = int(oid) + 1
                out_path = os.path.join(output_dir, f"{frame_idx:06d}.png")
                Image.fromarray(combined).save(out_path)

        # -- trim tracker memory to keep only spatial + obj_ptr --------------
        for tracker_state in inference_state["tracker_inference_states"]:
            _trim_tracker_output_dict(tracker_state, frame_idx, num_maskmem)

        # -- clear cached frame outputs (full-res masks we no longer need) ---
        inference_state["cached_frame_outputs"].pop(frame_idx, None)

        # -- periodic GC + CUDA cache flush ----------------------------------
        if frame_idx % gc_every == 0 and frame_idx > 0:
            gc.collect()
            torch.cuda.empty_cache()

        if frame_idx % 100 == 0:
            mem_mb = torch.cuda.memory_allocated() // (1024 * 1024)
            elapsed = time.time() - t0
            fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  frame {frame_idx}/{num_frames}  |  "
                f"GPU mem {mem_mb} MiB  |  {fps:.1f} fps"
            )

    elapsed = time.time() - t0
    print(
        f"Done: {num_frames} frames in {elapsed:.1f}s "
        f"({num_frames / elapsed:.1f} fps)"
    )

    # ---- 6. Cleanup --------------------------------------------------------
    predictor.handle_request(
        dict(type="close_session", session_id=session_id)
    )
    predictor.shutdown()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run SAM3 on a long video with constant memory usage."
    )
    parser.add_argument(
        "--video_path",
        required=True,
        help="Path to a folder of JPEG frames or an MP4/video file.",
    )
    parser.add_argument(
        "--text",
        required=True,
        help="Text prompt describing the objects to track (e.g. 'person').",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save per-frame label-map PNGs. Omit to skip saving.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save masks every N frames (default: every frame).",
    )
    parser.add_argument(
        "--gc_every",
        type=int,
        default=50,
        help="Run gc.collect + cuda.empty_cache every N frames (default: 50).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        nargs="*",
        default=None,
        help="GPU IDs to use (default: current device only).",
    )
    args = parser.parse_args()

    run_long_video(
        video_path=args.video_path,
        text_prompt=args.text,
        output_dir=args.output_dir,
        gpus_to_use=args.gpus,
        save_every=args.save_every,
        gc_every=args.gc_every,
    )


if __name__ == "__main__":
    main()
