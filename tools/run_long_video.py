"""
Memory-efficient SAM3 video inference for thousands of frames.

Outputs a visualization video with mask overlays rendered on-the-fly --
nothing accumulates in memory across frames.

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
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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

    @property
    def fps(self):
        if self._backend == "video":
            return self._fps
        return 30.0  # default for image folders


# ---------------------------------------------------------------------------
# Raw frame reader -- reads the original (un-normalised) RGB frame for
# visualization.  Shares the same sequential-seek cv2.VideoCapture for video
# files so that sequential reads are efficient.
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
# Render a single overlay frame -- masks + bounding boxes + labels
# ---------------------------------------------------------------------------

# Deterministic, perceptually-distinct colours (generated once)
_COLOR_CACHE = None


def _get_colors():
    global _COLOR_CACHE
    if _COLOR_CACHE is not None:
        return _COLOR_CACHE
    try:
        from sam3.visualization_utils import COLORS
        _COLOR_CACHE = COLORS
    except ImportError:
        # Fallback: simple tab-style palette
        rng = np.random.RandomState(42)
        _COLOR_CACHE = rng.rand(128, 3).astype(np.float64)
    return _COLOR_CACHE


def render_overlay(img_rgb: np.ndarray, outputs: dict, frame_idx: int,
                   alpha: float = 0.45) -> np.ndarray:
    """Overlay masks, boxes and labels on a raw RGB frame.

    Args:
        img_rgb:   (H, W, 3) uint8 RGB image.
        outputs:   Dict from propagate_in_video (keys: out_binary_masks,
                   out_obj_ids, out_boxes_xywh, out_probs).
        frame_idx: Frame number (drawn in top-left corner).
        alpha:     Mask opacity.

    Returns:
        (H, W, 3) uint8 RGB overlay image.
    """
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

            # -- mask overlay --
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

            # -- bounding box --
            if boxes is not None:
                bx, by, bw, bh = boxes[i]
                x1, y1 = int(bx * w), int(by * h)
                x2, y2 = int((bx + bw) * w), int((by + bh) * h)
                c_bgr = tuple(int(x) for x in c_uint8)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), c_bgr, 2)

                # -- label text --
                prob = probs[i] if probs is not None else None
                label = f"id={int(oid)}"
                if prob is not None:
                    label += f" {prob:.2f}"
                cv2.putText(
                    overlay, label, (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_bgr, 2, cv2.LINE_AA,
                )

    # -- frame counter --
    cv2.putText(
        overlay, f"Frame {frame_idx}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return overlay


# ---------------------------------------------------------------------------
# Trim tracker output_dict to keep only maskmem_features + obj_ptr
# ---------------------------------------------------------------------------

def _trim_tracker_output_dict(tracker_state):
    """Remove all heavy tensors from the tracker's output_dict except
    ``maskmem_features`` (spatial) and ``obj_ptr``.
    """
    output_dict = tracker_state.get("output_dict")
    if output_dict is None:
        return

    keys_to_keep = {"maskmem_features", "obj_ptr"}

    for bucket_name in ("cond_frame_outputs", "non_cond_frame_outputs"):
        bucket = output_dict.get(bucket_name, {})
        for frame_out in bucket.values():
            if frame_out is None:
                continue
            keys_to_delete = [k for k in frame_out if k not in keys_to_keep]
            for k in keys_to_delete:
                del frame_out[k]


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
    """
    from sam3.model_builder import build_sam3_video_predictor

    if gpus_to_use is None:
        gpus_to_use = [torch.cuda.current_device()]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

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
    if fps is None:
        fps = loader.fps
    print(f"Video: {num_frames} frames, {orig_w}x{orig_h} @ {fps:.1f} fps")

    # ---- 3. Raw frame reader for visualization -----------------------------
    raw_reader = RawFrameReader(video_path)

    # ---- 4. Build inference_state with the lazy loader ---------------------
    inference_state = model.init_state(
        resource_path=video_path,
        offload_video_to_cpu=True,
    )
    # Replace the pre-loaded image batch with our lazy loader.
    inference_state["input_batch"].img_batch = loader

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
    # Write to a temp file first, then re-encode with ffmpeg for compatibility.
    tmp_path = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (orig_w, orig_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {tmp_path}")

    # ---- 7. Propagate, render each frame, write to video -------------------
    print("Propagating and rendering ...")
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

        # -- read raw frame & render overlay ---------------------------------
        raw_frame = raw_reader.read(frame_idx)  # (H, W, 3) uint8 RGB
        if outputs is not None:
            overlay = render_overlay(raw_frame, outputs, frame_idx)
        else:
            overlay = raw_frame.copy()
            cv2.putText(
                overlay, f"Frame {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
            )
        # cv2.VideoWriter expects BGR
        writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # -- trim tracker memory to keep only spatial + obj_ptr --------------
        for tracker_state in inference_state["tracker_inference_states"]:
            _trim_tracker_output_dict(tracker_state)

        # -- clear cached frame outputs (full-res masks we no longer need) ---
        inference_state["cached_frame_outputs"].pop(frame_idx, None)

        # -- periodic GC + CUDA cache flush ----------------------------------
        if frame_idx % gc_every == 0 and frame_idx > 0:
            gc.collect()
            torch.cuda.empty_cache()

        if frame_idx % 100 == 0:
            mem_mb = torch.cuda.memory_allocated() // (1024 * 1024)
            elapsed = time.time() - t0
            speed = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  frame {frame_idx}/{num_frames}  |  "
                f"GPU mem {mem_mb} MiB  |  {speed:.1f} fps"
            )

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
        # ffmpeg not available -- keep the raw mp4v file
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
    args = parser.parse_args()

    run_long_video(
        video_path=args.video_path,
        text_prompt=args.text,
        output_path=args.output,
        gpus_to_use=args.gpus,
        fps=args.fps,
        gc_every=args.gc_every,
    )


if __name__ == "__main__":
    main()
