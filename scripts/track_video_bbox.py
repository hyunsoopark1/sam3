"""
Track an object across a video using only the SAM 3 tracker (no detector).

Given a video and a bounding box on the first frame, this script propagates the
mask of that object across the whole video. It uses
``build_sam3_tracker_only`` from :mod:`sam3.model_builder`, which loads only the
tracker plus the shared vision backbone from the SAM 3 checkpoint — the
detector, text encoder, detection transformer and segmentation head are never
instantiated.

Examples
--------
MP4 input, pixel-space XYXY box::

    python scripts/track_video_bbox.py \\
        --video sam3/assets/videos/bedroom.mp4 \\
        --bbox 300 0 500 400 \\
        --output-json /tmp/tracks.json \\
        --overlay-video /tmp/tracked.mp4

JPEG folder input, XYWH box in normalized coordinates::

    python scripts/track_video_bbox.py \\
        --video /path/to/jpeg_folder \\
        --bbox 0.25 0.10 0.20 0.40 \\
        --bbox-format xywh --bbox-normalized \\
        --output-json /tmp/tracks.json
"""

import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_tracker_only


def _import_cv2():
    """Lazy-import cv2 so the core tracking path works without it."""
    try:
        import cv2

        return cv2
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "opencv-python (cv2) is required for --overlay-video and "
            "--output-masks.  Install it with:  pip install opencv-python"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track an object through a video using the SAM 3 tracker, "
            "seeded with a single bounding box on the first frame."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to an MP4 file or a folder of JPEG frames named <idx>.jpg.",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        required=True,
        metavar=("A", "B", "C", "D"),
        help=(
            "First-frame bounding box. Default is absolute-pixel XYXY "
            "(x_min y_min x_max y_max). Use --bbox-format and "
            "--bbox-normalized to change the interpretation."
        ),
    )
    parser.add_argument(
        "--bbox-format",
        default="xyxy",
        choices=["xyxy", "xywh"],
        help="Box format: 'xyxy' (x_min y_min x_max y_max) or 'xywh'.",
    )
    parser.add_argument(
        "--bbox-normalized",
        action="store_true",
        help=(
            "If set, --bbox values are in [0, 1] relative to the video width/"
            "height instead of absolute pixels."
        ),
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Frame index where the bounding box is defined. Default: 0.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to track. Default: all frames.",
    )
    parser.add_argument(
        "--obj-id",
        type=int,
        default=1,
        help="Object id to associate with the bounding box. Default: 1.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save per-frame tracking results as JSON.",
    )
    parser.add_argument(
        "--output-masks",
        default=None,
        help=(
            "Optional directory to save per-frame binary masks as PNG files "
            "(one file per tracked frame)."
        ),
    )
    parser.add_argument(
        "--overlay-video",
        default=None,
        help=(
            "Optional path to save an MP4 video with the predicted mask and "
            "bounding box drawn on top of each frame."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional path to a SAM 3 checkpoint file. If omitted, the "
            "sam3.pt checkpoint is downloaded from Hugging Face."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on.",
    )
    parser.add_argument(
        "--apply-temporal-disambiguation",
        action="store_true",
        help="Enable SAM2Long-style memory selection inside the tracker.",
    )
    parser.add_argument(
        "--offload-to-cpu",
        action="store_true",
        help=(
            "Offload per-frame tracking state (memory features, mask logits) "
            "to CPU RAM. Essential for long videos (thousands of frames) to "
            "avoid GPU OOM. Slightly slower due to CPU<->GPU transfers."
        ),
    )
    parser.add_argument(
        "--offload-video-to-cpu",
        action="store_true",
        help=(
            "Keep decoded video frames in CPU RAM instead of GPU. Saves GPU "
            "memory at the cost of a small per-frame transfer overhead."
        ),
    )
    parser.add_argument(
        "--trim-memory",
        action="store_true",
        help=(
            "Discard memory features from frames that have fallen outside the "
            "tracker's attention window. Keeps GPU/CPU memory bounded for "
            "forward-only tracking on very long videos."
        ),
    )
    return parser.parse_args()


def normalize_bbox_xyxy(
    raw_bbox: List[float],
    bbox_format: str,
    is_normalized: bool,
    video_w: int,
    video_h: int,
) -> np.ndarray:
    """Return a single-object normalized XYXY box of shape ``(1, 4)``."""
    a, b, c, d = raw_bbox
    if bbox_format == "xywh":
        x1, y1, x2, y2 = a, b, a + c, b + d
    else:
        x1, y1, x2, y2 = a, b, c, d

    if not is_normalized:
        x1 /= video_w
        x2 /= video_w
        y1 /= video_h
        y2 /= video_h

    # Clamp to [0, 1] so we don't ask the model to look outside the image.
    x1 = float(np.clip(x1, 0.0, 1.0))
    x2 = float(np.clip(x2, 0.0, 1.0))
    y1 = float(np.clip(y1, 0.0, 1.0))
    y2 = float(np.clip(y2, 0.0, 1.0))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid bounding box after parsing (xyxy norm): "
            f"({x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f}). "
            "Check --bbox, --bbox-format, and --bbox-normalized."
        )

    return np.array([[x1, y1, x2, y2]], dtype=np.float32)


def mask_to_bbox_xyxy(mask: np.ndarray) -> Optional[List[int]]:
    """Compute an XYXY bounding box (pixel space) from a boolean mask."""
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def iter_frames_from_video(video_path: str) -> Tuple[int, int, float, List[np.ndarray]]:
    """Load all RGB frames from an MP4 file.

    Used only to redraw overlays on top of the original frames; the tracker
    itself loads its own copy through ``init_state``.
    """
    cv2 = _import_cv2()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: List[np.ndarray] = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    h, w = frames[0].shape[:2]
    return w, h, float(fps), frames


def iter_frames_from_jpeg_folder(folder: str) -> Tuple[int, int, float, List[np.ndarray]]:
    """Load all RGB frames from a folder of ``<idx>.jpg`` images."""
    jpg_exts = (".jpg", ".jpeg", ".JPG", ".JPEG")
    names = [p for p in os.listdir(folder) if p.endswith(jpg_exts)]
    if not names:
        raise RuntimeError(f"No JPEG frames found in {folder}")
    names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    frames: List[np.ndarray] = []
    for name in names:
        img = Image.open(os.path.join(folder, name)).convert("RGB")
        frames.append(np.array(img))
    h, w = frames[0].shape[:2]
    return w, h, 10.0, frames


def load_original_frames(video_path: str) -> Tuple[int, int, float, List[np.ndarray]]:
    if os.path.isdir(video_path):
        return iter_frames_from_jpeg_folder(video_path)
    return iter_frames_from_video(video_path)


def draw_overlay(
    frame_rgb: np.ndarray,
    mask: Optional[np.ndarray],
    bbox_xyxy: Optional[List[int]],
    obj_id: int,
    frame_idx: int,
) -> np.ndarray:
    """Return an RGB overlay with the mask (cyan tint) and bbox (green) drawn."""
    cv2 = _import_cv2()
    overlay = frame_rgb.copy()
    if mask is not None and mask.any():
        color = np.array([0, 255, 255], dtype=np.uint8)  # cyan
        blended = overlay.copy()
        blended[mask] = (0.5 * overlay[mask] + 0.5 * color).astype(np.uint8)
        overlay = blended
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = bbox_xyxy
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            f"id={obj_id}",
            (x1, max(y1 - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
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


def main() -> None:
    args = parse_args()

    print("Building SAM 3 tracker (no detector)...")
    predictor = build_sam3_tracker_only(
        checkpoint_path=args.checkpoint,
        apply_temporal_disambiguation=args.apply_temporal_disambiguation,
        trim_past_memory=args.trim_memory,
        device=args.device,
    )

    print(f"Initializing inference state from: {args.video}")
    inference_state = predictor.init_state(
        video_path=args.video,
        offload_video_to_cpu=args.offload_video_to_cpu,
        offload_state_to_cpu=args.offload_to_cpu,
    )
    video_h = inference_state["video_height"]
    video_w = inference_state["video_width"]
    num_frames = inference_state["num_frames"]
    print(f"  video: {video_w}x{video_h}, {num_frames} frames")

    box_norm = normalize_bbox_xyxy(
        args.bbox,
        bbox_format=args.bbox_format,
        is_normalized=args.bbox_normalized,
        video_w=video_w,
        video_h=video_h,
    )
    print(f"  seeding object id={args.obj_id} on frame {args.start_frame} "
          f"with normalized XYXY bbox {box_norm.tolist()[0]}")

    predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=args.start_frame,
        obj_id=args.obj_id,
        box=box_norm,
    )

    max_frame_num_to_track = (
        args.max_frames if args.max_frames is not None else num_frames
    )

    per_frame: dict = {}
    print("Propagating mask through the video...")
    for (
        frame_idx,
        obj_ids,
        _low_res_masks,
        video_res_masks,
        obj_scores,
    ) in predictor.propagate_in_video(
        inference_state=inference_state,
        start_frame_idx=args.start_frame,
        max_frame_num_to_track=max_frame_num_to_track,
        reverse=False,
        propagate_preflight=True,
    ):
        frame_entries = []
        # video_res_masks: [num_objects, 1, H, W] mask logits
        # obj_scores:      [num_objects, 1]       objectness logits
        masks_np = (video_res_masks > 0.0).squeeze(1).cpu().numpy()
        scores_np = obj_scores.sigmoid().squeeze(-1).float().cpu().numpy()
        for i, obj_id in enumerate(obj_ids):
            mask_bool = masks_np[i].astype(bool)
            bbox_xyxy_px = mask_to_bbox_xyxy(mask_bool)
            frame_entries.append(
                {
                    "obj_id": int(obj_id),
                    "score": float(scores_np[i]),
                    "bbox_xyxy_pixels": bbox_xyxy_px,
                    "present": bbox_xyxy_px is not None,
                }
            )
        per_frame[int(frame_idx)] = {
            "objects": frame_entries,
            "masks": masks_np,  # kept in-memory only; stripped before JSON dump
        }

    print(f"Tracked {len(per_frame)} frames.")

    if args.output_json is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        serializable = {
            "video": os.path.abspath(args.video),
            "video_height": video_h,
            "video_width": video_w,
            "num_frames": num_frames,
            "seed_frame": args.start_frame,
            "seed_bbox_xyxy_normalized": box_norm.tolist()[0],
            "frames": {
                str(idx): [
                    {k: v for k, v in obj.items()}
                    for obj in data["objects"]
                ]
                for idx, data in sorted(per_frame.items())
            },
        }
        with open(args.output_json, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Saved tracking results to {args.output_json}")

    if args.output_masks is not None:
        os.makedirs(args.output_masks, exist_ok=True)
        for idx, data in sorted(per_frame.items()):
            for i, obj in enumerate(data["objects"]):
                mask_bool = data["masks"][i]
                png_path = os.path.join(
                    args.output_masks,
                    f"frame{idx:06d}_obj{obj['obj_id']}.png",
                )
                Image.fromarray(mask_bool.astype(np.uint8) * 255, mode="L").save(
                    png_path
                )
        print(f"Saved per-frame masks to {args.output_masks}")

    if args.overlay_video is not None:
        cv2 = _import_cv2()
        print("Loading original frames for overlay rendering...")
        orig_w, orig_h, fps, orig_frames = load_original_frames(args.video)
        if (orig_w, orig_h) != (video_w, video_h):
            print(
                f"  note: overlay frame size {orig_w}x{orig_h} differs from "
                f"tracker-reported {video_w}x{video_h}; using overlay size."
            )
        os.makedirs(
            os.path.dirname(os.path.abspath(args.overlay_video)), exist_ok=True
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            args.overlay_video, fourcc, fps, (orig_w, orig_h)
        )
        try:
            for frame_idx, frame_rgb in enumerate(orig_frames):
                data = per_frame.get(frame_idx)
                if data is None or not data["objects"]:
                    overlay = draw_overlay(
                        frame_rgb, None, None, args.obj_id, frame_idx
                    )
                else:
                    obj = data["objects"][0]
                    mask_bool = data["masks"][0].astype(bool)
                    if mask_bool.shape != frame_rgb.shape[:2]:
                        mask_bool = cv2.resize(
                            mask_bool.astype(np.uint8),
                            (frame_rgb.shape[1], frame_rgb.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                    overlay = draw_overlay(
                        frame_rgb,
                        mask_bool,
                        obj["bbox_xyxy_pixels"],
                        obj["obj_id"],
                        frame_idx,
                    )
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        print(f"Saved overlay video to {args.overlay_video}")


if __name__ == "__main__":
    main()
