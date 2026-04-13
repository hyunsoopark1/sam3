"""
Multi-Person Tracking with EfficientSAM3 (SAM 3.1 Object Multiplex)

Detects and tracks multiple people in a video using SAM 3.1's efficient
multiplex architecture. Supports text-based detection ("person"), optional
point/box refinement, and outputs an annotated video with per-person
segmentation masks, bounding boxes, and unique IDs.

Key features:
  - Automatic multi-person detection via text prompt
  - Efficient joint tracking using SAM 3.1 Object Multiplex (~7x faster)
  - Per-person colored segmentation masks with unique IDs
  - Bounding box overlay with confidence display
  - Exports annotated MP4 video + optional per-frame mask data
  - Supports MP4 video or JPEG frame directory as input

Usage:
  # Track people in an MP4 video (default text prompt: "person")
  python multi_person_tracking.py --video assets/videos/bedroom.mp4

  # Track people in a JPEG frame directory
  python multi_person_tracking.py --video assets/videos/0001

  # Custom output directory and confidence threshold
  python multi_person_tracking.py --video my_video.mp4 --output results/ --threshold 0.6

  # Save per-frame mask data as .npz files
  python multi_person_tracking.py --video my_video.mp4 --save-masks

  # Use a specific text prompt (e.g., "dancer" or "athlete")
  python multi_person_tracking.py --video my_video.mp4 --text "dancer"

  # Limit tracking to first N frames
  python multi_person_tracking.py --video my_video.mp4 --max-frames 100

Requirements:
  - CUDA-capable GPU
  - SAM 3.1 checkpoint (auto-downloaded from HuggingFace)
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch

# -- Color palette for tracked persons ----------------------------------------

PERSON_COLORS = [
    (230, 25, 75),    # red
    (60, 180, 75),    # green
    (255, 225, 25),   # yellow
    (0, 130, 200),    # blue
    (245, 130, 48),   # orange
    (145, 30, 180),   # purple
    (70, 240, 240),   # cyan
    (240, 50, 230),   # magenta
    (210, 245, 60),   # lime
    (250, 190, 212),  # pink
    (0, 128, 128),    # teal
    (220, 190, 255),  # lavender
    (170, 110, 40),   # brown
    (255, 250, 200),  # beige
    (128, 0, 0),      # maroon
    (170, 255, 195),  # mint
    (128, 128, 0),    # olive
    (255, 215, 180),  # apricot
    (0, 0, 128),      # navy
    (128, 128, 128),  # grey
]


def get_color(obj_id: int) -> tuple:
    """Return a consistent RGB color for a given object ID."""
    return PERSON_COLORS[obj_id % len(PERSON_COLORS)]


# -- Video I/O helpers --------------------------------------------------------


def load_video_frames(video_path: str):
    """Load video frames from an MP4 file or a directory of JPEG images.

    Returns:
        frames: list of numpy arrays (H, W, 3) in RGB
        fps: frames per second (default 30 for JPEG directories)
    """
    if os.path.isfile(video_path) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        print(f"Loaded {len(frames)} frames from {video_path} ({fps:.1f} FPS)")
        return frames, fps
    elif os.path.isdir(video_path):
        jpg_files = glob.glob(os.path.join(video_path, "*.jpg"))
        if not jpg_files:
            jpg_files = glob.glob(os.path.join(video_path, "*.png"))
        if not jpg_files:
            raise FileNotFoundError(f"No image files found in {video_path}")
        # Sort numerically by filename
        try:
            jpg_files.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        except ValueError:
            jpg_files.sort()
        frames = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in jpg_files]
        print(f"Loaded {len(frames)} frames from {video_path}")
        return frames, 30.0
    else:
        raise FileNotFoundError(f"Video path not found: {video_path}")


def render_tracking_frame(
    frame: np.ndarray,
    obj_ids: np.ndarray,
    masks: np.ndarray,
    boxes: np.ndarray,
    mask_alpha: float = 0.45,
) -> np.ndarray:
    """Overlay segmentation masks, bounding boxes, and IDs on a single frame.

    Args:
        frame: (H, W, 3) RGB image
        obj_ids: (N,) object IDs
        masks: (N, H, W) boolean masks
        boxes: (N, 4) bounding boxes in xywh format (relative or absolute)
        mask_alpha: transparency of the mask overlay

    Returns:
        annotated: (H, W, 3) RGB image with overlays
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]

    for idx, obj_id in enumerate(obj_ids):
        color = get_color(int(obj_id))
        mask = masks[idx].astype(bool)

        # Blend colored mask onto frame
        overlay = annotated.copy()
        overlay[mask] = color
        annotated = cv2.addWeighted(overlay, mask_alpha, annotated, 1 - mask_alpha, 0)

        # Draw mask contour
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, color, 2)

        # Compute bounding box in absolute pixels
        bx, by, bw, bh = boxes[idx]
        if max(bx, by, bw, bh) <= 1.0:
            bx, by, bw, bh = bx * w, by * h, bw * w, bh * h
        x1, y1 = int(bx), int(by)
        x2, y2 = int(bx + bw), int(by + bh)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw ID label with background
        label = f"Person {int(obj_id)}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y_top = max(y1 - th - baseline - 4, 0)
        cv2.rectangle(annotated, (x1, label_y_top), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 2, max(y1 - baseline - 2, th + 2)),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


def save_video(frames: list, output_path: str, fps: float):
    """Save a list of RGB frames as an MP4 video."""
    if not frames:
        print("No frames to save.")
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved output video to {output_path} ({len(frames)} frames, {fps:.1f} FPS)")


# -- Main tracking pipeline ---------------------------------------------------


def run_multi_person_tracking(args):
    """Main pipeline: load model, detect people, track, render, and save."""

    print("=" * 70)
    print("  Multi-Person Tracking with EfficientSAM3 (SAM 3.1 Multiplex)")
    print("=" * 70)

    # --- 1. Build the SAM 3.1 predictor ---
    print("\n[1/5] Building SAM 3.1 Multiplex predictor...")
    t0 = time.time()
    from sam3.model_builder import build_sam3_multiplex_video_predictor

    predictor = build_sam3_multiplex_video_predictor(
        max_num_objects=args.max_persons,
        default_output_prob_thresh=args.threshold,
    )
    print(f"       Model loaded in {time.time() - t0:.1f}s")

    # --- 2. Load video frames ---
    print(f"\n[2/5] Loading video from: {args.video}")
    video_frames, fps = load_video_frames(args.video)
    if args.max_frames and args.max_frames < len(video_frames):
        video_frames = video_frames[: args.max_frames]
        print(f"       Truncated to first {args.max_frames} frames")

    # --- 3. Start session and detect people ---
    print(f"\n[3/5] Starting tracking session (text prompt: \"{args.text}\")...")
    t0 = time.time()

    # Determine video source for predictor (needs path, not frames)
    video_source = args.video
    if os.path.isfile(args.video) and args.video.endswith(".mp4"):
        # For MP4, the predictor can handle it directly
        video_source = args.video
    elif os.path.isdir(args.video):
        video_source = args.video

    response = predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=video_source,
        )
    )
    session_id = response["session_id"]

    # Add text prompt on frame 0 to detect all people
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=args.text,
        )
    )
    initial_out = response["outputs"]
    num_detected = len(initial_out.get("out_obj_ids", []))
    print(f"       Detected {num_detected} person(s) on frame 0")
    print(f"       Detection took {time.time() - t0:.1f}s")

    # --- 4. Propagate tracking through all frames ---
    print(f"\n[4/5] Propagating tracking across {len(video_frames)} frames...")
    t0 = time.time()

    outputs_per_frame = {}
    # Store the initial frame output
    outputs_per_frame[0] = initial_out

    propagate_kwargs = dict(
        type="propagate_in_video",
        session_id=session_id,
    )
    if args.max_frames:
        propagate_kwargs["max_frame_num_to_track"] = args.max_frames

    for response in predictor.handle_stream_request(request=propagate_kwargs):
        frame_idx = response["frame_index"]
        outputs_per_frame[frame_idx] = response["outputs"]

    propagation_time = time.time() - t0
    frames_tracked = len(outputs_per_frame)
    track_fps = frames_tracked / propagation_time if propagation_time > 0 else 0
    print(f"       Tracked {frames_tracked} frames in {propagation_time:.1f}s ({track_fps:.1f} FPS)")

    # Collect all unique person IDs across the video
    all_person_ids = set()
    for out in outputs_per_frame.values():
        if "out_obj_ids" in out:
            all_person_ids.update(out["out_obj_ids"].tolist())
    print(f"       Total unique persons tracked: {len(all_person_ids)}")
    if all_person_ids:
        print(f"       Person IDs: {sorted(all_person_ids)}")

    # --- 5. Render annotated frames and save ---
    print(f"\n[5/5] Rendering annotated video...")
    t0 = time.time()

    os.makedirs(args.output, exist_ok=True)
    annotated_frames = []

    # Optional: save per-frame mask data
    mask_output_dir = None
    if args.save_masks:
        mask_output_dir = os.path.join(args.output, "masks")
        os.makedirs(mask_output_dir, exist_ok=True)

    num_frames_to_render = min(len(video_frames), max(outputs_per_frame.keys()) + 1) if outputs_per_frame else len(video_frames)

    for frame_idx in range(num_frames_to_render):
        frame = video_frames[frame_idx]
        out = outputs_per_frame.get(frame_idx)

        if out is not None and len(out.get("out_obj_ids", [])) > 0:
            obj_ids = out["out_obj_ids"]
            masks = out["out_binary_masks"]
            boxes = out["out_boxes_xywh"]

            # Convert to numpy if tensors
            if isinstance(obj_ids, torch.Tensor):
                obj_ids = obj_ids.cpu().numpy()
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()

            annotated = render_tracking_frame(frame, obj_ids, masks, boxes, args.mask_alpha)

            # Save per-frame masks if requested
            if mask_output_dir is not None:
                np.savez_compressed(
                    os.path.join(mask_output_dir, f"frame_{frame_idx:06d}.npz"),
                    obj_ids=obj_ids,
                    masks=masks,
                    boxes=boxes,
                )
        else:
            annotated = frame.copy()

        annotated_frames.append(annotated)

        # Progress indicator
        if (frame_idx + 1) % 50 == 0 or frame_idx == num_frames_to_render - 1:
            print(f"       Rendered {frame_idx + 1}/{num_frames_to_render} frames", end="\r")

    print()

    # Save annotated video
    output_video_path = os.path.join(args.output, "tracked_persons.mp4")
    save_video(annotated_frames, output_video_path, fps)

    # Save a summary image grid (first, middle, last frames)
    _save_summary_grid(annotated_frames, args.output)

    render_time = time.time() - t0
    print(f"       Rendering took {render_time:.1f}s")

    # --- Cleanup ---
    predictor.handle_request(
        request=dict(type="close_session", session_id=session_id)
    )

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("  Tracking Complete!")
    print("=" * 70)
    print(f"  Input:            {args.video}")
    print(f"  Text prompt:      \"{args.text}\"")
    print(f"  Persons detected: {len(all_person_ids)}")
    print(f"  Frames tracked:   {frames_tracked}")
    print(f"  Tracking speed:   {track_fps:.1f} FPS")
    print(f"  Output video:     {output_video_path}")
    if mask_output_dir:
        print(f"  Mask data:        {mask_output_dir}/")
    print(f"  Summary grid:     {os.path.join(args.output, 'summary_grid.jpg')}")
    print("=" * 70)


def _save_summary_grid(frames: list, output_dir: str):
    """Save a 1x3 summary grid showing first, middle, and last tracked frames."""
    if len(frames) < 1:
        return

    indices = [0]
    if len(frames) > 2:
        indices.append(len(frames) // 2)
    if len(frames) > 1:
        indices.append(len(frames) - 1)

    grid_frames = [frames[i] for i in indices]

    # Resize all to same height for concatenation
    target_h = min(f.shape[0] for f in grid_frames)
    resized = []
    for f in grid_frames:
        if f.shape[0] != target_h:
            scale = target_h / f.shape[0]
            new_w = int(f.shape[1] * scale)
            f = cv2.resize(f, (new_w, target_h))
        resized.append(f)

    grid = np.concatenate(resized, axis=1)
    grid_bgr = cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)
    grid_path = os.path.join(output_dir, "summary_grid.jpg")
    cv2.imwrite(grid_path, grid_bgr)


# -- CLI -----------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-Person Tracking with EfficientSAM3 (SAM 3.1 Multiplex)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python multi_person_tracking.py --video assets/videos/bedroom.mp4
  python multi_person_tracking.py --video assets/videos/0001
  python multi_person_tracking.py --video my_video.mp4 --text "dancer" --threshold 0.6
  python multi_person_tracking.py --video my_video.mp4 --save-masks --output results/
        """,
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video (MP4 file or directory of JPEG frames)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="person",
        help="Text prompt for detection (default: 'person')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tracking_output",
        help="Output directory for results (default: tracking_output/)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold for detection/mask output (default: 0.5)",
    )
    parser.add_argument(
        "--max-persons",
        type=int,
        default=32,
        help="Maximum number of persons to track simultaneously (default: 32)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: all)",
    )
    parser.add_argument(
        "--mask-alpha",
        type=float,
        default=0.45,
        help="Mask overlay transparency (default: 0.45)",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save per-frame mask data as .npz files",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_multi_person_tracking(args)
