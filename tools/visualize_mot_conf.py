#!/usr/bin/env python3
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple

def parse_pred_file(pred_path: str) -> Dict[int, List[Tuple[int, float, float, float, float, float]]]:
    """
    Parse prediction file.
    Expected columns per line: frame(1-based), id, x, y, w, h, confidence
    """
    per_frame: Dict[int, List[Tuple[int, float, float, float, float, float]]] = {}
    with open(pred_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 7:
                continue
            try:
                frame1 = int(parts[0])
                track_id = int(float(parts[1]))
                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])
                conf = float(parts[6])
            except Exception:
                continue
            per_frame.setdefault(frame1, []).append((track_id, x, y, w, h, conf))
    return per_frame

def draw_boxes_with_transparency(image: np.ndarray, boxes: List[Tuple[int, float, float, float, float, float]]) -> np.ndarray:
    """
    Draw hollow boxes with transparency based on confidence.
    """
    out_img = image.copy()
    
    # Sort boxes by confidence so stronger ones are drawn on top if they overlap
    boxes = sorted(boxes, key=lambda x: x[5])

    for track_id, x, y, w, h, conf in boxes:
        # Clamping confidence to [0, 1] just in case
        alpha = max(0.0, min(1.0, conf))
        
        # Define box coordinates
        pt1 = (int(x), int(y))
        pt2 = (int(x + w), int(y + h))
        
        # Color: Red (BGR)
        color = (0, 0, 255) 
        thickness = 2
        
        # To draw a transparent border, we need to manually blend the pixels 
        # that would form the rectangle border.
        
        # 1. Define ROI that covers the border area
        roi_x1 = max(0, pt1[0] - thickness)
        roi_y1 = max(0, pt1[1] - thickness)
        roi_x2 = min(image.shape[1], pt2[0] + thickness)
        roi_y2 = min(image.shape[0], pt2[1] + thickness)
        
        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
            # Extract ROI from source image
            roi_src = out_img[roi_y1:roi_y2, roi_x1:roi_x2]
            
            # Create a local mask for the lines
            local_mask = np.zeros(roi_src.shape[:2], dtype=np.uint8)
            
            # Adjust coords to ROI relative
            l_pt1 = (pt1[0] - roi_x1, pt1[1] - roi_y1)
            l_pt2 = (pt2[0] - roi_x1, pt2[1] - roi_y1)
            
            # Draw solid white rectangle on mask to define line pixels
            cv2.rectangle(local_mask, l_pt1, l_pt2, 255, thickness)
            
            # Where mask is 255, we want to blend color.
            # Where mask is 0, keep original.
            
            # Create color overlay for ROI
            color_overlay = np.full_like(roi_src, color, dtype=np.uint8)
            
            # Expand mask to 3 channels for broadcasting
            mask_bool = (local_mask > 0) # boolean mask
            mask_3c = np.stack([mask_bool] * 3, axis=-1)
            
            # Blending calculation
            # result = src * (1 - alpha) + color * alpha  (for line pixels)
            # result = src                                (for non-line pixels)
            # This simplifies to: result = src + (color - src) * alpha
            # But we only apply it where mask is True.
            
            # We can use cv2.addWeighted logic on arrays
            # Extract line pixels from src
            src_pixels = roi_src[mask_bool] # shape (N, 3)
            overlay_pixels = color_overlay[mask_bool] # shape (N, 3), actually just color
            
            # Blend float
            blended_pixels = src_pixels.astype(float) * (1.0 - alpha) + overlay_pixels.astype(float) * alpha
            
            # Assign back
            roi_src[mask_bool] = blended_pixels.astype(np.uint8)
            
            # Put modified ROI back
            out_img[roi_y1:roi_y2, roi_x1:roi_x2] = roi_src

            # Text label (ID only)
            # Show ID regardless of confidence (solid white)
            label = f"{track_id}"
            font_scale = 0.5
            thickness_font = 1
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_font)
            
            # Position text above box
            text_pos = (pt1[0], max(10, pt1[1]-5))
            
            # Draw text directly on top (solid white for readability)
            cv2.putText(out_img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness_font)
    
    return out_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, default="track_result/gt_refer/0005/black-cars-in-the-left/predict_with_conf.txt", help="Path to predict_with_conf.txt")
    parser.add_argument("--image_root", type=str, default="/data/sq_2023/refer_kitti/KITTI/training/image_02", help="Path to KITTI image_02 root")
    parser.add_argument("--output", type=str, default="track_result/gt_refer/0005/black-cars-in-the-left/vis/output.mp4", help="Output video path")
    parser.add_argument("--fps", type=int, default=10, help="Video FPS")
    parser.add_argument("--seq", type=str, default=None, help="Sequence ID (e.g. 0005). If None, inferred from path.")
    args = parser.parse_args()

    # Infer sequence if not provided
    seq_id = args.seq
    if seq_id is None:
        # Try to find 4-digit sequence in path parts
        parts = os.path.abspath(args.pred_path).split(os.sep)
        for p in parts:
            if len(p) == 4 and p.isdigit():
                seq_id = p
                break
    
    if seq_id is None:
        print("Error: Could not infer sequence ID from path. Please provide --seq.")
        return

    print(f"Sequence ID: {seq_id}")
    
    # Parse predictions
    preds = parse_pred_file(args.pred_path)
    if not preds:
        print("No predictions found.")
        return

    frames = sorted(preds.keys())
    min_frame = min(frames)
    max_frame = max(frames)
    
    print(f"Frames: {min_frame} to {max_frame}, Total frames with detections: {len(frames)}")

    # Prepare video writer
    # Need to read one image to get size
    # Check 1-based or 0-based image naming. KITTI usually 000000.png, 0-based.
    # The txt file is usually 1-based (MOT format).
    # refer_llm_eval_from_mot.py uses: fidx_img = frame - 1 if int(args.mot_frame_start_one) == 1 else frame
    # Usually predict_with_conf.txt from this pipeline is 1-based.
    
    first_frame_idx = frames[0]
    # Try 0-based file
    img_path_0 = os.path.join(args.image_root, seq_id, f"{first_frame_idx-1:06d}.png")
    if not os.path.exists(img_path_0):
        # Try 1-based (unlikely for KITTI file naming, but maybe the pred frame is 0-based?)
        img_path_0 = os.path.join(args.image_root, seq_id, f"{first_frame_idx:06d}.png")
    
    if not os.path.exists(img_path_0):
        print(f"Error: Could not find first image at {img_path_0}")
        # Try finding ANY image in the folder to get size
        seq_dir = os.path.join(args.image_root, seq_id)
        if not os.path.isdir(seq_dir):
             print(f"Error: Sequence directory not found: {seq_dir}")
             return
        files = os.listdir(seq_dir)
        if not files:
             print("Error: No files in sequence directory.")
             return
        img_path_0 = os.path.join(seq_dir, sorted(files)[0])

    img0 = cv2.imread(img_path_0)
    if img0 is None:
        print(f"Error: Failed to load image {img_path_0}")
        return
        
    H, W, _ = img0.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (W, H))
    
    # Iterate ALL frames in range to avoid skipping empty frames in video
    # Assume continuous video from min to max (or start 0?)
    # Usually start at 0.
    
    # We'll range from 0 (or min_pred - something) to max_pred.
    # KITTI usually starts at 0.
    
    # Let's find all images in the directory to cover full sequence if possible
    # Or just cover the range of predictions.
    # Safest is cover predictions range, maybe slightly padded.
    # But usually we want the whole video.
    
    # Let's just iterate from min_frame to max_frame for now.
    
    # Adjust for 0-based image indexing
    # We assume 'frame' in txt is 1-based, matching KITTI conventions in this repo.
    # So image file is (frame-1).png
    
    for f in tqdm(range(min_frame, max_frame + 1)):
        # Load image
        # Assuming 1-based frame index in txt, so image is f-1
        img_file = os.path.join(args.image_root, seq_id, f"{f-1:06d}.png")
        if not os.path.exists(img_file):
            # Fallback if maybe frames are 0-based in txt?
            img_file_alt = os.path.join(args.image_root, seq_id, f"{f:06d}.png")
            if os.path.exists(img_file_alt):
                img_file = img_file_alt
            else:
                # Black frame if missing
                frame_img = np.zeros((H, W, 3), dtype=np.uint8)
                print(f"Warning: Image missing for frame {f}")
        else:
            frame_img = cv2.imread(img_file)
            
        if frame_img is None:
            frame_img = np.zeros((H, W, 3), dtype=np.uint8)

        # Get detections
        dets = preds.get(f, [])
        
        # Draw
        vis_img = draw_boxes_with_transparency(frame_img, dets)
        
        out.write(vis_img)

    out.release()
    print(f"Video saved to {args.output}")

if __name__ == "__main__":
    main()

