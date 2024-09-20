import argparse
import os
import threading
import time
from collections import deque
from glob import glob

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from rich.progress import Progress

from x3d import X3D


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_path", type=str, help="path to video file")
    parser.add_argument("-m", "--model_path", type=str, help="path to model file")
    return parser.parse_args()


def preprocess_frame(frame, transform):
    # Apply the albumentations transform to the frame
    frame = transform(image=frame)["image"]
    # Add batch dimension
    frame = frame.unsqueeze(0)  # (1, channels, height, width)
    return frame


def inference_thread(model, device, frame_buffer, result_buffer, lock):
    while True:
        # Wait until we have enough frames for inference
        if len(frame_buffer) >= 16:
            # Take and remove the first 16 frames
            with lock:
                frames = [frame_buffer.popleft() for _ in range(16)]

            # Stack frames along the temporal dimension
            # Shape: (1, 3, 16, 312, 312)
            input_tensor = torch.stack(frames, dim=2).to(device)

            # Perform inference
            with torch.no_grad():
                pred = model(input_tensor)

            # Store the result
            with lock:
                result_buffer.append(pred.detach().cpu().numpy()[0])


def draw_results(frame, result, is_live):
    result = np.array(result)
    none, smoke, fire = result[0], result[1], result[2]
    cls = result.argmax()

    results.pop(0)
    results.append(int(cls))
    voted = np.bincount(np.array(results)).argmax()

    color = (255, 0, 0) if is_live else (0, 0, 0)

    cv2.putText(
        frame,
        f"fire: {fire:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )
    cv2.putText(
        frame,
        f"smoke: {smoke:.2f}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )
    cv2.putText(
        frame,
        f"none: {none:.2f}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )
    cv2.putText(
        frame,
        f"current class: {cls}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )
    cv2.putText(
        frame,
        f"total class: {results}",
        (10, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )
    cv2.putText(
        frame,
        f"voted class: {CLASSES[voted]}",
        (10, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )


results = [0] * 5
CLASSES = ["NONE", "SMOKE", "FIRE"]


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model
    x3d = X3D.load_from_checkpoint(args.model_path) if args.model_path else X3D()
    x3d.to(device)
    x3d.eval()
    print("Model loaded!")

    # Define the albumentations transform
    transform = A.Compose(
        [
            A.Resize(312, 312),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    if os.path.isdir(args.video_path):
        video_files = glob(f"{args.video_path}/*.mp4")
    else:
        video_files = [args.video_path]

    os.makedirs("outputs", exist_ok=True)

    # Open the video file
    for video_file in video_files:
        print(f"Processing video: {video_file}")
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        # Video Writer setup
        outname = f"outputs/{os.path.basename(video_file)}.mp4"
        fps = 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Total frames:", total_frames)
        print("Bathes: ", total_frames // 16, "+", total_frames % 16, "frames")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            outname,
            fourcc,
            fps,
            (frame_width, frame_height),
        )
        print(f"Output video: {outname}")

        frame_buffer = deque(maxlen=32)  # Buffer to store frames for temporal input
        result_buffer = deque()  # Buffer to store results
        lock = threading.Lock()

        # Start the inference thread
        inference_thread_handle = threading.Thread(
            target=inference_thread,
            args=(x3d, device, frame_buffer, result_buffer, lock),
        )
        inference_thread_handle.daemon = True
        inference_thread_handle.start()

        result = 0, 0, 0
        # Frame read and processing loop
        with Progress() as progress:
            task_frames = progress.add_task(
                "[green]Processing frames...", total=total_frames
            )
            task_result = progress.add_task(
                "[blue]Displaying results...", total=total_frames // 16
            )

            while cap.isOpened():
                ret, frame = cap.read()
                progress.update(task_frames, advance=1)
                if not ret:
                    break

                # Preprocess the frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                frame_tensor = preprocess_frame(frame_rgb, transform)

                with lock:
                    frame_buffer.append(frame_tensor)

                # Display the result if available
                if len(result_buffer) > 0:
                    progress.update(task_result, advance=1)
                    with lock:
                        result = result_buffer.popleft()

                    draw_results(frame, result, is_live=True)
                else:
                    draw_results(frame, result, is_live=False)

                # Write the frame to the output video
                out.write(frame)

                # Control the frame rate to 30 FPS
                time.sleep(1 / 30.0)

        result_buffer.clear()
        cap.release()
        out.release()


if __name__ == "__main__":
    main()
