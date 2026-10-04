"""
Module 3: Video Inpainting / OSD Removal Pipeline using YOLOv8 and OpenCV.
"""

import gc
import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO


def remove_osd_from_video(
    input_video_path: str,
    output_video_path: str = "clean_video_without_osd.mp4",
    weights_path: str = "best.pt",
    padding: int = 6,
):
    gc.collect()
    torch.cuda.empty_cache()

    trained_model = YOLO(weights_path)
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        raise ValueError(f"Не вдалося відкрити відео: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(
        f"Розпочинаємо обробку: {total_frames} кадрів, {width}x{height} @ {fps:.2f} FPS"
    )

    for _ in tqdm(range(total_frames), desc="Inpainting Video"):
        ret, frame = cap.read()
        if not ret:
            break

        results = trained_model.predict(source=frame, conf=0.25, verbose=False)[0]
        mask = np.zeros((height, width), dtype=np.uint8)

        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.cpu().numpy())
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        clean_frame = cv2.inpaint(
            frame, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA
        )
        out.write(clean_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nОбробку завершено! Збережено у: {output_video_path}")


if __name__ == "__main__":
    remove_osd_from_video("Video_1.mp4")
