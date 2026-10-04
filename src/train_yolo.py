"""
Module 1: Dataset downloading from Roboflow and YOLOv8 Model Training.
"""

from roboflow import Roboflow
from ultralytics import YOLO


def train_yolo_model(api_key: str, epochs: int = 50, imgsz: int = 640):
    # 1. Завантаження датасету
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("olena-bondaruk").project("osd3")
    version = project.version(1)
    dataset = version.download("yolov8")

    # 2. Навчання YOLOv8
    model = YOLO("yolov8n.pt")
    model.train(data=f"{dataset.location}/data.yaml", epochs=epochs, imgsz=imgsz)
    print("Навчання успішно завершено! Файл ваг збережено в runs/detect/train/weights/best.pt")


if __name__ == "__main__":
    API_KEY = "HaZNX79WvKULZPlwlAjq"
    train_yolo_model(api_key=API_KEY)
