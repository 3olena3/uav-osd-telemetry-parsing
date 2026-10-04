# 🧠 Trained Model Weights (`best.pt`)

To keep the repository lightweight, the trained model weights file (`best.pt`) is not tracked directly in Git.

### 📥 How to get `best.pt`:

1. **Train the model:** 
   Run `python src/train_yolo.py` to train YOLOv8 on the Roboflow dataset. The generated weights will be automatically saved at `runs/detect/train/weights/best.pt`.

2. **Run inference:** 
   Copy or move `best.pt` into the root directory before running `vlm_parser.py` or `video_cleaner.py`.
