# 📊 Roboflow Dataset Information

The dataset used for training the YOLOv8 OSD element detection model is hosted on Roboflow.

- **Workspace:** `olena-bondaruk`
- **Project:** `osd3` (Version 1)
- **Format:** YOLOv8 PyTorch

### 📥 Download via Python
You can download the dataset automatically using the Roboflow API:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="HaZNX79WvKULZPlwlAjq")
project = rf.workspace("olena-bondaruk").project("osd3")
version = project.version(1)
dataset = version.download("yolov8")
