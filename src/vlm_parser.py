"""
Module 2: OSD Telemetry Extraction Pipeline using YOLOv8 and Qwen2-VL.
"""

import gc
import json
import cv2
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from ultralytics import YOLO

PROMPT_TEXT = """
You are an expert OCR system specializing in reading low-resolution UAV OSD (On-Screen Display) telemetry fonts.

Examine the provided crop images and the highlighted video frame to extract exact visible telemetry data into a JSON object.

STRICT EXTRACTION RULES:
1. Extract exact visible text, numbers, and units (e.g. 'UJEE', '08:26', '20', '15.0v').
2. If a specific field is not present or cannot be read, set its value strictly to null.
3. NEVER write 'unknown', 0, or guess missing values.
4. Keep all extracted non-null values as string formatted values.

JSON Schema:
{
  "craft_name": null,
  "cell_voltage": null,
  "battery_voltage": null,
  "battery_percentage": null,
  "drawn_capacity": null,
  "efficiency": null,
  "current": null,
  "altitude": null,
  "vertical_speed": null,
  "ground_speed": null,
  "distance": null,
  "flight_time": null,
  "flight_mode": null,
  "arming_state": null,
  "compass": null,
  "rssi_signal": null,
  "satellites": null,
  "vtx_power": null,
  "latency_bitrate": null,
  "camera_settings": null,
  "system_warnings": null,
  "watermark_unit": null
}
"""


def process_telemetry_frame(
    video_path: str,
    yolo_weights: str = "best.pt",
    output_json: str = "osd_parsed_data.json",
):
    gc.collect()
    torch.cuda.empty_cache()

    # 1. Завантаження VLM Qwen2-VL
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)

    # 2. Отримання кадру з відео
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Не вдалося витягнути кадр з відео.")

    frame_path = "middle_frame.jpg"
    cv2.imwrite(frame_path, frame)

    # 3. Детекція через YOLOv8
    yolo_model = YOLO(yolo_weights)
    results = yolo_model.predict(source=frame_path, conf=0.25)[0]

    cropped_pil_images = []
    h, w, _ = frame.shape
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        x1, y1 = max(0, x1 - 8), max(0, y1 - 8)
        x2, y2 = min(w, x2 + 8), min(h, y2 + 8)

        crop = frame[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        cropped_pil_images.append(Image.fromarray(crop_rgb))

    annotated_frame = frame.copy()
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.cpu().numpy())
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    full_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    full_frame_pil = Image.fromarray(full_frame_rgb)
    full_frame_pil.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    # 4. Формування промпту
    content_list = [{"type": "text", "text": PROMPT_TEXT}]
    for img in cropped_pil_images:
        content_list.append({"type": "image", "image": img})
    content_list.append({"type": "image", "image": full_frame_pil})

    messages = [{"role": "user", "content": content_list}]
    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # 5. Інференс
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)

    del inputs
    torch.cuda.empty_cache()

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 6. Парсинг JSON
    try:
        clean_text = (
            output_text.split("JSON Schema:")[-1]
            if "JSON Schema:" in output_text
            else output_text
        )
        clean_text = clean_text.replace("```json", "").replace("```", "").strip()

        start_idx = clean_text.find("{")
        end_idx = clean_text.rfind("}")
        if start_idx != -1 and end_idx != -1:
            clean_text = clean_text[start_idx : end_idx + 1]

        parsed_json = json.loads(clean_text)
        print("\nУспішно згенеровано JSON:")
        print(json.dumps(parsed_json, indent=4, ensure_ascii=False))

        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(parsed_json, f, indent=4, ensure_ascii=False)

    except json.JSONDecodeError:
        print("Результат генерації (не вдалося розпарсити як JSON):")
        print(output_text)


if __name__ == "__main__":
    process_telemetry_frame("Video_1.mp4")
