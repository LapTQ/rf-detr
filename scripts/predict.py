import requests
import supervision as sv
from PIL import Image, ImageDraw, ImageFont, ImageOps
from rfdetr import RFDETRMedium

# from rfdetr.util.coco_classes import COCO_CLASSES
import os
from tqdm import tqdm

model = RFDETRMedium(
    pretrain_weights="/home/laptq/rf-detr/outputs/train/fs26/v1.medium.person_hand/checkpoint_best_total.pth",
    device="cuda:5",
)
path = "/home/laptq/laptq-fs26-shoplifting-detection/data/test_sat_personhand/images"
CLASSES = ["person", "hand"]

output_dir = "/home/laptq/rf-detr/outputs/output_torch"
os.makedirs(output_dir, exist_ok=True)
for img_name in tqdm(os.listdir(path)):
    fout = img_name.split(".")[0] + ".txt"
    f = open(os.path.join(output_dir, fout), "w")
    # result_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    frame_rgb = Image.open(os.path.join(path, img_name)).convert("RGB")
    result_rgb = frame_rgb.copy()
    orig_w, orig_h = frame_rgb.size
    detections = model.predict(frame_rgb, threshold=0.3)
    # print(detections)
    # labels = [
    #     f"{CLASSES[class_id]}"
    #     for class_id
    #     in detections.class_id
    # ]
    draw = ImageDraw.Draw(result_rgb)
    font = ImageFont.load_default()
    for result in detections:
        box, _, score, label, _, _ = result
        draw.rectangle(box.tolist(), outline="red", width=4)
        # print(label)
        classes = CLASSES[label]
        # Draw label text
        text_x = box[0] + 5
        text_y = box[1] + 5
        draw.text(
            (text_x, text_y),
            text=f"{classes} {round(score.item(), 2)}",
            fill="blue",
            font_size=20,
        )
        f.write(
            f"{int(label)} {round(score,2)} {round(box[0])} {round(box[1])} {round(box[2])} {round(box[3])}"
            + "\n"
        )
        # boxes = result.xyxy
    result_rgb.save(os.path.join(output_dir, img_name))
