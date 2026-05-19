import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
import os 
# from evaluation import evaluate
# Load the ONNX model
session = ort.InferenceSession("/home/thuongnh/rf-detr/output/inference_model.onnx")
path = '/mnt/hdd10tb/Users/thuongnh/datasets/test_sat_personhand/images'
CLASSES = ['person', 'hand']
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# accuracy_stats = evaluate(inference, images_dir, annotations_file_path, inv_class_mapping, buffer_time=artifact_request.buffer_time, max_images=artifact_request.max_images, max_dets=artifact_request.max_dets)

# Prepare input image
def process_img(image, imgsz):
    # image = Image.open("image.jpg").convert("RGB")
    # image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = image.resize((imgsz, imgsz))  # Resize to model's input resolution
    image_array = np.array(image).astype(np.float32) / 255.0

    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std

    # Convert to NCHW format
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype(np.float32)
    return image_array

def box_cxcywh_to_xyxyn(x):
    cx, cy, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2
    return np.stack([xmin, ymin, xmax, ymax], axis=-1)

# Run inference
def post_process(outputs, origin_height, origin_width, confidence_threshold=0.3, max_number_boxes=300):
    """
    Post-process the model's output to extract bounding boxes and class information.
    Inspired by the PostProcess class in rfdetr/lwdetr.py: https://github.com/roboflow/rf-detr/blob/1.3.0/rfdetr/models/lwdetr.py#L701
    """
    # Get masks if instance segmentation
    if len(outputs) == 3:  
        masks = outputs[2]
    else:
        masks = None
    
    # Apply sigmoid activation
    prob = sigmoid(outputs[1]) 
    
    # Get detections with highest confidence and limit to max_number_boxes
    scores = np.max(prob, axis=2).squeeze()
    labels = np.argmax(prob, axis=2).squeeze()
    sorted_idx = np.argsort(scores)[::-1]
    scores = scores[sorted_idx][:max_number_boxes]
    labels = labels[sorted_idx][:max_number_boxes]
    boxes = outputs[0].squeeze()[sorted_idx][:max_number_boxes]
    if masks is not None:
        masks = masks.squeeze()[sorted_idx][:max_number_boxes]
    
    # Convert boxes from cxcywh to xyxyn format and scale to image size (i.e xyxyn -> xyxy)
    boxes = box_cxcywh_to_xyxyn(boxes)
    boxes[..., [0, 2]] *= origin_width
    boxes[..., [1, 3]] *= origin_height
    
    # Resize the masks to the original image size if available
    if masks is not None:
        new_w, new_h = origin_width, origin_height
        masks = np.stack([
            np.array(Image.fromarray(img).resize((new_w, new_h)))
            for img in masks
        ], axis=0)
        masks = (masks > 0).astype(np.uint8) * 255 
    
    # Filter detections based on the confidence threshold
    confidence_mask = scores > confidence_threshold
    scores = scores[confidence_mask]
    labels = labels[confidence_mask]
    boxes = boxes[confidence_mask]
    if masks is not None:
        masks = masks[confidence_mask]
    
    return scores, labels, boxes, masks


video_capture = cv2.VideoCapture("/home/thuongnh/ultralytics/test/1551232919461_62014.mp4")
if not video_capture.isOpened():
    raise RuntimeError("Failed to open video source: <SOURCE_VIDEO_PATH>")

# cap = cv2.VideoCapture(file_path)

# Get video properties
# fps = video_capture.get(cv2.CAP_PROP_FPS)
# orig_w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# orig_h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*"mp4v")
# out = cv2.VideoWriter("onnx_results.mp4", fourcc, fps, (orig_w, orig_h))

# frame_count = 0
# while True:
#     success, frame_bgr = video_capture.read()
#     if not success:
#         break
output_dir = 'output_onnx'

for img_path in os.listdir(path):
    fout = img_path.split('.')[0] + '.txt'
    f = open(os.path.join(output_dir, fout), 'w')
    # result_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    frame_rgb = Image.open(os.path.join(path,img_path)).convert("RGB")
    result_rgb = frame_rgb.copy()
    orig_w, orig_h = frame_rgb.size
    # detections = model.predict(frame_rgb, threshold=0.3)
    img = process_img(frame_rgb, 576)
    # print(type(img))
    outputs = session.run(None, {"input": img})
    # boxes, labels = outputs

    # print(boxes, labels)
    scores, labels, boxes, masks = post_process(outputs, orig_h, orig_w,confidence_threshold=0.3)

    draw = ImageDraw.Draw(result_rgb)
    font = ImageFont.load_default()

    # Loop over boxes and draw
    for i, box in enumerate(boxes.astype(int)):
        label = labels[i]
        # print(label)
        classes = CLASSES[label]
        conf = scores[i]
        # Use same color as mask but fully opaque for the outline
        # box_color = tuple(label_colors[label][:3])  # ignore alpha
        draw.rectangle(box.tolist(), outline="red", width=4)

        # Draw label text
        text_x = box[0] + 5
        text_y = box[1] + 5
        draw.text((text_x, text_y), 
        text=f"{classes} {round(conf.item(), 2)}", 
        fill='blue', font_size=20)
        f.write(f'{int(label)} {round(conf,2)} {box[0]} {box[1]} {box[2]} {box[3]}' + '\n')
    # result_rgb.save(os.path.join(output_dir,img_path))
#     frame_out = cv2.cvtColor(np.array(result_rgb), cv2.COLOR_RGB2BGR)
#     out.write(frame_out)
#     frame_count += 1

#     if frame_count % 10 == 0:
#         print(f"Processed {frame_count} frames...")

# video_capture.release()
# out.release()
# print("Video processing complete. Result saved as 'results_video.mp4'.")