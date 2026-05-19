import collections
import contextlib
import os
import time
from collections import OrderedDict

import tensorrt as trt
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2
# Load the ONNX model
model = "/home/thuongnh/rf-detr/output/inference_model.engine"
CLASSES = ['person', 'hand']
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class TimeProfiler(contextlib.ContextDecorator):
    def __init__(self):
        self.total = 0

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.total += self.time() - self.start

    def reset(self):
        self.total = 0

    def time(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()


class TRTInference(object):
    def __init__(
        self, engine_path, device="cuda:0", backend="torch", max_batch_size=32, verbose=False
    ):
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size

        self.logger = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger(trt.Logger.INFO)

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.bindings = self.get_bindings(
            self.engine, self.context, self.max_batch_size, self.device
        )
        self.bindings_addr = OrderedDict((n, v.ptr) for n, v in self.bindings.items())
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()
        self.time_profile = TimeProfiler()

    def load_engine(self, path):
        trt.init_libnvinfer_plugins(self.logger, "")
        with open(path, "rb") as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def get_input_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                names.append(name)
        return names

    def get_output_names(self):
        names = []
        for _, name in enumerate(self.engine):
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                names.append(name)
        return names

    def get_bindings(self, engine, context, max_batch_size=32, device=None) -> OrderedDict:
        Binding = collections.namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        bindings = OrderedDict()

        for i, name in enumerate(engine):
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))

            if shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, shape)

            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = Binding(name, dtype, shape, data, data.data_ptr())

        return bindings

    def run_torch(self, blob):
        # print(self.input_names)
        for n in self.input_names:
            # print(n)
            # print(blob[n].dtype)
            if blob[n].dtype is not self.bindings[n].data.dtype:
                blob[n] = blob[n].to(dtype=self.bindings[n].data.dtype)
            if self.bindings[n].shape != blob[n].shape:
                self.context.set_input_shape(n, blob[n].shape)
                self.bindings[n] = self.bindings[n]._replace(shape=blob[n].shape)

            assert self.bindings[n].data.dtype == blob[n].dtype, "{} dtype mismatch".format(n)

        self.bindings_addr.update({n: blob[n].data_ptr() for n in self.input_names})
        self.context.execute_v2(list(self.bindings_addr.values()))
        outputs = {n: self.bindings[n].data for n in self.output_names}

        return outputs

    def __call__(self, blob):
        if self.backend == "torch":
            return self.run_torch(blob)
        else:
            raise NotImplementedError("Only 'torch' backend is implemented.")

    def synchronize(self):
        if self.backend == "torch" and torch.cuda.is_available():
            torch.cuda.synchronize()

# Prepare input image
def process_img(frame, imgsz):
    # image = Image.open("image.jpg").convert("RGB")
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
fps = video_capture.get(cv2.CAP_PROP_FPS)
orig_w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("trt_results.mp4", fourcc, fps, (orig_w, orig_h))
m = TRTInference(model, device='cuda:0')

frame_count = 0
while True:
    success, frame_bgr = video_capture.read()
    if not success:
        break

    result_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    # detections = model.predict(frame_rgb, threshold=0.3)
    img = process_img(frame_bgr, 576)

    # print(type(img))
    outputs = m(img)
    # boxes, labels = outputs

    # print(boxes, labels)
    scores, labels, boxes, masks = post_process(outputs, orig_h, orig_w)

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

    frame_out = cv2.cvtColor(np.array(result_rgb), cv2.COLOR_RGB2BGR)
    out.write(frame_out)
    frame_count += 1

    if frame_count % 10 == 0:
        print(f"Processed {frame_count} frames...")

video_capture.release()
out.release()
print("Video processing complete. Result saved as 'results_video.mp4'.")