import tensorrt as trt
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
import cv2

import collections
import contextlib
import os
import time
from collections import OrderedDict

import torch.nn as nn
import torchvision.transforms.functional as TF
from profiler import CUDAProfiler
import time

import torch
from typing import List, Optional, Dict
from contextlib import contextmanager
import numpy as np


class CUDAProfiler:
    """Hardware-accurate CUDA profiler with stream-aware timing

    This profiler provides two timing modes:
    1. Synchronous profiling for standard operations
    2. Asynchronous profiling for CUDA graphs (avoids interference)
    """

    def __init__(self, stream: Optional[torch.cuda.Stream] = None):
        self.timings: List[float] = []
        self.stream = stream  # Allow profiling on specific streams
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    def set_stream(self, stream: torch.cuda.Stream):
        """Set the stream to profile on"""
        self.stream = stream

    @contextmanager
    def profile(self, stream: Optional[torch.cuda.Stream] = None):
        """Synchronous profiling for standard operations

        Use this for regular PyTorch operations, TensorRT standard execution, etc.
        This method handles synchronization automatically.

        Args:
            stream: Optional stream to profile on. If None, uses self.stream or current stream.
        """
        # Use provided stream, or fall back to instance stream, or current stream
        target_stream = stream or self.stream

        if target_stream is not None:
            # Record events on the specific stream for accurate timing
            self._start_event.record(target_stream)

            # Yield control back to the caller
            yield

            # Record end event on the same stream
            self._end_event.record(target_stream)

            # Only synchronize the target stream (more efficient than global sync)
            target_stream.synchronize()
        else:
            # Fallback to default stream timing
            self._start_event.record()

            yield

            self._end_event.record()
            torch.cuda.synchronize()

        # Calculate and store elapsed time
        try:
            elapsed_ms = self._start_event.elapsed_time(self._end_event)
            self.timings.append(elapsed_ms)
        except RuntimeError as e:
            # Handle CUDA errors gracefully (e.g., context corruption)
            print(f"Timing measurement failed: {e}")
            # Don't append invalid timing

    @contextmanager
    def profile_async(self, stream: Optional[torch.cuda.Stream] = None):
        """Asynchronous profiling for CUDA graphs and async operations

        Use this for CUDA graphs, async kernel launches, etc.
        This method records timing events but doesn't synchronize within the context.
        Call get_last_timing_async() after ensuring the stream is complete.

        Example:
            with profiler.profile_async(stream=my_stream):
                cuda_graph.replay()
            my_stream.synchronize()  # Ensure completion
            timing = profiler.get_last_timing_async()
        """
        target_stream = stream or self.stream

        if target_stream is not None:
            self._start_event.record(target_stream)

            yield

            self._end_event.record(target_stream)
            # Don't synchronize here - let caller handle it for async operations
        else:
            self._start_event.record()

            yield

            self._end_event.record()

    def get_last_timing_async(self) -> Optional[float]:
        """Get timing from async profiling after stream completion

        Call this after ensuring the stream has completed (e.g., via stream.synchronize()).

        Returns:
            Timing in milliseconds if available, None if events aren't ready
        """
        try:
            # Check if events are ready (non-blocking)
            if self._end_event.query():
                elapsed_ms = self._start_event.elapsed_time(self._end_event)
                self.timings.append(elapsed_ms)
                return elapsed_ms
            else:
                return None  # Events not ready yet
        except RuntimeError as e:
            print(f"Async timing measurement failed: {e}")
            return None

    def get_stats(self) -> Dict[str, float]:
        """Compute statistics from collected timings"""
        if not self.timings:
            raise ValueError(
                "No timings recorded. Use .profile() context manager first."
            )

        timings_array = np.array(self.timings)

        return {
            "mean": np.mean(timings_array),
            "median": np.median(timings_array),
            "min": np.min(timings_array),
            "max": np.max(timings_array),
            "std": np.std(timings_array),
            "p90": np.percentile(timings_array, 90),
            "p95": np.percentile(timings_array, 95),
            "p99": np.percentile(timings_array, 99),
            "count": len(self.timings),
        }

    def print_stats(self, name: Optional[str] = None):
        """Pretty print statistics"""
        try:
            stats = self.get_stats()

            if name:
                print(f"\n=== {name} ===")
            else:
                print("\n=== Profiling Results ===")

            print(f"Samples: {stats['count']}")
            print(f"Mean:    {stats['mean']:.3f} ms")
            print(f"Median:  {stats['median']:.3f} ms")
            print(f"Min:     {stats['min']:.3f} ms")
            print(f"Max:     {stats['max']:.3f} ms")
            print(f"Std:     {stats['std']:.3f} ms")
            print(f"P90:     {stats['p90']:.3f} ms")
            print(f"P95:     {stats['p95']:.3f} ms")
            print(f"P99:     {stats['p99']:.3f} ms")
        except ValueError as e:
            print(f"Cannot print stats: {e}")

    def reset(self):
        """Clear all recorded timings"""
        self.timings.clear()

    def get_last_timing(self) -> float:
        """Get the most recent timing"""
        if not self.timings:
            raise ValueError("No timings recorded.")
        return self.timings[-1]


# Additional utility for benchmarking specific operations
class StreamAwareBenchmark:
    """Utility for benchmarking operations with proper stream handling"""

    @staticmethod
    def time_operation(
        operation_fn,
        stream: torch.cuda.Stream,
        iterations: int = 100,
        use_async: bool = False,
    ):
        """Time a specific operation on a stream

        Args:
            operation_fn: Function to time
            stream: CUDA stream to run on
            iterations: Number of iterations
            use_async: Whether to use async profiling (for CUDA graphs)
        """
        profiler = CUDAProfiler(stream)

        # Warmup
        for _ in range(10):
            with torch.cuda.stream(stream):
                operation_fn()
            stream.synchronize()

        # Actual timing
        for _ in range(iterations):
            with torch.cuda.stream(stream):
                if use_async:
                    with profiler.profile_async():
                        operation_fn()
                    stream.synchronize()
                    profiler.get_last_timing_async()
                else:
                    with profiler.profile():
                        operation_fn()

        return profiler.get_stats()

    @staticmethod
    def compare_cuda_graph_vs_standard(
        cuda_graph_fn, standard_fn, stream: torch.cuda.Stream, iterations: int = 100
    ):
        """Compare CUDA graph vs standard execution performance"""

        print("Benchmarking Standard Execution...")
        stats_standard = StreamAwareBenchmark.time_operation(
            standard_fn, stream, iterations, use_async=False
        )

        print("Benchmarking CUDA Graph Execution...")
        stats_graph = StreamAwareBenchmark.time_operation(
            cuda_graph_fn, stream, iterations, use_async=True
        )

        print(f"\nStandard Execution - Mean: {stats_standard['mean']:.3f} ms")
        print(f"CUDA Graph Execution - Mean: {stats_graph['mean']:.3f} ms")
        print(
            f"CUDA Graph Speedup: {stats_standard['mean'] / stats_graph['mean']:.2f}x"
        )

        return stats_standard, stats_graph


def build_engine(model_path, engine_path, use_fp16=False):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)

    config = builder.create_builder_config()
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(EXPLICIT_BATCH)

    parser = trt.OnnxParser(network, logger)

    with open(model_path, "rb") as f:
        model_data = f.read()

    if not parser.parse(model_data):
        print("Failed to parse ONNX model")
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return None

    # Create optimization profile to fix dynamic batch dimensions
    profile = builder.create_optimization_profile()

    # Handle dynamic input shapes - fix batch size to 1
    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_shape = input_tensor.shape
        print(f"Input {i} ({input_tensor.name}): {input_shape}")

        # Check if batch dimension is dynamic (typically -1)
        if input_shape[0] == -1:
            # Fix batch size to 1
            fixed_shape = (1,) + tuple(input_shape[1:])
            print(f"  Setting fixed batch shape: {fixed_shape}")

            # Set min, optimal, and max shapes all to batch size 1
            profile.set_shape(input_tensor.name, fixed_shape, fixed_shape, fixed_shape)

    # Add the optimization profile to the configuration
    config.add_optimization_profile(profile)

    print(f"Building engine from {model_path} to {engine_path}")
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        print("Failed to build engine")
        return None

    print(f"Engine built successfully")

    with open(engine_path, "wb") as f:
        f.write(engine)

    return engine


class TRTInference:
    def __init__(
        self,
        engine_path: str,
        image_input_name: str | None = None,
        use_cuda_graph: bool = True,
        prediction_type: str = "bbox",
    ):
        logger = trt.Logger()
        trt.init_libnvinfer_plugins(logger, "")

        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

        print(
            f"Using {self.engine.num_aux_streams} auxiliary streams for TensorRT inference"
        )

        self.context = self.engine.create_execution_context()

        # Create dedicated CUDA stream for inference
        self.torch_stream = torch.cuda.Stream()
        self.cuda_stream_ptr = self.torch_stream.cuda_stream

        # Create separate stream for warm-up to avoid capture state conflicts
        self.warmup_stream = torch.cuda.Stream()
        self.warmup_stream_ptr = self.warmup_stream.cuda_stream
        self.use_cuda_graph = use_cuda_graph
        self.cuda_graph_compatible = (
            True  # Track if model is compatible with CUDA graphs
        )

        names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]

        self.input_names = [
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            name
            for name in names
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        self.output_shapes = [
            tuple(self.engine.get_tensor_shape(name)) for name in self.output_names
        ]

        # Initialize persistent tensors for CUDA graphs
        self.initialize_persistent_tensors()

        if len(self.input_names) != 1 and image_input_name is None:
            raise ValueError(
                "Model has multiple inputs, but no image input name was provided"
            )
        elif len(self.input_names) == 1 and image_input_name is not None:
            assert (
                image_input_name in self.input_names
            ), f"Image input name {image_input_name} not found in model inputs"

        self.image_input_name = (
            image_input_name if image_input_name is not None else self.input_names[0]
        )
        self.image_input_shape = tuple(
            self.engine.get_tensor_shape(self.image_input_name)
        )

        # CUDA graph management
        self.graph_cache = {}  # Cache graphs for different input shapes
        self.current_input_shape = None

        self.profiler = CUDAProfiler(stream=self.torch_stream)  # Stream-aware profiling

        print(f"TensorRT inference initialized with CUDA graphs: {self.use_cuda_graph}")
        if self.use_cuda_graph:
            print(
                "Note: CUDA graphs will be tested on first inference. If incompatible, will fall back to standard execution."
            )

        self.prediction_type = prediction_type

    def initialize_persistent_tensors(self):
        """Initialize persistent PyTorch tensors and bind them to TensorRT"""
        self.persistent_tensors = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))  # Convert Dims to tuple
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            # Convert numpy dtype to torch dtype
            if dtype == np.float32:
                torch_dtype = torch.float32
            elif dtype == np.float16:
                torch_dtype = torch.float16
            elif dtype == np.int32:
                torch_dtype = torch.int32
            elif dtype == np.int64:
                torch_dtype = torch.int64
            else:
                torch_dtype = torch.float32  # Default fallback

            # Create persistent tensor
            tensor = torch.empty(shape, dtype=torch_dtype, device="cuda")
            self.persistent_tensors[name] = tensor

            # Bind tensor address to TensorRT context
            self.context.set_tensor_address(name, tensor.data_ptr())

            print(f"Allocated persistent tensor '{name}': {shape} ({torch_dtype})")

    def _reallocate_tensor_for_shape(self, tensor_name: str, new_shape: tuple):
        """Reallocate a tensor for a new shape and update TensorRT binding"""
        current_tensor = self.persistent_tensors[tensor_name]

        # Create new tensor with the same dtype
        new_tensor = torch.empty(new_shape, dtype=current_tensor.dtype, device="cuda")

        # Update our reference and TensorRT binding
        self.persistent_tensors[tensor_name] = new_tensor
        self.context.set_tensor_address(tensor_name, new_tensor.data_ptr())

        print(
            f"Reallocated tensor '{tensor_name}' from {tuple(current_tensor.shape)} to {new_shape}"
        )

    def _capture_cuda_graph(self, input_shape: tuple):
        """Capture a CUDA graph for the given input shape"""
        print(f"Capturing CUDA graph for input shape: {input_shape}")

        # Update input tensor shape if needed
        current_input_shape = tuple(
            self.persistent_tensors[self.image_input_name].shape
        )
        if input_shape != current_input_shape:
            self._reallocate_tensor_for_shape(self.image_input_name, input_shape)

        try:
            # IMPORTANT: Warm-up execution on SEPARATE stream (not capture stream or default)
            print("Performing warm-up execution before graph capture...")

            # Do warm-up runs on separate warm-up stream to avoid capture state confusion
            for _ in range(3):
                success = self.context.execute_async_v3(self.warmup_stream_ptr)
                if not success:
                    raise RuntimeError("TensorRT warm-up execution failed")
                self.warmup_stream.synchronize()  # Sync the warm-up stream

            # Now capture the CUDA graph on our dedicated stream
            print("Starting CUDA graph capture...")
            graph = torch.cuda.CUDAGraph()

            # Capture on the dedicated stream
            with torch.cuda.graph(graph, stream=self.torch_stream):
                success = self.context.execute_async_v3(self.cuda_stream_ptr)
                if not success:
                    raise RuntimeError("TensorRT execution failed during graph capture")

            # Cache the graph
            shape_key = input_shape
            self.graph_cache[shape_key] = graph

            print(f"Successfully captured CUDA graph for shape {input_shape}")
            return graph

        except Exception as e:
            print(f"CUDA graph capture failed: {e}")

            # Mark this shape as incompatible with CUDA graphs
            self.graph_cache[input_shape] = None
            return None

    def _recover_cuda_context(self):
        """Recover from CUDA context corruption"""
        try:
            print("Recovering CUDA context...")

            # Force synchronization
            torch.cuda.synchronize()

            # Create new streams
            self.torch_stream = torch.cuda.Stream()
            self.cuda_stream_ptr = self.torch_stream.cuda_stream
            self.warmup_stream = torch.cuda.Stream()
            self.warmup_stream_ptr = self.warmup_stream.cuda_stream

            # Update profiler to use new stream
            self.profiler.set_stream(self.torch_stream)

            # Clear cache
            torch.cuda.empty_cache()

            print("CUDA context recovery completed")

        except Exception as e:
            print(f"CUDA context recovery failed: {e}")

    def _execute_with_graph(self, input_shape: tuple):
        """Execute inference using CUDA graph"""
        shape_key = input_shape

        # Get or create graph for this shape
        if shape_key not in self.graph_cache:
            self._capture_cuda_graph(input_shape)

        # Check if graph capture failed for this shape
        graph = self.graph_cache[shape_key]
        if graph is None:
            # Fall back to standard execution
            self._execute_standard()
        else:
            # Execute the cached graph
            graph.replay()

    def _execute_standard(self):
        """Execute inference using standard TensorRT execution"""
        success = self.context.execute_async_v3(self.cuda_stream_ptr)
        if not success:
            raise RuntimeError("TensorRT execution failed")

    # def preprocess(self, input_image: torch.Tensor) -> tuple[torch.Tensor, dict]:
    #     raise NotImplementedError("Subclasses must implement this method")

    def copy_input_data(self, input_image: torch.Tensor):
        """Copy input data to persistent tensor"""
        if len(self.input_names) != 1:
            raise RuntimeError(
                "Default implementation only supports models with a single input, please subclass and implement this method"
            )

        input_image = input_image.contiguous()
        input_shape = tuple(input_image.shape)

        # Ensure input tensor has the right shape
        current_shape = tuple(self.persistent_tensors[self.image_input_name].shape)
        if input_shape != current_shape:
            # If using CUDA graphs, we need to manage shape changes carefully
            if self.use_cuda_graph:
                # This will trigger graph re-capture if needed
                self.current_input_shape = input_shape
            else:
                # For standard execution, just reallocate
                self._reallocate_tensor_for_shape(self.image_input_name, input_shape)

        # Copy data to persistent tensor
        self.persistent_tensors[self.image_input_name].copy_(input_image)

        return input_shape

    def get_outputs(self) -> dict[str, torch.Tensor]:
        """Get output tensors (cloned to avoid data races)"""
        return {
            name: self.persistent_tensors[name].clone() for name in self.output_names
        }

    def infer(
        self, input_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # input_image, metadata = self.preprocess(input_image)

        torch.cuda.synchronize()

        with torch.cuda.stream(self.torch_stream):
            # Copy input data to persistent tensor
            input_shape = self.copy_input_data(input_image)

            if self.use_cuda_graph and self.cuda_graph_compatible:
                # Use async profiling for CUDA graphs to avoid interference
                with self.profiler.profile_async(stream=self.torch_stream):
                    try:
                        self._execute_with_graph(input_shape)
                    except Exception as e:
                        print(f"CUDA graph execution failed: {e}")
                        print("Disabling CUDA graphs for this model")
                        self.cuda_graph_compatible = False
                        # Perform recovery and retry with standard execution
                        self._recover_cuda_context()
                        self._execute_standard()

                # Get async timing result after synchronization
                self.torch_stream.synchronize()
                async_timing = self.profiler.get_last_timing_async()
                if async_timing is None:
                    print("Warning: Could not retrieve async timing measurement")

            else:
                # Use regular profiling for standard execution
                with self.profiler.profile(stream=self.torch_stream):
                    self._execute_standard()

        torch.cuda.synchronize()

        # Get outputs
        outputs = self.get_outputs()

        return outputs

    def print_latency_stats(self):
        try:
            self.profiler.print_stats()
        except Exception as e:
            print(f"Could not print profiler stats due to CUDA error: {e}")
            print("Profiling may have been disabled due to CUDA context issues")

    def cleanup(self):
        """Clean up CUDA graph resources"""
        # PyTorch CUDA graphs are automatically cleaned up by Python's garbage collector
        # Just clear our cache
        self.graph_cache.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("CUDA graph resources cleaned up")

        # Note: PyTorch streams (self.torch_stream, self.warmup_stream) are automatically
        # cleaned up when they go out of scope

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup


def preprocess_image(image: torch.Tensor) -> tuple[torch.Tensor, dict]:
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    means = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    stds = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)

    image = TF.normalize(image, means, stds)
    # image = TF.resize(image, image_input_shape[2:])
    return image


def process_img(frame, imgsz):
    # image = Image.open("image.jpg").convert("RGB")
    # image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = frame.resize((imgsz, imgsz))  # Resize to model's input resolution
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


def cxcywh_to_xyxy(boxes):
    boxes = boxes.clone()
    boxes[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    boxes[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
    boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
    return boxes


def postprocess_output(outputs, origin_height, origin_width, confidence_threshold=0.3):
    bboxes = outputs["dets"]
    out_logits = outputs["labels"]
    scores = out_logits.sigmoid()

    flat_scores = scores.view(scores.shape[0], -1)
    num_select = min(300, flat_scores.shape[1])

    topk_values, topk_indexes = torch.topk(flat_scores, num_select, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    bboxes = torch.gather(bboxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    bboxes = cxcywh_to_xyxy(bboxes)
    bboxes[..., [0, 2]] *= origin_width
    bboxes[..., [1, 3]] *= origin_height
    confidence_mask = scores > confidence_threshold
    scores = scores[confidence_mask]
    labels = labels[confidence_mask]
    bboxes = bboxes[confidence_mask]

    return bboxes.contiguous(), labels.contiguous(), scores.contiguous()


model = "inference_model.engine"
path = "/mnt/ssd1t/Users/thuongnh/rf-detr/test_sat_personhand/images"
CLASSES = ["person", "hand"]
text_color = (0, 0, 0)  # black text
box_color = (255, 0, 0)
bg_color = (255, 255, 255)
m = TRTInference(model)
output_dir = "outputs/output_trt"
os.makedirs(output_dir, exist_ok=True)
# font = ImageFont.truetype("arial.ttf", 34)
i = 0
t = 0
font = ImageFont.load_default()
for img_path in os.listdir(path):
    fout = img_path.split(".")[0] + ".txt"
    f = open(os.path.join(output_dir, fout), "w")
    # result_rgb = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    frame_rgb = Image.open(os.path.join(path, img_path)).convert("RGB")
    result_rgb = frame_rgb.copy()
    orig_w, orig_h = frame_rgb.size
    img = process_img(frame_rgb, 576)

    # print(type(img))
    t1 = time.time()
    outputs = m.infer(torch.Tensor(img))
    t2 = time.time()
    t_inf = t2 - t1
    t += t_inf
    i += 1
    print("FPS here:", 1 / (t_inf))
    # print(boxes, labels)
    boxes, labels, scores = postprocess_output(outputs, orig_h, orig_w, 0.3)

    draw = ImageDraw.Draw(result_rgb)
    # font = ImageFont.load_default()

    # print(scores, labels, boxes)
    # Loop over boxes and draw
    for i, box in enumerate(boxes):
        label = labels[i]
        # print(label)
        classes = CLASSES[label]
        conf = scores[i]
        # Use same color as mask but fully opaque for the outline
        # box_color = tuple(label_colors[label][:3])  # ignore alpha
        draw.rectangle(list(box), outline=(255 * int(label), 255, 0), width=4)
        lab = f"{classes} {round(conf.item(), 2)}"
        # Measure text
        text_bbox = draw.textbbox((0, 0), lab, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        pad = 6
        # Label position (top-left of bbox)
        x0 = box[0]
        y0 = max(0, box[1] - th - pad * 2)
        # If label goes above image, move it inside
        if y0 < 0:
            y0 = box[1]
        # Draw label text
        draw.rectangle([x0, y0, x0 + tw + pad * 2, y0 + th + pad * 2], fill=bg_color)
        draw.text(
            (x0 + pad, y0 + pad),
            text=lab,
            fill=text_color,
            font=font,
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )
        f.write(
            f"{int(label)} {round(conf.item(),2)} {round(box[0].item())} {round(box[1].item())} {round(box[2].item())} {round(box[3].item())}"
            + "\n"
        )
    # result_rgb.save(os.path.join(output_dir,img_path))
print(i)
print(t)
print("FPS average:", i / t)
#     frame_out = cv2.cvtColor(np.array(result_rgb), cv2.COLOR_RGB2BGR)
#     out.write(frame_out)
#     frame_count += 1

#     if frame_count % 10 == 0:
#         print(f"Processed {frame_count} frames...")

# video_capture.release()
# out.release()
# print("Video processing complete. Result saved as 'results_video.mp4'.")
