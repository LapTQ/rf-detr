export PATH=/usr/src/tensorrt/bin/:$PATH
trtexec --onnx=inference_model.onnx --saveEngine=inference_model.engine 