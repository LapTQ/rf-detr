docker run \
    -it \
    -d \
    --gpus all \
    --runtime nvidia \
    --network=host \
    --privileged \
    -v /mnt:/mnt \
    -v /home:/home \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    --workdir $( pwd ) \
    --name laptq-tensorrt \
    nvcr.io/nvidia/l4t-tensorrt:r10.3.0-devel

docker exec laptq-tensorrt pip install externals/torch-2.7.0-cp310-cp310-linux_aarch64.whl
docker exec laptq-tensorrt pip install externals/torchvision-0.22.0-cp310-cp310-linux_aarch64.whl

docker exec laptq-tensorrt pip install "numpy<2"
docker exec laptq-tensorrt pip install opencv-python pillow