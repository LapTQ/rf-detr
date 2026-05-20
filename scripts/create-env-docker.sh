docker run \
    -it \
    -d \
    --gpus all \
    --runtime nvidia \
    --network=host \
    --privileged \
    -v /mnt:/mnt \
    -v /home:/home \
    -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu \
    -v /usr/lib/firmware/nvpva_020.fw:/usr/lib/firmware/nvpva_020.fw \
    -v /usr/lib/firmware/nvpva_010.fw:/usr/lib/firmware/nvpva_010.fw \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    --shm-size=2G \
    --workdir $(dirname $(pwd)) \
    --name laptq-tensorrt \
    nvcr.io/nvidia/l4t-tensorrt:r10.3.0-devel

docker exec laptq-tensorrt pip install rf-detr/externals/torch-2.7.0-cp310-cp310-linux_aarch64.whl
docker exec laptq-tensorrt pip install rf-detr/externals/torchvision-0.22.0-cp310-cp310-linux_aarch64.whl

docker exec laptq-tensorrt pip install "numpy<2"
docker exec laptq-tensorrt pip install opencv-python pillow