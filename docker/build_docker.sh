docker build \
    --file Dockerfile ./ \
    --tag $USER/pytorch:1.13.1-cuda11.6-cudnn8-devel
