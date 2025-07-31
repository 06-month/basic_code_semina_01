USER_HOME=/home/$USER
nvidia-docker run \
    --name $USER-tiny-imagenet \
    -d \
    --ipc=host \
    -it \
    -v ${USER_HOME}:${USER_HOME} \
    --privileged=true \
    --network=host \
    $USER/pytorch:1.13.1-cuda11.6-cudnn8-devel \
    bash -c 'cd "${USER_HOME}"; exec "${SHELL:-sh}"'
