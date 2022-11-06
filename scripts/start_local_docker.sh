docker run \
    --gpus all \
    --shm-size 32g \
    --ipc=host \
    --name coco-funit \
    -it \
    -v /mnt:/mnt \
    -v ~/:/home \
    -p 8083:6006 \
    nvcr.io/nvidian/lpr-imagine/imaginaire:${1}-py3 \
    /bin/bash

