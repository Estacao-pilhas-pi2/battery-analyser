#!/bin/bash

IMAGE_FILE="tests/imagem.jpg"

while true; do
    sleep 0.5s;

    ffmpeg -y -f video4linux2 -i /dev/video0 -vframes 1 \
         -video_size 640x480 $IMAGE_FILE 2>/dev/null > /dev/null
    if [[ "$?" != "0" ]]; then
        echo "Falha ao capturar webcam";
        exit 1;
    fi;

    echo $IMAGE_FILE;
done | poetry run python test.py
