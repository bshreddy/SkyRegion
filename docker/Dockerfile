FROM ubuntu:18.04

COPY . /app

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y python3 python3-pip python3-dev && \
    pip3 install -r requirements.txt

RUN python3 download_models.py

RUN mkdir -p /root/.cache/torch/checkpoints/ && \
    mv models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth /root/.cache/torch/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth && \
    mv models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth /root/.cache/torch/checkpoints/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth

EXPOSE 8001

ENTRYPOINT [ "python3" ]

CMD [ "app.py" ]