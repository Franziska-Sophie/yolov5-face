#!/usr/bin/fish

set experiment_id "03"
set dataset_file "yolo_dataset_synth.yaml"
set frame_skip 90

uv run train.py \
    --device 0,1,2,3 \
    --cfg models/yolov5m.yaml \
    --weights weights/yolov5m-face.pt \
    --hyp hyp.face-detection.yaml \
    --frame-skip $frame_skip \
    --img-size 960 \
    --epochs 200 \
    --workers 8 \
    --batch 96 \
    --single-cls \
    --data ../cabin-pre-annotations/annotations/"$dataset_file" \
    --name "experiment_$experiment_id"

uv run test.py \
    --task "test" \
    --device 0 \
    --verbose \
    --save-txt \
    --save-conf \
    --frame-skip 10 \
    --img-size 960 \
    --workers 8 \
    --batch-size 48 \
    --single-cls \
    --data ../cabin-pre-annotations/annotations/yolo_dataset.yaml \
    --weights runs/train/experiment_"$experiment_id"/weights/best.pt \
    --name "experiment_"$experiment_id"_test_real"

uv run test.py \
    --task "test" \
    --device 0 \
    --verbose \
    --save-txt \
    --save-conf \
    --frame-skip 17 \
    --img-size 960 \
    --workers 8 \
    --batch-size 48 \
    --single-cls \
    --data ../cabin-pre-annotations/annotations/yolo_dataset_synth.yaml \
    --weights runs/train/experiment_"$experiment_id"/weights/best.pt \
    --name "experiment_"$experiment_id"_test_synth"