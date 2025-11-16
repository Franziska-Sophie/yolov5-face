#!/usr/bin/fish

uv run test.py \
    --task "test" \
    --device 0 \
    --verbose \
    --save-txt \
    --save-conf \
    --frame-skip 34 \
    --img-size 960 \
    --workers 8 \
    --batch-size 48 \
    --single-cls \
    --data ../cabin-pre-annotations/annotations/yolo_dataset_synth_noAnimations.yaml \
    --weights runs/train/experiment_11/weights/best.pt \
    --name "experiment_11_test_synth_noAnimations"

uv run test.py \
    --task "test" \
    --device 0 \
    --verbose \
    --save-txt \
    --save-conf \
    --frame-skip 34 \
    --img-size 960 \
    --workers 8 \
    --batch-size 48 \
    --single-cls \
    --data ../cabin-pre-annotations/annotations/yolo_dataset_synth_noOutdoorLighting.yaml \
    --weights runs/train/experiment_12/weights/best.pt \
    --name "experiment_12_test_synth_noOutdoorLighting"
