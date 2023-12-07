#!/bin/bash

for i in $(seq 60 -1 1); do
    echo "Running with checkpoint: $i.pt"
    CUDA_VISIBLE_DEVICES=0 python test.py --checkpoint /media/data/hucao/zehua/results_dsec/cross_4layer/csv_fpn_homographic_retinanet_retinanet101_$i.pt
done