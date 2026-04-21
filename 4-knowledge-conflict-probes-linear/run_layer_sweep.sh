#!/bin/bash
# Layer sweep for Qwen2.5-VL-7B-Instruct (layers 0-27)
# Run: bash run_layer_sweep.sh 2>&1 | tee layer_sweep.log

for layer in $(seq 0 27); do
    echo "========================================="
    echo "Training layer ${layer} / 27"
    echo "========================================="
    CUDA_VISIBLE_DEVICES=2 python -m probe.train \
        --config configs/train_config_linear_qwen_layer_${layer}.yaml
    echo "Layer ${layer} done."
done

echo "All layers complete."
