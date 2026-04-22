#!/bin/bash
# Evaluation sweep for Qwen2.5-VL-7B-Instruct (layers 0-27)
# Run: bash run_layer_eval.sh 2>&1 | tee layer_eval.log

for layer in $(seq 0 27); do
    echo "========================================="
    echo "Evaluating layer ${layer} / 27"
    echo "========================================="
    CUDA_VISIBLE_DEVICES=2 python -m probe.evaluate \
        --config configs/eval_config_linear_qwen_layer_${layer}.yaml
    echo "Layer ${layer} eval done."
done

echo "All layers evaluated."
