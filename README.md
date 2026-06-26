# Deep Learning for Anatomical Region Segmentation in Medical Images

<p align="left">
  <img src="https://img.shields.io/badge/Task-Multi--class%20Region%20Segmentation-1565C0?style=flat-square">
  <img src="https://img.shields.io/badge/Scale-Whole%20Section%20(serial)-7B1FA2?style=flat-square">
  <img src="https://img.shields.io/badge/Models-Mask2Former%20%7C%20Attn--UNet%20%7C%20DeepLabV3%2B-EE4C2C?style=flat-square">
  <img src="https://img.shields.io/badge/Status-Active%20R%26D%20(in%20progress)-F9A825?style=flat-square">
  <img src="https://img.shields.io/badge/Current%20best%20mIoU-0.717-2E7D32?style=flat-square">
</p>

> A deep learning system that segments anatomical regions in large medical images so that experts review and adjust instead of tracing every boundary by hand. Built on serial whole-section histology at the Sudha Gopalakrishnan Brain Centre, IIT Madras. This repository documents the method and an in-progress model-selection study; source is held under institutional IP.

> Status, work in progress. The goal here is the workflow, not a finished model. Current models already produce usable first-pass region proposals, and accuracy is actively being improved with more annotated data, better long-tail handling, and post-processing. The numbers below are current baselines on the path up, not final results.

## The goal: cut the manual annotation burden

A single specimen is reconstructed from roughly 10,000 serial sections, and every section has to be partitioned into many anatomical regions. Done by hand, this is the single most labor-intensive step in the program. It is slow, costly in expert hours, and inconsistent between annotators.

The primary objective of this project is to reduce that manual labor, time, and effort. Instead of an expert drawing every region boundary from scratch on every section, the model produces a first-pass region map the expert only has to review and adjust, converting hours of manual tracing per section into minutes of correction. Raw accuracy is secondary to that workflow gain: even an imperfect model that gets the expert most of the way there is a large net time saving across thousands of sections.

That framing, labor saved rather than leaderboard score, is the lens for everything below, and it is why this ships as a human-in-the-loop assist rather than a fully autonomous segmenter.

## System overview

![System overview](./assets/architecture.svg)

A section is preprocessed, a region model predicts a dense multi-class map, the raster map is vectorized to polygons and exported as GeoJSON, and an expert refines the proposal in an OpenSeadragon deep-zoom viewer. The reviewed regions feed the downstream 3D reconstruction pipeline.

## Model-selection study (ongoing)

Rather than commit to one architecture early, three strong segmentation families are trained on the same task and data splits and compared on mean IoU, per-class IoU, and rare-region behaviour. This is the disciplined part: a defensible, measured direction with an honest baseline to improve against.

![Three architectures benchmarked](./assets/model-benchmark.svg)

| Model | Backbone or core idea | Input | Loss | Result |
|---|---|---|---|---|
| 1, Mask2Former (memory-optimized) | Residual encoder-decoder plus lightweight transformer bottleneck | 256 | Focal plus Dice | qualitative, see Fig. 1 |
| 2, Attention U-Net (plus ASPP, curriculum) | Attention-gated U-Net plus ASPP | 512 | Weighted Dice plus Weighted CE | IoU 0.592 |
| 3, DeepLabV3+ (plus curriculum) | ResNet50V2 (ImageNet) plus ASPP | 512 | Weighted Dice plus Weighted CE | IoU 0.717, best |

Takeaway so far: transfer learning from an ImageNet-pretrained ResNet50V2 backbone (Model 3), with curriculum learning and multi-scale ASPP context, currently gives the most stable region maps, and ones already good enough to seed expert review. These are working baselines, and the next iterations target higher accuracy on rare regions.

## Model 1: Mask2Former (memory-optimized)

A transformer-augmented encoder-decoder tuned to run under tight memory.

- Architecture: `Conv 7x7(32) + MaxPool`, then stacked residual blocks (32, 64, 128) with an attention-gated decoder (gates at 32, 16, 8) and skip connections. The bottleneck is 2 residual blocks plus a lightweight transformer (depthwise conv plus 1x1 channel-mixing) for global context. Final `Upsample + Conv 1x1 + Softmax`.
- Training: 256 by 256 input, batch 2, 50 epochs, mixed precision (fp16), Adam (1e-3) with LR reduction, Focal plus Dice loss, residual learning throughout. Split 80/10/10.

![Model 1 prediction](./assets/model1-prediction.png)
<sub>Fig. 1, test image, ground truth, prediction (Region Segmentation Model 1).</sub>

## Model 2: Attention U-Net with ASPP and Curriculum

- Architecture: encoder conv-blocks 64, 128, 256, 512 with MaxPool. ASPP (rates 1, 6, 12, 18 plus global average pooling) at the bridge. Attention-gated transpose-conv decoder (gates 256, 128, 64, 32). Final `Conv 1x1 + Softmax`.
- Curriculum learning: three phases (40, 40, 50 epochs) with Adam annealed 1e-4 to 6e-5 to 1e-5, early stopping plus LR reduction.
- Class imbalance: regions tiered by frequency (Tier 1 at least 50, Tier 2 from 10 to 49, Tier 3 below 10), rare classes oversampled 5x, Weighted Dice plus Weighted Sparse CE. 512 by 512, batch 1, fp16, split 70/15/15.

![Model 2 prediction](./assets/model2-prediction.png)
<sub>Fig. 2, input, ground truth, prediction, IoU 0.592 (Region Segmentation Model 2).</sub>

## Model 3: DeepLabV3+ with ResNet50V2 and Curriculum (best)

- Architecture: ResNet50V2 (ImageNet-pretrained) backbone. Low-level features from `conv2_block3_1_relu` fused with high-level `conv4_block6_1_relu`. ASPP (1, 6, 12, 18 plus GAP). Decoder of bilinear upsample, 1x1 projection(48), concat, `Conv-BN-ReLU(256)`, Dropout(0.3), bilinear upsample, `Conv 1x1 + Softmax`.
- Transfer learning plus curriculum: encoder frozen in Phase 1, then progressively unfrozen across phases (40, 40, 50 epochs) for stable fine-tuning, Adam 1e-4 to 6e-5 to 1e-5.
- Class imbalance: regions tiered by pixel area (Tier 1 at least 5000 px, Tier 2 from 500 to 4999 px, Tier 3 below 500 px), rare classes oversampled 5x, Weighted Dice plus Weighted Sparse CE. 512 by 512, batch 8, split 70/15/15.

![Model 3 prediction](./assets/model3-prediction.png)
<sub>Fig. 3, input, ground truth, prediction, IoU 0.717 (Region Segmentation Model 3).</sub>

## Engineering rigor

- Long-tail anatomy is the real problem. Rare regions dominate the error budget, so every model uses frequency or area tiering, 5x oversampling, and class-weighted loss, and tracks per-class IoU and rare-class bias rather than hiding behind a single mean.
- Curriculum learning stabilized the harder 512 models by introducing common structure first and rare structure later.
- Reproducible protocol: fixed splits, deterministic preprocessing, and the same metric suite across all three models.
- Memory-aware design: mixed precision throughout, with Model 1 re-engineering a transformer segmenter to run within constrained GPU memory.

## Roadmap

- More annotated sections to lift accuracy, especially on rare regions.
- Better long-tail handling beyond oversampling, such as boundary-aware and region-aware losses.
- Post-processing with CRF or morphological cleanup and polygon smoothing for cleaner editable boundaries.
- Quantify the time saved by measuring expert minutes per section with and without the assist, the metric that matters most.

## Tech stack

`PyTorch / TensorFlow-Keras`, `Mask2Former`, `Attention U-Net`, `DeepLabV3+ (ResNet50V2)`, `ASPP`, `mixed precision`, `shapely / GeoJSON`, `OpenSeadragon`, `FastAPI`

## Why it matters

This is an early but disciplined attempt at the most expensive step in a large imaging program, the manual annotation bottleneck. The aim is a working human-in-the-loop assist that turns hours of expert tracing into minutes of review. The approach is what an applied team wants to see in an in-progress effort: a controlled architecture comparison, explicit handling of the long-tail that breaks naive segmenters, reproducible training, and honest baselines (current best mIoU 0.717) on a clear trajectory of improvement.

<sub>Code and trained weights are private under institutional agreements. 
