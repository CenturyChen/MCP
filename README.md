# MCP: Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models

> **Official Implementation of "MCP: Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models" (ICCV 2025)**  
> 🔗 Read our paper: [arxiv](https://arxiv.org/pdf/2508.01225) 

---

## Overview
🚀 This repository contains the official implementation of **MCP (Multi-Cache Enhanced Prototype Learning)**, proposed in our ICCV 2025 paper.  
MCP introduces a multi-cache mechanism that dynamically maintains prototype representations during test-time adaptation (TTA) of vision-language models (VLMs), enabling robust generalization under distribution shifts.

<p align="center">
  <img src="assets/framework.png" width="700">
</p>
---

## Installation

```bash
git clone https://github.com/yourusername/MCP.git
cd MCP
conda create -n mcp python=3.10
conda activate mcp
pip install -r requirements.txt
```

## Dataset

📦 To set up all required datasets, kindly refer to the guidance in DATASETS.md, which incorporates steps for two benchmarks.

## Run MCP
### OOD Benchmark
- ResNet50: Run MCP on the OOD Benchmark using the ResNet50 model:
```bash
bash ./scripts/run_ood_benchmark_rn50.sh 
```
- ViT-B/16: Run MCP on the OOD Benchmark using the ViT-B/16 model:
```bash
bash ./scripts/run_ood_benchmark_vit.sh 
```
### Cross-Domain Benchmark
- ResNet50: Run MCP on the Cross-Domain Benchmark using the ResNet50 model:
```bash
bash ./scripts/run_ood_benchmark_rn50.sh 
```
- ViT-B/16: Run MCP on the Cross-Domain Benchmark using the ViT-B/16 model:
```bash
bash ./scripts/run_ood_benchmark_vit.sh 
```

## Citation

🤗 If you find our work useful, please consider citing:

```bibtex
@inproceedings{chen2025mcp,
  title={Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models},
  author={Chen, Xinyu and Zhai, Haotian and Zhang, Can and Shi, Xiupeng and Li, Ruirui},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```
## Acknowledgements

This research is inspired by previous works including [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter), [TPT](https://github.com/azshue/TPT), [CuPL](https://github.com/sarahpratt/CuPL), [TDA](https://github.com/kdiAAA/TDA), and [DPE-CLIP](https://github.com/zhangce01/DPE-CLIP). We appreciate their valuable open-source efforts.
