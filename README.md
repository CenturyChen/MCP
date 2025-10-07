# ICCV2025-MCP

> **Official Implementation of "MCP: Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models" (ICCV 2025)**  
> [Paper Link](https://arxiv.org/pdf/2508.01225) 

---

## 📘 Abstract
This repository contains the official implementation of **MCP (Multi-Cache Enhanced Prototype Learning)**, proposed in our ICCV 2025 paper.  
MCP introduces a multi-cache mechanism that dynamically maintains prototype representations during test-time adaptation (TTA) of vision-language models (VLMs), enabling robust generalization under distribution shifts.

---

## 🧩 Method Overview
<p align="center">
  <img src="assets/framework.png" width="700">
</p>

MCP maintains multiple caches (positive, negative, alignment) to model both intra-class diversity and inter-class discrimination.  
During test-time, each incoming sample updates the cache adaptively, balancing stability and plasticity.

**Key Highlights**
- 🧠 **Multi-Cache Mechanism:** Positive, negative, and alignment caches enhance feature calibration.  
- 🔁 **Entropy-based Update Strategy:** Dynamic replacement ensures sample representativeness.  
- 🌍 **Distribution-Aware Prototype Learning:** Achieves strong OOD generalization.  
- ⚙️ **Plug-and-Play Design:** Compatible with CLIP, BLIP, and other vision-language models.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/MCP.git
cd MCP
conda create -n mcp python=3.10
conda activate mcp
pip install -r requirements.txt
