#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python mcp_runner.py     --config configs \
                                                --datasets eurosat/aircraft/dtd/caltech101/oxford_flowers/oxford_pets/stanford_cars/sun397/ucf101/food101 \
                                                --backbone ViT-B/16 --res False
