#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python mcp_runner.py     --config configs \
                                                --datasets I/A/V/R/S \
                                                --backbone RN50 --res False
                                                
                                              