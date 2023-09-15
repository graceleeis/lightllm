#!/bin/bash

python benchmark_throughput.py --backend vllm --dataset ./ShareGPT_V3_unfiltered_cleaned_split.json --model meta-llama/Llama-2-7b-chat-hf -tp 16