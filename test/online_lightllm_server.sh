#!/bin/bash

gpu_type = $1
tp_size = 2
if [ $gpu_type = "nv" ]; then
    echo "Using Nvidia"
    tp_size = 4
elif [ $gpu_type = "amd" ]; then
    echo "Using AMD"
fi

python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-7b-chat-hf --swap-space 16 --disable-log-requests -tp $tp_size --port 12346 --host 0.0.0.0