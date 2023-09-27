#!/bin/bash

mod=$1
if [ "$mod" == "server" ]; then
    python -m lightllm.server.api_server --model_dir /home/aiscuser/models/Llama-2-7b-chat-hf --tp 2  --tokenizer_mode auto --host 127.0.0.1 --port 12345
elif [ "$mod" == "client" ]; then
    python benchmark_serving.py --tokenizer /home/aiscuser/models/Llama-2-7b-chat-hf --dataset ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 2000 --request-rate 200
elif [ "$mod" == "data" ]; then
    wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
fi