#!/bin/bash

python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-7b-chat-hf --swap-space 16 --disable-log-requests -tp 2 --port 12346 --host 0.0.0.0
