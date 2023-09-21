"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Tuple
import asyncio
import sys
import numpy as np

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from tqdm import tqdm
import uuid

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.sampling_params import SamplingParams as LightLLMSamplingParams
from lightllm.server.router.manager import start_router_process
from lightllm.server.detokenization.manager import start_detokenization_process
from lightllm.utils.net_utils import alloc_can_use_network_port
import multiprocessing as mp

# python xr.py --backend lightllm --model_dir /home/aiscuser/models/Llama-2-7b-chat-hf -
# -dataset ShareGPT_V3_unfiltered_cleaned_split.json
isFirst = True

def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    print("read data set finish")
    # Tokenize the prompts and completions.
    import random
    dataset = random.sample(dataset, num_requests * 3)
    prompts = [prompt for prompt, _ in dataset]
    completions = [completion for _, completion in dataset]

    prompt_token_ids = tokenizer(prompts).input_ids
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    sum_len = 0
    for e in sampled_requests:
        sum_len += e[1] + e[2]
    print("total tokens:", sum_len)
    return sampled_requests

async def run_lightllm(
    requests: List[Tuple[str, int, int]],
) -> float:

    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    start = time.time()
    final_output = []
    for prompt, _, output_len in requests:
        sampling_params = LightLLMSamplingParams(
            do_sample = False,
            ignore_eos=True,
            max_new_tokens=output_len,
        )
        sampling_params.verify()
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        results_generator = [item async for item in httpserver_manager.generate(prompt, sampling_params, request_id)]
        for request_output, metadata, _  in results_generator:
            # print(f"Received result: {request_output}")
            final_output.append(request_output)

    assert final_output is not None
    # ret = {"generated_text": ["".join(final_output)]}

    end = time.time()
    return end - start


def run_vllm(
    requests: List[Tuple[str, int, int]],
    model: str,
    tokenizer: str,
    tensor_parallel_size: int,
    seed: int,
    n: int,
    use_beam_search: bool,
    trust_remote_code: bool,
) -> float:
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        tensor_parallel_size=tensor_parallel_size,
        seed=seed,
        trust_remote_code=trust_remote_code,
    )

    # Add the requests to the engine.
    for prompt, _, output_len in requests:
        sampling_params = SamplingParams(
            n=n,
            temperature=0.0 if use_beam_search else 1.0,
            top_p=1.0,
            use_beam_search=use_beam_search,
            ignore_eos=True,
            max_tokens=output_len,
        )
        # FIXME(woosuk): Do not use internal method.
        llm._add_request(
            prompt=prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    start = time.time()
    # FIXME(woosuk): Do use internal method.
    llm._run_engine(use_tqdm=True)
    end = time.time()
    return end - start


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Sample the requests.
    tokenizer = get_tokenizer(args.tokenizer, tokenizer_mode="slow")
    requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    if args.backend == "vllm":
        elapsed_time = run_vllm(
            requests, args.model, args.tokenizer, args.tensor_parallel_size,
            args.seed, args.n, args.use_beam_search, args.trust_remote_code)
    elif args.backend == "lightllm":
        global httpserver_manager
        args = parser.parse_args()
        args.nccl_port = 28765
        args.tokenizer_mode = "slow"
        args.batch_max_tokens = args.max_req_total_len
        args.running_max_req_size = 1000
        args.eos_id = 2
        args.disable_log_stats = True
        args.log_stats_interval = 10
        args.trust_remote_code = False
        can_use_ports = alloc_can_use_network_port(
            num=3 + args.tp, used_nccl_port=args.nccl_port
        )
        router_port, detokenization_port, httpserver_port = can_use_ports[0:3]
        model_rpc_ports = can_use_ports[3:]

        httpserver_manager = HttpServerManager(args.model_dir,
                                           "auto",
                                           router_port=router_port,
                                           httpserver_port=httpserver_port,
                                           total_token_num=args.max_total_token_num,
                                           max_req_input_len=args.max_req_input_len,
                                           max_req_total_len=args.max_req_total_len,
                                           trust_remote_code=True)
        pipe_router_reader, pipe_router_writer = mp.Pipe(duplex=False)
        pipe_detoken_reader, pipe_detoken_writer = mp.Pipe(duplex=False)
        proc_router = mp.Process(target=start_router_process, args=(
            args, router_port, detokenization_port, model_rpc_ports, [], pipe_router_writer))
        proc_router.start()
        proc_detoken = mp.Process(target=start_detokenization_process, args=(
            args, detokenization_port, httpserver_port, pipe_detoken_writer, args.trust_remote_code))
        proc_detoken.start()

        # wait load model ready
        router_init_state = pipe_router_reader.recv()
        detoken_init_state = pipe_detoken_reader.recv()

        if router_init_state != "init ok" or detoken_init_state != "init ok":
            proc_router.kill()
            proc_detoken.kill()
            print("router init state:", router_init_state, "detoken init state:", detoken_init_state)
            sys.exit(1)

        assert proc_router.is_alive() and proc_detoken.is_alive()

        elapsed_time = asyncio.run(run_lightllm(requests))
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(
        prompt_len + output_len
        for _, prompt_len, output_len in requests
    )
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")
    sys.exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend", type=str, choices=["vllm","lightllm"],
                        default="vllm")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    # parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--tp", "-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--n", type=int, default=1,
                        help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-prompts", type=int, default=1000,
                        help="Number of prompts to process.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hf-max-batch-size", type=int, default=None,
                        help="Maximum batch size for HF backend.")
    parser.add_argument('--trust-remote-code',
                        action='store_true',
                        help='trust remote code from huggingface')
    parser.add_argument("--max_total_token_num", type=int, default=120000)
    parser.add_argument("--max_req_input_len", type=int, default=2048)
    parser.add_argument("--max_req_total_len", type=int, default=2048 + 1024)
    args = parser.parse_args()

    if args.backend == "vllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    elif args.backend == "lightllm":
        if args.hf_max_batch_size is not None:
            raise ValueError("HF max batch size is only for HF backend.")
    if args.tokenizer is None:
        args.tokenizer = args.model

    main(args)