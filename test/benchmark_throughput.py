"""Benchmark offline inference throughput."""
import argparse
import json
import random
import time
from typing import List, Tuple
import asyncio

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase
from tqdm import tqdm
import uuid

from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from lightllm.server.httpserver.manager import HttpServerManager
from lightllm.server.sampling_params import SamplingParams as LightLLMSamplingParams


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

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
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
    return sampled_requests

async def run_lightllm(
    requests: List[Tuple[str, int, int]],
    model_dir: str,
    n: int,
    use_beam_search: bool,
    max_total_token_num: int,
    max_req_input_len: int,
    max_req_total_len: int,
) -> float:
    httpserver_manager = HttpServerManager(model_dir,
                                           "auto",
                                           router_port=8000,
                                           httpserver_port=12345,
                                           total_token_num=max_total_token_num,
                                           max_req_input_len=max_req_input_len,
                                           max_req_total_len=max_req_total_len,
                                           trust_remote_code=True)
    start = time.time()
    sampling_params = LightLLMSamplingParams(
            temperature=0.0 if use_beam_search else 1.0,
            top_p=1.0,
            ignore_eos=True,
            max_new_tokens=output_len,
        )
    for prompt, _, output_len in requests:      
        request_id = uuid.uuid4().hex
        results_generator = httpserver_manager.generate(prompt, sampling_params, request_id)
    
    final_output = []
    import ipdb; ipdb.set_trace() 
    async for request_output, _ in results_generator:
            _, _, finished, _ = httpserver_manager.req_id_to_out_inf[request_id]
            if finished:
                final_output.append(request_output)
    end = time.time()
    assert final_output is not None
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

    # Sample the requests.
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    requests = sample_requests(args.dataset, args.num_prompts, tokenizer)

    if args.backend == "vllm":
        elapsed_time = run_vllm(
            requests, args.model, args.tokenizer, args.tensor_parallel_size,
            args.seed, args.n, args.use_beam_search, args.trust_remote_code)
    elif args.backend == "lightllm":
        elapsed_time = asyncio.run(run_lightllm(
            requests, args.model_dir, args.n, args.use_beam_search,
            args.max_total_token_num, args.max_req_input_len, args.max_req_total_len))
    else:
        raise ValueError(f"Unknown backend: {args.backend}")
    total_num_tokens = sum(
        prompt_len + output_len
        for _, prompt_len, output_len in requests
    )
    print(f"Throughput: {len(requests) / elapsed_time:.2f} requests/s, "
          f"{total_num_tokens / elapsed_time:.2f} tokens/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the throughput.")
    parser.add_argument("--backend", type=str, choices=["vllm","lightllm"],
                        default="vllm")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--model", type=str, default="facebook/opt-125m")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
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
