# SPDX-License-Identifier: Apache-2.0
"""Text-first split pipeline for Qwen3-Omni."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time

from sglang_omni.config import build_pipeline_runner
from sglang_omni.models.qwen3_omni.config import Qwen3OmniPipelineConfig
from sglang_omni.proto import CompleteMessage, OmniRequest, StreamMessage

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "DEBUG").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Hugging Face model id",
    )
    parser.add_argument("--prompt", type=str, default="Describe this input.")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--thinker-max-seq-len", type=int, default=8192)
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallel size for the SGLang-backed Qwen3 thinker",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        help="Optional SGLang quantization mode for the thinker",
    )
    parser.add_argument(
        "--cpu-offload-gb",
        type=float,
        default=None,
        help="CPU offload size in GB for the thinker model",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help="Static GPU memory fraction reserved by the thinker backend",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--use-audio-in-video", action="store_true")
    parser.add_argument("--audio-path", type=str, default=None)
    parser.add_argument("--audio-target-sr", type=int, default=16000)
    parser.add_argument(
        "--measure-ttft",
        action="store_true",
        help="Measure time-to-first-token via coordinator streaming",
    )
    parser.add_argument(
        "--relay-backend", type=str, default="nixl", choices=["nixl", "shm"]
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    overrides = {}
    if args.tp_size and args.tp_size > 1:
        overrides["tp_size"] = args.tp_size
    if args.quantization:
        overrides["quantization"] = args.quantization
    if args.cpu_offload_gb:
        overrides["cpu_offload_gb"] = args.cpu_offload_gb
    if args.mem_fraction_static is not None:
        overrides["mem_fraction_static"] = args.mem_fraction_static

    config = Qwen3OmniPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
        server_args_overrides=overrides if overrides else None,
    )
    runner = build_pipeline_runner(config)

    await runner.start()
    try:
        images = [args.image_path] if args.image_path else []
        videos = [args.video_path] if args.video_path else []
        audios = [args.audio_path] if args.audio_path else []
        request = {
            "messages": [
                {"role": "user", "content": args.prompt},
            ],
            "images": images,
            "videos": videos,
            "video_fps": args.video_fps,
            "use_audio_in_video": args.use_audio_in_video,
            "audios": audios,
            "audio_target_sr": args.audio_target_sr,
        }
        params = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        }
        request_id = "qwen3-omni-text-first"
        result = None

        if args.measure_ttft:
            params["stream"] = True
            t0 = time.perf_counter()
            ttft_s = None
            streamed_token_count = 0

            async for msg in runner.coordinator.stream(
                request_id,
                OmniRequest(inputs=request, params=params),
            ):
                if isinstance(msg, StreamMessage):
                    if ttft_s is None:
                        ttft_s = time.perf_counter() - t0
                        print(f"TTFT: {ttft_s:.3f}s")

                    chunk = msg.chunk if isinstance(msg.chunk, dict) else {}
                    token_ids = chunk.get("token_ids")
                    if isinstance(token_ids, list):
                        streamed_token_count += len(token_ids)
                    elif chunk.get("token_id") is not None:
                        streamed_token_count += 1
                elif isinstance(msg, CompleteMessage):
                    result = msg.result

            total_s = time.perf_counter() - t0
            if ttft_s is not None:
                decode_s = max(total_s - ttft_s, 1e-9)
                print(f"Total latency: {total_s:.3f}s")
                print(f"Streamed tokens: {streamed_token_count}")
                print(f"Decode throughput: {streamed_token_count / decode_s:.3f} tok/s")
            else:
                print(f"No stream chunk received before completion. Total latency: {total_s:.3f}s")
        else:
            result = await runner.coordinator.submit(
                request_id,
                OmniRequest(inputs=request, params=params),
            )
        print(result)
    finally:
        await runner.stop()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
