#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download

os.environ.setdefault("SSD_CUDA_ARCH", "12.0")

from securesd.engine.llm_engine import LLMEngine
from securesd.sampling_params import SamplingParams


HF_ALLOW_PATTERNS = [
    "*.json",
    "*.model",
    "*.safetensors",
    "tokenizer*",
    "special_tokens_map.json",
]


class ContextLengthError(ValueError):
    pass


def _snapshot(model_id: str, cache_dir: str) -> str:
    return snapshot_download(
        model_id,
        cache_dir=cache_dir,
        allow_patterns=HF_ALLOW_PATTERNS,
    )


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for block in content:
            if isinstance(block, str):
                pieces.append(block)
            elif isinstance(block, dict):
                value = block.get("text", block.get("content", ""))
                pieces.append(str(value))
            else:
                pieces.append(str(block))
        return "\n".join(pieces)
    return str(content)


def _messages_for_qwen(messages: list[dict]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        if role == "developer":
            role = "system"
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        normalized.append({"role": role, "content": _content_text(message.get("content"))})
    if not normalized:
        normalized.append({"role": "user", "content": ""})
    return normalized


class SecureSDService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.condition = args.condition
        self.model_id = args.served_model_name or args.condition
        self.stats_path = Path(args.stats_path)
        self.stats_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.records: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []

        target_path = args.target_path or _snapshot(args.target_model, args.hf_cache)
        draft_path = args.draft_path or _snapshot(args.draft_model, args.hf_cache)
        engine_kwargs = dict(
            model=target_path,
            draft=draft_path,
            method=args.method,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_model_len,
            max_num_seqs=1,
            kvcache_block_size=args.kvcache_block_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            draft_gpu_memory_utilization=args.draft_gpu_memory_utilization,
            speculate_k=args.speculate_k,
            lossy_epsilon=args.lossy_epsilon,
            bild_fallback_threshold=args.bild_fallback_threshold,
            bild_rollback_threshold=args.bild_rollback_threshold,
            fsd_threshold=args.fsd_threshold,
            fsd_div_type=args.fsd_div_type,
            mars_theta=args.mars_theta,
            sc_alpha=args.sc_alpha,
            sc_rule=args.sc_rule,
            fly_entropy_threshold=args.fly_entropy_threshold,
            fly_window_size=args.fly_window_size,
            eta_schedule="step",
            eta_start=0.0,
            eta_end=1.0,
            eta_len=args.eta_len,
        )
        torch.manual_seed(args.seed)
        self.engine = LLMEngine(**engine_kwargs)
        self.tokenizer = self.engine.tokenizer
        self.meta = {
            "condition": self.condition,
            "served_model_name": self.model_id,
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "target_path": target_path,
            "draft_path": draft_path,
            "method": args.method,
            "eta_len": args.eta_len,
            "secure_sd": bool(args.eta_len >= 2 and args.method not in {"ar_target", "ar_draft"}),
            "speculate_k": args.speculate_k,
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "draft_gpu_memory_utilization": args.draft_gpu_memory_utilization,
            "seed": args.seed,
            "physical_gpu": args.physical_gpu,
            "operating_point": {
                "lossy_epsilon": args.lossy_epsilon,
                "bild_fallback_threshold": args.bild_fallback_threshold,
                "bild_rollback_threshold": args.bild_rollback_threshold,
                "fsd_threshold": args.fsd_threshold,
                "fsd_div_type": args.fsd_div_type,
                "mars_theta": args.mars_theta,
                "sc_alpha": args.sc_alpha,
                "sc_rule": args.sc_rule,
                "fly_entropy_threshold": args.fly_entropy_threshold,
                "fly_window_size": args.fly_window_size,
            },
        }
        self._warmup()
        self._flush()

    def _render(self, messages: list[dict]) -> list[int]:
        normalized = _messages_for_qwen(messages)
        try:
            ids = self.tokenizer.apply_chat_template(
                normalized,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            ids = self.tokenizer.apply_chat_template(
                normalized,
                tokenize=True,
                add_generation_prompt=True,
            )
        return [int(x) for x in ids]

    def _generate(self, messages: list[dict], max_new_tokens: int, temperature: float) -> tuple[str, dict]:
        prompt_ids = self._render(messages)
        lookahead = self.args.speculate_k + 1 if self.engine.config.speculate else 1
        if len(prompt_ids) + max_new_tokens + lookahead > self.args.max_model_len:
            raise ContextLengthError(
                f"context length {len(prompt_ids)} + completion {max_new_tokens} "
                f"exceeds max_model_len={self.args.max_model_len}"
            )
        params = [SamplingParams(
            temperature=temperature,
            draft_temperature=temperature,
            max_new_tokens=max_new_tokens,
            ignore_eos=False,
        )]
        started = time.perf_counter()
        outputs, raw = self.engine.generate([prompt_ids], params, use_tqdm=False)
        wall = time.perf_counter() - started
        token_ids = outputs[0]["token_ids"]
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        decode_time = float(raw.get("decode_total_time", 0.0))
        decode_tokens = int(raw.get("decode_total_tokens", 0))
        record = {
            "request_index": len(self.records),
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(token_ids),
            "decode_tokens": decode_tokens,
            "prefill_time_s": float(raw.get("prefill_total_time", 0.0)),
            "decode_time_s": decode_time,
            "wall_time_s": wall,
            "decode_tps": decode_tokens / decode_time if decode_time > 0 else 0.0,
        }
        return text, record

    def _warmup(self) -> None:
        self._generate(
            [{"role": "user", "content": "Reply with OK."}],
            min(8, self.args.max_new_tokens),
            0.0,
        )

    def completion(self, payload: dict) -> dict:
        requested = int(payload.get("max_tokens") or self.args.max_new_tokens)
        max_new_tokens = max(1, min(requested, self.args.max_new_tokens))
        temperature = float(payload.get("temperature") or 0.0)
        with self.lock:
            try:
                text, record = self._generate(
                    payload.get("messages") or [], max_new_tokens, temperature
                )
            except Exception as exc:
                self.errors.append({
                    "request_index": len(self.records),
                    "type": type(exc).__name__,
                    "message": str(exc),
                })
                self._flush()
                raise
            self.records.append(record)
            self._flush()
        finish_reason = "length" if record["completion_tokens"] >= max_new_tokens else "stop"
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }],
            "usage": {
                "prompt_tokens": record["prompt_tokens"],
                "completion_tokens": record["completion_tokens"],
                "total_tokens": record["prompt_tokens"] + record["completion_tokens"],
            },
        }

    def summary(self) -> dict:
        decode_tokens = sum(int(x["decode_tokens"]) for x in self.records)
        decode_time = sum(float(x["decode_time_s"]) for x in self.records)
        return {
            "meta": self.meta,
            "summary": {
                "num_requests": len(self.records),
                "num_errors": len(self.errors),
                "prompt_tokens": sum(int(x["prompt_tokens"]) for x in self.records),
                "completion_tokens": sum(int(x["completion_tokens"]) for x in self.records),
                "decode_tokens": decode_tokens,
                "decode_time_s": decode_time,
                "decode_tps": decode_tokens / decode_time if decode_time > 0 else 0.0,
                "prefill_time_s": sum(float(x["prefill_time_s"]) for x in self.records),
                "wall_time_s": sum(float(x["wall_time_s"]) for x in self.records),
            },
            "requests": self.records,
            "errors": self.errors,
        }

    def _flush(self) -> None:
        self.stats_path.write_text(json.dumps(self.summary(), indent=2) + "\n")

    def close(self) -> None:
        self._flush()
        self.engine.exit(hard=False)


def _handler(service: SecureSDService):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SecureSDOpenAI/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send(200, {"status": "ok", "condition": service.condition})
            elif self.path == "/v1/models":
                self._send(200, {
                    "object": "list",
                    "data": [{"id": service.model_id, "object": "model", "owned_by": "securesd"}],
                })
            elif self.path == "/stats":
                self._send(200, service.summary())
            else:
                self._send(404, {"error": {"message": "not found"}})

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send(404, {"error": {"message": "not found"}})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                self._send(200, service.completion(payload))
            except Exception as exc:
                error = {"message": str(exc), "type": type(exc).__name__}
                if isinstance(exc, ContextLengthError):
                    error["code"] = "context_length_exceeded"
                self._send(400, {"error": error})

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--eta-len", type=float, default=1.0)
    parser.add_argument("--target-model", default="Qwen/Qwen3-8B")
    parser.add_argument("--draft-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--target-path")
    parser.add_argument("--draft-path")
    parser.add_argument("--hf-cache", default=str(Path.home() / ".cache/huggingface/hub"))
    parser.add_argument("--served-model-name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--kvcache-block-size", type=int, default=256)
    parser.add_argument("--speculate-k", type=int, default=5)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--draft-gpu-memory-utilization", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--physical-gpu", type=int)

    parser.add_argument("--lossy-epsilon", type=float, default=0.3)
    parser.add_argument("--bild-fallback-threshold", type=float, default=0.28)
    parser.add_argument("--bild-rollback-threshold", type=float, default=2.0)
    parser.add_argument("--fsd-threshold", type=float, default=0.26)
    parser.add_argument("--fsd-div-type", default="js_div")
    parser.add_argument("--mars-theta", type=float, default=0.945)
    parser.add_argument("--sc-alpha", type=float, default=0.36)
    parser.add_argument("--sc-rule", default="chow")
    parser.add_argument("--fly-entropy-threshold", type=float, default=0.1)
    parser.add_argument("--fly-window-size", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = SecureSDService(args)
    server = ThreadingHTTPServer((args.host, args.port), _handler(service))

    def _stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    print(f"READY http://{args.host}:{args.port} condition={args.condition}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
