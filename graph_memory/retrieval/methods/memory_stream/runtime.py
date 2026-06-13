from __future__ import annotations

import time
from typing import Any, Protocol

from graph_memory.retrieval.methods.memory_stream.contracts import (
    GenerationResult,
    ImportanceMessage,
    ImportanceSettings,
)


class ImportanceRuntime(Protocol):
    def load(self) -> dict[str, object]:
        ...

    def generate(
        self,
        messages: list[ImportanceMessage],
        settings: ImportanceSettings,
    ) -> GenerationResult:
        ...


class LocalTransformersImportanceRuntime:
    def __init__(self, settings: ImportanceSettings) -> None:
        self.settings = settings
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: Any | None = None

    def load(self) -> dict[str, object]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        started = time.perf_counter()
        tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model_path,
            trust_remote_code=self.settings.trust_remote_code,
        )
        device_map = self._device_map(torch)
        model = AutoModelForCausalLM.from_pretrained(
            self.settings.model_path,
            trust_remote_code=self.settings.trust_remote_code,
            torch_dtype=self.settings.torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=self.settings.low_cpu_mem_usage,
            tp_plan=None,
        )
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._device = next(model.parameters()).device
        return {
            "model_load_seconds": time.perf_counter() - started,
            "device": str(self._device),
        }

    def generate(
        self,
        messages: list[ImportanceMessage],
        settings: ImportanceSettings,
    ) -> GenerationResult:
        if self._torch is None or self._tokenizer is None or self._model is None or self._device is None:
            raise RuntimeError("LocalTransformersImportanceRuntime.generate() called before load().")
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        prompt = (
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if getattr(tokenizer, "chat_template", None)
            else "\n\n".join(message["content"] for message in messages)
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        input_len = inputs["input_ids"].shape[1]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=settings.max_new_tokens,
                do_sample=settings.do_sample,
                use_cache=settings.use_cache,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generated = output[0, input_len:]
        return GenerationResult(
            text=tokenizer.decode(generated, skip_special_tokens=True),
            generated_tokens=int(generated.numel()),
            generation_seconds=time.perf_counter() - started,
        )

    def _device_map(self, torch: Any) -> dict[str, int | str]:
        if self.settings.device == "cpu":
            return {"": "cpu"}
        if self.settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Memory Stream importance device=cuda requested but CUDA is unavailable.")
        if torch.cuda.is_available():
            return {"": 0}
        return {"": "cpu"}


__all__ = ["ImportanceRuntime", "LocalTransformersImportanceRuntime"]
