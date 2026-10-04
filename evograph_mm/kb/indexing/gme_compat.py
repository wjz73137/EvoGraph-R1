"""Local GME inference on Transformers 5, without changing its trained weights.

The official GME forward predates Qwen2-VL's split visual/text model. In
particular, it uses ordinary sequential positions, not the generative model's
automatic spatial M-RoPE positions. Keep that forward, prompt and last-token
pooling while using the installed native implementation and weight conversion.
No remote model code, network fetch, or dependency modification is performed.
"""

from __future__ import annotations

from pathlib import Path


DEFAULT_INSTRUCTION = "You are a helpful assistant."


def gme_prompt(text: str | None, has_image: bool, instruction: str | None = None) -> str:
    content = "<|vision_start|><|image_pad|><|vision_end|>" if has_image else ""
    content += text or ""
    return (f"<|im_start|>system\n{instruction or DEFAULT_INSTRUCTION}<|im_end|>\n"
            f"<|im_start|>user\n{content}<|im_end|>\n"
            "<|im_start|>assistant\n<|endoftext|>")


def last_token_pool(hidden, attention_mask):
    """Select the actual last non-padding position, for either padding side."""
    import torch
    positions = torch.arange(attention_mask.shape[1], device=hidden.device)
    indices = positions.expand_as(attention_mask).masked_fill(attention_mask == 0, -1).max(1).values
    if bool((indices < 0).any()):
        raise ValueError("cannot pool an empty sequence")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]


class NativeGME:
    def __init__(self, model_path: str | Path, device: str | None, batch_size: int = 1):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.batch_size = batch_size
        device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model, info = Qwen2VLForConditionalGeneration.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
            dtype=dtype, attn_implementation="sdpa", device_map={"": str(self.device)},
            output_loading_info=True,
        )
        self.loading_info = info
        # A silently random embedding/vision tensor would invalidate retrieval.
        missing = set(info.get("missing_keys", []))
        tied_head = (self.model.lm_head.weight.data_ptr()
                     == self.model.model.language_model.embed_tokens.weight.data_ptr())
        if tied_head:
            missing.discard("lm_head.weight")
        if missing or info.get("unexpected_keys") or info.get("mismatched_keys") or info.get("error_msgs"):
            raise RuntimeError(f"GME weights did not load completely: {info}")
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
            min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
        )
        self.processor.tokenizer.padding_side = "right"
        self.max_length = 1800

    def _encode(self, texts, images=None, instruction=None):
        import torch
        texts = list(texts)
        if images is not None:
            images = list(images)
            if len(images) != len(texts):
                raise ValueError("GME text/image batch sizes differ")
        if not texts:
            return torch.empty((0, 1536), dtype=torch.float32, device=self.device)
        output = []
        for start in range(0, len(texts), self.batch_size):
            stop = start + self.batch_size
            image_batch = None if images is None else images[start:stop]
            prompts = [gme_prompt(t, image_batch is not None, instruction) for t in texts[start:stop]]
            inputs = self.processor(text=prompts, images=image_batch, padding=True,
                                    truncation=True, max_length=self.max_length,
                                    return_tensors="pt").to(self.device)
            with torch.inference_mode():
                embeds = self.model.get_input_embeddings()(inputs["input_ids"])
                if image_batch is not None:
                    visual = self.model.model.visual
                    result = visual(inputs["pixel_values"].to(visual.get_dtype()),
                                    grid_thw=inputs["image_grid_thw"], return_dict=True)
                    image_embeds = result.pooler_output.to(embeds.device, embeds.dtype)
                    mask = inputs["input_ids"] == self.model.config.image_token_id
                    if int(mask.sum()) != len(image_embeds):
                        raise RuntimeError("GME image tokens/features do not correspond")
                    embeds[mask] = image_embeds
                # Explicitly bypass automatic multimodal position generation,
                # reproducing the embedding provider's original forward.
                result = self.model.model.language_model(
                    inputs_embeds=embeds, attention_mask=inputs["attention_mask"],
                    use_cache=False, return_dict=True,
                )
                pooled = last_token_pool(result.last_hidden_state, inputs["attention_mask"])
                output.append(torch.nn.functional.normalize(pooled.float(), p=2, dim=1))
        return torch.cat(output, dim=0)

    def get_text_embeddings(self, texts, instruction=None):
        return self._encode(texts, instruction=instruction)

    def get_image_embeddings(self, images, instruction=None, **kwargs):
        return self._encode([None] * len(images), images, instruction=instruction)

    def get_fused_embeddings(self, texts, images, instruction=None):
        # Only local paths/Pillow images are accepted. No URL download here.
        from PIL import Image, ImageOps
        prepared = []
        for value in images:
            if isinstance(value, Image.Image):
                prepared.append(value.convert("RGB"))
            else:
                path = Path(value)
                if not path.is_file():
                    raise ValueError(f"GME image must be an existing local file: {path}")
                with Image.open(path) as image:
                    prepared.append(ImageOps.exif_transpose(image).convert("RGB"))
        return self._encode(texts, prepared, instruction=instruction)
