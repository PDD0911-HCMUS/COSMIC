"""Shared BERT text features with content-token pooling and a bounded cache."""

from collections import OrderedDict
from contextlib import nullcontext

import torch
from torch import nn


class BertTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # No Transformers/model construction during data export or imports.
        from transformers import BertModel, BertTokenizerFast

        self.config = config
        if config.checkpoint is not None:
            if not config.checkpoint.is_dir():
                raise FileNotFoundError(f"Missing local BERT directory: {config.checkpoint}")
            source, options = str(config.checkpoint), {"local_files_only": True}
        else:
            source = "google-bert/bert-base-uncased"
            options = {"cache_dir": str(config.cache_dir), "local_files_only": config.local_files_only}
        self.tokenizer = BertTokenizerFast.from_pretrained(source, **options)
        self.backbone = BertModel.from_pretrained(source, add_pooling_layer=False, **options)
        if config.max_length > self.backbone.config.max_position_embeddings:
            raise ValueError("osg_encoder.max_length exceeds the BERT position limit")
        self.hidden_size = self.backbone.config.hidden_size
        self._cache = OrderedDict()
        if config.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    def clear_cache(self):
        self._cache.clear()

    @property
    def cache_entries(self):
        return len(self._cache)

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def _encode_chunk(self, texts, device):
        encoded = self.tokenizer(texts, padding=True, truncation=False,
                                 return_special_tokens_mask=True, return_tensors="pt")
        if encoded["input_ids"].shape[1] > self.config.max_length:
            raise ValueError("OSG text exceeds osg_encoder.max_length; increase the limit "
                             "rather than truncate a subject/predicate/object")
        special = encoded.pop("special_tokens_mask").to(device)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        context = torch.no_grad() if self.config.freeze_backbone else nullcontext()
        with context:
            features = self.backbone(**encoded, return_dict=True).last_hidden_state
            content_mask = encoded["attention_mask"].bool() & ~special.bool()
            weights = content_mask.unsqueeze(-1).to(features.dtype)
            pooled = (features * weights).sum(1) / weights.sum(1).clamp_min(1)
        return pooled

    def forward(self, texts):
        parameter = next(self.backbone.parameters())
        if not texts:
            return parameter.new_empty((0, self.hidden_size))
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("BERT inputs must be nonempty text strings")
        # Deduplicate within each forward, including A/B, without caching any
        # trainable projections. Disable persistent cache for trainable BERT/AMP.
        use_cache = (self.config.freeze_backbone and self.config.cache_size > 0
                     and not torch.is_autocast_enabled(parameter.device.type))
        unique = list(dict.fromkeys(texts))
        vectors, missing = {}, []
        for text in unique:
            if use_cache and text in self._cache:
                self._cache.move_to_end(text)
                vectors[text] = self._cache[text].to(device=parameter.device, dtype=parameter.dtype)
            else:
                missing.append(text)
        for start in range(0, len(missing), self.config.text_batch_size):
            chunk = missing[start:start + self.config.text_batch_size]
            pooled = self._encode_chunk(chunk, parameter.device)
            for text, vector in zip(chunk, pooled):
                vectors[text] = vector
                if use_cache:
                    self._cache[text] = vector.detach().to(device="cpu", dtype=torch.float32).clone()
                    while len(self._cache) > self.config.cache_size:
                        self._cache.popitem(last=False)
        return torch.stack([vectors[text] for text in texts])
