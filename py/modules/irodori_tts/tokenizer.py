from collections.abc import Iterable

import torch


class PretrainedTextTokenizer:
    """
    Hugging Face tokenizer wrapper for text conditioning.
    - right-padding for stable positional behavior
    - optional explicit BOS prepend
    """

    def __init__(self, tokenizer, add_bos: bool = True) -> None:
        self.tokenizer = tokenizer
        self.add_bos = bool(add_bos)
        # TTS collator uses fixed-length right-padding; enforce this regardless of pretrained defaults.
        self.tokenizer.padding_side = "right"

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None and self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                raise ValueError(
                    "Tokenizer has no pad_token_id (and no eos_token fallback). "
                    "Set a pad token before training/inference."
                )

        if self.add_bos and self.tokenizer.bos_token_id is None:
            raise ValueError("Tokenizer has no bos_token_id but add_bos=True.")

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        add_bos: bool = True,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> "PretrainedTextTokenizer":
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required for pretrained text tokenization. "
                "Install with `pip install transformers sentencepiece`."
            ) from exc

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                repo_id,
                use_fast=True,
                trust_remote_code=False,
                local_files_only=local_files_only,
                revision=revision,
            )
        except (ValueError, KeyError, OSError) as exc:
            # transformers < 5.x cannot resolve the v5-style "TokenizersBackend"
            # tokenizer_class emitted by v4 checkpoints. The tokenizer is fully
            # described by tokenizer.json + special tokens, so build a generic
            # fast tokenizer directly from the bundled files.
            import json
            import os

            from transformers import PreTrainedTokenizerFast

            tok_file = os.path.join(str(repo_id), "tokenizer.json")
            if not os.path.isfile(tok_file):
                raise
            cfg_file = os.path.join(str(repo_id), "tokenizer_config.json")
            special: dict[str, str] = {}
            if os.path.isfile(cfg_file):
                with open(cfg_file, encoding="utf-8") as fh:
                    raw_cfg = json.load(fh)
                for key in (
                    "bos_token",
                    "eos_token",
                    "unk_token",
                    "pad_token",
                    "cls_token",
                    "sep_token",
                    "mask_token",
                ):
                    value = raw_cfg.get(key)
                    if isinstance(value, str) and value:
                        special[key] = value
            print(
                "[IrodoriTTS] tokenizer: AutoTokenizer failed "
                f"({type(exc).__name__}); loading tokenizer.json via "
                "PreTrainedTokenizerFast fallback.",
                flush=True,
            )
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tok_file, **special)
        return cls(tokenizer=tokenizer, add_bos=add_bos)

    @property
    def vocab_size(self) -> int:
        return int(len(self.tokenizer))

    @property
    def bos_token_id(self) -> int | None:
        return self.tokenizer.bos_token_id

    @property
    def pad_token_id(self) -> int:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise RuntimeError("pad_token_id is unexpectedly None.")
        return int(pad_id)

    def encode(self, text: str, add_bos: bool | None = None) -> torch.Tensor:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        use_bos = self.add_bos if add_bos is None else bool(add_bos)
        if use_bos:
            bos_id = self.bos_token_id
            if bos_id is None:
                raise ValueError("Tokenizer has no bos_token_id but BOS prepend was requested.")
            token_ids.insert(0, int(bos_id))
        return torch.tensor(token_ids, dtype=torch.long)

    def batch_encode(
        self,
        texts: Iterable[str],
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        texts = list(texts)
        if not texts:
            raise ValueError("texts must contain at least one item.")
        if max_length is None:
            encoded = [self.encode(t) for t in texts]
            max_length = max(max(x.numel(), 1) for x in encoded)
        if max_length <= 0:
            raise ValueError(f"max_length must be > 0, got {max_length}")

        if self.add_bos:
            bos_id = self.bos_token_id
            if bos_id is None:
                raise ValueError("Tokenizer has no bos_token_id but BOS prepend was requested.")
            if max_length == 1:
                batch = torch.full(
                    (len(texts), 1),
                    fill_value=int(bos_id),
                    dtype=torch.long,
                )
                mask = torch.ones((len(texts), 1), dtype=torch.bool)
                return batch, mask
            body_max_length = max_length - 1
        else:
            body_max_length = max_length

        encoded_batch = self.tokenizer(
            texts,
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=body_max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        body_ids = encoded_batch["input_ids"].to(dtype=torch.long)
        body_mask = encoded_batch["attention_mask"].to(dtype=torch.bool)

        if not self.add_bos:
            return body_ids, body_mask

        batch = torch.full(
            (len(texts), max_length),
            fill_value=self.pad_token_id,
            dtype=torch.long,
        )
        mask = torch.zeros((len(texts), max_length), dtype=torch.bool)
        batch[:, 0] = int(self.bos_token_id)
        mask[:, 0] = True
        batch[:, 1:] = body_ids
        mask[:, 1:] = body_mask
        return batch, mask
