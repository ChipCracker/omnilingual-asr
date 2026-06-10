"""Orthographic syllable-unit tokenizer family for CTC ASR.

Every token is one WHOLE pyphen syllable (if that syllable is in the vocab) or
one character (fallback). A literal space token marks word boundaries — kept so
that word-level WER stays computable. No BPE.

  Encode: text → pyphen-syllabify each word → for each syllable: emit the
          whole-syllable token if in vocab, else char-by-char; emit the space
          token between words.
  Decode: concatenate token surfaces (the space token's surface is ' ').

Vocab JSON is produced by scripts/build_syllable_unit_vocab.py:
  {"tokens": ["<unk>", "<pad>", " ", "a", ..., "hal", ...],
   "unk_id": 0, "pad_id": 1, "space_id": 2, "lang": "de_DE", ...}
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final, final

import pyphen
import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.data.tokenizers.tokenizer import TokenDecoder, TokenEncoder, Tokenizer
from fairseq2.data.tokenizers.vocab_info import VocabularyInfo
from fairseq2.device import Device

SYLLABLE_UNIT_TOKENIZER_FAMILY: Final = "syllable_unit_tokenizer"


@final
class SyllableUnitEncoder(TokenEncoder):
    def __init__(
        self,
        vocab: list[str],
        token_to_id: dict[str, int],
        unk_id: int,
        space_id: int,
        dic: pyphen.Pyphen,
        *,
        device: Device | None = None,
        pin_memory: bool = False,
    ) -> None:
        self._vocab = vocab
        self._tok2id = token_to_id
        self._unk = unk_id
        self._space = space_id
        self._dic = dic
        self._device = device
        self._pin_memory = pin_memory

    def _to_ids(self, text: str) -> list[int]:
        ids: list[int] = []
        for wi, word in enumerate(text.split()):
            if wi > 0:
                ids.append(self._space)
            for syl in self._dic.inserted(word).split("-"):
                if not syl:
                    continue
                tid = self._tok2id.get(syl)
                if tid is not None:
                    ids.append(tid)
                else:
                    for ch in syl:
                        ids.append(self._tok2id.get(ch, self._unk))
        return ids

    @override
    def __call__(self, text: str) -> Tensor:
        t = torch.tensor(self._to_ids(text), dtype=torch.int64, device=self._device)
        if self._pin_memory and (self._device is None or self._device.type == "cpu"):
            t = t.pin_memory()
        return t

    @override
    def encode_as_tokens(self, text: str) -> list[str]:
        return [self._vocab[i] if i < len(self._vocab) else "<unk>" for i in self._to_ids(text)]

    @property
    @override
    def prefix_indices(self) -> Tensor | None:
        return None

    @property
    @override
    def suffix_indices(self) -> Tensor | None:
        return None


@final
class SyllableUnitDecoder(TokenDecoder):
    """Concatenate token surfaces; the space token reconstructs word boundaries."""

    def __init__(self, vocab: list[str], unk_id: int, pad_id: int) -> None:
        self._vocab = vocab
        self._unk = unk_id
        self._pad = pad_id

    def _surfaces(self, ids: list[int]) -> list[str]:
        return [
            self._vocab[i]
            for i in ids
            if i not in (self._unk, self._pad) and 0 <= i < len(self._vocab)
        ]

    @override
    def __call__(self, token_indices: Tensor) -> str:
        ids = token_indices.tolist() if token_indices.ndim == 1 else token_indices.view(-1).tolist()
        return "".join(self._surfaces(ids))

    @override
    def decode_from_tokens(self, tokens: Sequence[str]) -> str:
        return "".join(t for t in tokens if t not in {"<unk>", "<pad>"})


@final
class SyllableUnitTokenizer(Tokenizer):
    def __init__(
        self,
        vocab: list[str],
        unk_id: int,
        pad_id: int,
        space_id: int,
        lang: str = "de_DE",
    ) -> None:
        self._vocab = vocab
        self._tok2id = {t: i for i, t in enumerate(vocab)}
        self._unk_id = unk_id
        self._pad_id = pad_id
        self._space_id = space_id
        self._dic = pyphen.Pyphen(lang=lang)
        self._vocab_info = VocabularyInfo(
            size=len(vocab), unk_idx=unk_id, bos_idx=None, eos_idx=None, pad_idx=pad_id,
        )

    @override
    def create_encoder(
        self,
        *,
        task: str | None = None,
        lang: str | None = None,
        mode: str | None = None,
        device: Device | None = None,
        pin_memory: bool = False,
    ) -> TokenEncoder:
        for nm, val in (("task", task), ("lang", lang), ("mode", mode)):
            if val is not None:
                raise ValueError(f"`{nm}` must be `None`, but is '{val}' instead.")
        return SyllableUnitEncoder(
            self._vocab, self._tok2id, self._unk_id, self._space_id, self._dic,
            device=device, pin_memory=pin_memory,
        )

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TokenEncoder:
        return SyllableUnitEncoder(
            self._vocab, self._tok2id, self._unk_id, self._space_id, self._dic,
            device=device, pin_memory=pin_memory,
        )

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TokenDecoder:
        return SyllableUnitDecoder(self._vocab, self._unk_id, self._pad_id)

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._vocab_info


def load_syllable_unit_tokenizer(path: Path, config: None) -> Tokenizer:
    meta = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab: list[str] = meta["tokens"]
    unk_id = int(meta.get("unk_id", 0))
    pad_id = int(meta.get("pad_id", 1))
    space_id = int(meta.get("space_id", vocab.index(" ")))
    lang = meta.get("lang", "de_DE")
    return SyllableUnitTokenizer(vocab, unk_id, pad_id, space_id, lang=lang)
