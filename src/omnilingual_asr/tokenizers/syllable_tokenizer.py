"""Syllable-aware SentencePiece tokenizer family for CTC ASR.

Encoding: text → pyphen syllabification → SentencePiece encoding
Decoding: SentencePiece decoding → remove hyphens
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, final

import pyphen
from torch import Tensor
from typing_extensions import override

from fairseq2.data.tokenizers.sentencepiece import (
    SentencePieceDecoder,
    SentencePieceEncoder,
    SentencePieceModel,
    get_sentencepiece_vocabulary_info,
    load_sentencepiece_model,
)
from fairseq2.data.tokenizers.tokenizer import TokenDecoder, TokenEncoder, Tokenizer
from fairseq2.data.tokenizers.vocab_info import VocabularyInfo
from fairseq2.device import Device

SYLLABLE_TOKENIZER_FAMILY: Final = "syllable_tokenizer"


@final
class SyllableEncoder(TokenEncoder):
    """Syllabifies text with pyphen, then encodes with SentencePiece."""

    def __init__(
        self,
        model: SentencePieceModel,
        dic: pyphen.Pyphen,
        *,
        device: Device | None = None,
        pin_memory: bool = False,
    ) -> None:
        self._dic = dic
        self._inner = SentencePieceEncoder(model, device=device, pin_memory=pin_memory)

    def _syllabify(self, text: str) -> str:
        words = text.split()
        return " ".join(self._dic.inserted(w) for w in words)

    @override
    def __call__(self, text: str) -> Tensor:
        return self._inner(self._syllabify(text))

    @override
    def encode_as_tokens(self, text: str) -> list[str]:
        return self._inner.encode_as_tokens(self._syllabify(text))

    @property
    @override
    def prefix_indices(self) -> Tensor | None:
        return self._inner.prefix_indices

    @property
    @override
    def suffix_indices(self) -> Tensor | None:
        return self._inner.suffix_indices


@final
class SyllableDecoder(TokenDecoder):
    """Decodes SentencePiece tokens, then removes hyphen syllable separators."""

    def __init__(self, model: SentencePieceModel) -> None:
        self._inner = SentencePieceDecoder(model)

    @override
    def __call__(self, token_indices: Tensor) -> str:
        text = self._inner(token_indices)
        return text.replace("-", "")

    @override
    def decode_from_tokens(self, tokens: Sequence[str]) -> str:
        text = self._inner.decode_from_tokens(tokens)
        return text.replace("-", "")


@final
class SyllableSentencePieceTokenizer(Tokenizer):
    """Wraps RawSentencePieceTokenizer with pyphen syllabification."""

    def __init__(self, model: SentencePieceModel, lang: str = "de_DE") -> None:
        self._model = model
        self._dic = pyphen.Pyphen(lang=lang)
        self._vocab_info = get_sentencepiece_vocabulary_info(model)

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
        if task is not None:
            raise ValueError(f"`task` must be `None`, but is '{task}' instead.")
        if lang is not None:
            raise ValueError(f"`lang` must be `None`, but is '{lang}' instead.")
        if mode is not None:
            raise ValueError(f"`mode` must be `None`, but is '{mode}' instead.")
        return SyllableEncoder(self._model, self._dic, device=device, pin_memory=pin_memory)

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TokenEncoder:
        return SyllableEncoder(self._model, self._dic, device=device, pin_memory=pin_memory)

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TokenDecoder:
        return SyllableDecoder(self._model)

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._vocab_info


def load_syllable_tokenizer(path: Path, config: None) -> Tokenizer:
    """Load a syllable SentencePiece tokenizer from a model file."""
    model = load_sentencepiece_model(path)
    return SyllableSentencePieceTokenizer(model)
