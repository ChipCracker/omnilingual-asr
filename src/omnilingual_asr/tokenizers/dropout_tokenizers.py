"""Subword-regularized SentencePiece tokenizers for CTC ASR (train-time sampling).

``create_encoder`` enables SentencePiece sampling -- BPE-dropout for a BPE model,
unigram lattice sampling for a unigram model -- so the CTC targets seen during
training vary per step (the head learns many valid segmentations of each word).
``create_raw_encoder`` and ``create_decoder`` are DETERMINISTIC, so evaluation,
cross-tokenizer fertility stats and the encode->decode roundtrip are unaffected
(the training data pipeline calls create_encoder; eval/stats use the others).

Two families:
  sampling_sentencepiece        plain SentencePiece (e.g. unigram-512) + sampling
  sampling_syllable_tokenizer   pyphen syllabification + SentencePiece + sampling
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

SAMPLING_SENTENCEPIECE_FAMILY: Final = "sampling_sentencepiece"
SAMPLING_SYLLABLE_TOKENIZER_FAMILY: Final = "sampling_syllable_tokenizer"

# Train-time sampling strength (typical BPE-dropout / unigram-sampling range 0.05-0.1).
_SAMPLING_ALPHA: Final = 0.1


def _sampling_sp_encoder(
    model: SentencePieceModel, *, device: Device | None = None, pin_memory: bool = False
) -> SentencePieceEncoder:
    return SentencePieceEncoder(
        model,
        device=device,
        pin_memory=pin_memory,
        enable_sampling=True,
        nbest_size=-1,
        alpha=_SAMPLING_ALPHA,
    )


@final
class _SyllabifyingEncoder(TokenEncoder):
    """pyphen-syllabify the text, then run a (sampling or plain) SP encoder."""

    def __init__(self, inner: SentencePieceEncoder, dic: pyphen.Pyphen) -> None:
        self._inner = inner
        self._dic = dic

    def _syllabify(self, text: str) -> str:
        return " ".join(self._dic.inserted(w) for w in text.split())

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
class _HyphenStrippingDecoder(TokenDecoder):
    """SentencePiece decode, then drop the '-' syllable separators (syllable family)."""

    def __init__(self, model: SentencePieceModel) -> None:
        self._inner = SentencePieceDecoder(model)

    @override
    def __call__(self, token_indices: Tensor) -> str:
        return self._inner(token_indices).replace("-", "")

    @override
    def decode_from_tokens(self, tokens: Sequence[str]) -> str:
        return self._inner.decode_from_tokens(tokens).replace("-", "")


@final
class SamplingSentencePieceTokenizer(Tokenizer):
    """Plain SentencePiece; create_encoder samples (train), raw/decoder deterministic."""

    def __init__(self, model: SentencePieceModel) -> None:
        self._model = model
        self._vocab_info = get_sentencepiece_vocabulary_info(model)

    @override
    def create_encoder(
        self, *, task: str | None = None, lang: str | None = None,
        mode: str | None = None, device: Device | None = None, pin_memory: bool = False,
    ) -> TokenEncoder:
        return _sampling_sp_encoder(self._model, device=device, pin_memory=pin_memory)

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TokenEncoder:
        return SentencePieceEncoder(self._model, device=device, pin_memory=pin_memory)

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TokenDecoder:
        return SentencePieceDecoder(self._model)

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._vocab_info


@final
class SamplingSyllableTokenizer(Tokenizer):
    """pyphen + SentencePiece; create_encoder samples (train), raw/decoder deterministic.

    Same vocab/behaviour as the syllable_tokenizer family except create_encoder applies
    SentencePiece BPE-dropout -- i.e. de_syllable + dynamic finer-fallback at train time.
    """

    def __init__(self, model: SentencePieceModel, lang: str = "de_DE") -> None:
        self._model = model
        self._dic = pyphen.Pyphen(lang=lang)
        self._vocab_info = get_sentencepiece_vocabulary_info(model)

    @override
    def create_encoder(
        self, *, task: str | None = None, lang: str | None = None,
        mode: str | None = None, device: Device | None = None, pin_memory: bool = False,
    ) -> TokenEncoder:
        inner = _sampling_sp_encoder(self._model, device=device, pin_memory=pin_memory)
        return _SyllabifyingEncoder(inner, self._dic)

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TokenEncoder:
        inner = SentencePieceEncoder(self._model, device=device, pin_memory=pin_memory)
        return _SyllabifyingEncoder(inner, self._dic)

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TokenDecoder:
        return _HyphenStrippingDecoder(self._model)

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._vocab_info


def load_sampling_sentencepiece_tokenizer(path: Path, config: None) -> Tokenizer:
    return SamplingSentencePieceTokenizer(load_sentencepiece_model(path))


def load_sampling_syllable_tokenizer(path: Path, config: None) -> Tokenizer:
    return SamplingSyllableTokenizer(load_sentencepiece_model(path))
