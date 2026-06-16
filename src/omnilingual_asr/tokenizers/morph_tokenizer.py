"""Morpheme-aware SentencePiece tokenizer for CTC ASR (experiment #6, de_morph_bpe).

Mirrors the syllable_tokenizer family but replaces pyphen syllabification with a
*frozen* Morfessor morphological segmentation:
  Encoding: text -> Morfessor morphs joined by '-' -> SentencePiece
  Decoding: SentencePiece -> strip '-'
A clean boundary-type ablation against de_syllable (morpheme vs syllable boundary,
everything else identical).

The Morfessor model is loaded from ``<sp_model_stem>.morfessor`` next to the SP model.
``morfessor`` is imported LAZILY (inside the loader), so importing omnilingual_asr never
fails when morfessor is not installed -- only constructing this tokenizer needs it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final, final

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

MORPH_TOKENIZER_FAMILY: Final = "morph_tokenizer"


@final
class MorphEncoder(TokenEncoder):
    """Morfessor-segment each word, join morphs with '-', then SentencePiece-encode."""

    def __init__(
        self, model: SentencePieceModel, morf, *,
        device: Device | None = None, pin_memory: bool = False,
    ) -> None:
        self._morf = morf
        self._cache: dict[str, str] = {}
        self._inner = SentencePieceEncoder(model, device=device, pin_memory=pin_memory)

    def _seg_word(self, w: str) -> str:
        s = self._cache.get(w)
        if s is None:
            morphs, _ = self._morf.viterbi_segment(w)
            s = "-".join(morphs)
            self._cache[w] = s
        return s

    def _morphify(self, text: str) -> str:
        return " ".join(self._seg_word(w) for w in text.split())

    @override
    def __call__(self, text: str) -> Tensor:
        return self._inner(self._morphify(text))

    @override
    def encode_as_tokens(self, text: str) -> list[str]:
        return self._inner.encode_as_tokens(self._morphify(text))

    @property
    @override
    def prefix_indices(self) -> Tensor | None:
        return self._inner.prefix_indices

    @property
    @override
    def suffix_indices(self) -> Tensor | None:
        return self._inner.suffix_indices


@final
class MorphDecoder(TokenDecoder):
    def __init__(self, model: SentencePieceModel) -> None:
        self._inner = SentencePieceDecoder(model)

    @override
    def __call__(self, token_indices: Tensor) -> str:
        return self._inner(token_indices).replace("-", "")

    @override
    def decode_from_tokens(self, tokens: Sequence[str]) -> str:
        return self._inner.decode_from_tokens(tokens).replace("-", "")


@final
class MorphSentencePieceTokenizer(Tokenizer):
    def __init__(self, model: SentencePieceModel, morf) -> None:
        self._model = model
        self._morf = morf
        self._vocab_info = get_sentencepiece_vocabulary_info(model)

    @override
    def create_encoder(
        self, *, task: str | None = None, lang: str | None = None,
        mode: str | None = None, device: Device | None = None, pin_memory: bool = False,
    ) -> TokenEncoder:
        return MorphEncoder(self._model, self._morf, device=device, pin_memory=pin_memory)

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TokenEncoder:
        return MorphEncoder(self._model, self._morf, device=device, pin_memory=pin_memory)

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TokenDecoder:
        return MorphDecoder(self._model)

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._vocab_info


def load_morph_tokenizer(path: Path, config: None) -> Tokenizer:
    import morfessor  # lazy: never break the omnilingual_asr import if absent

    model = load_sentencepiece_model(path)
    morf_path = Path(str(path).rsplit(".", 1)[0] + ".morfessor")
    morf = morfessor.MorfessorIO().read_binary_model_file(str(morf_path))
    return MorphSentencePieceTokenizer(model, morf)
