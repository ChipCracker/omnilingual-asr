"""Enumeration-style phone tokenizer family for CTC ASR.

Three vocab variants share this family — distinguished by metadata inside the
JSON vocab file:

  - p_sl           (vocab ~53):   atomic SAMPA phones + standalone `|` syllable marker
  - p_sl_fused     (vocab ~99):   atomic phones + phone-followed-by-`|` fused tokens
  - syl200_phn     (vocab ~252):  top-200 syllables + atomic-phone fallback

Input format (all three variants):  "do:|naU|vE|l@ gu:t"
    - phones concat within a syllable, no inter-phone whitespace
    - `|` between syllables of the same word
    - `<space>` between words (word boundary is dropped during encoding —
      vocab does NOT contain a space token, the ASR head should learn implicit
      word segmentation from the model's contextual representation)

Vocab JSON schema (see scripts/cv_tokenizer_analysis/build_*_vocab.py and
build_syl200_phn.py):

  {
    "tokens":   ["<unk>", "<pad>", "|", "a", "a:", ...],
    "unk_id":   0,
    "pad_id":   1,
    "marker_id": 2 or null,            # p_sl: 2, p_sl_fused: null
    "top_n_syllables": 200 or absent,  # syl200_phn only
    "vocab_size": <int>,
    ...
  }
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Final, final

import torch
from torch import Tensor
from typing_extensions import override

from fairseq2.data.tokenizers.tokenizer import TokenDecoder, TokenEncoder, Tokenizer
from fairseq2.data.tokenizers.vocab_info import VocabularyInfo
from fairseq2.device import Device

ENUM_PHONE_TOKENIZER_FAMILY: Final = "enum_phone_tokenizer"

# Multi-character SAMPA phones — mirrors split_sampa() in scripts/cv_tokenizer_analysis/prepare_cv_phones.py.
SAMPA_MULTI_CHAR: Final[frozenset[str]] = frozenset({
    "aI", "aU", "OY",
    "a:", "e:", "i:", "o:", "u:", "y:", "E:", "2:", "9:",
})

# encoder_kind tags
_KIND_P_SL: Final = "p_sl"
_KIND_P_SL_FUSED: Final = "p_sl_fused"
_KIND_SYL_PHN: Final = "syl_phn"
_KIND_PHI_CHUNK: Final = "phi_chunk"


def _detect_encoder_kind(meta: dict) -> str:
    """Pick which encode logic applies based on which fields are present."""
    if "chunk_pattern" in meta:
        return _KIND_PHI_CHUNK
    if "top_n_syllables" in meta:
        return _KIND_SYL_PHN
    if meta.get("marker_id") is not None:
        return _KIND_P_SL
    return _KIND_P_SL_FUSED


def _split_sampa(s: str) -> list[str]:
    """Greedy longest-match (max 2 chars) into atomic SAMPA phones."""
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if i + 1 < n and s[i:i + 2] in SAMPA_MULTI_CHAR:
            out.append(s[i:i + 2])
            i += 2
        else:
            out.append(s[i])
            i += 1
    return out


@final
class EnumPhoneEncoder(TokenEncoder):
    """Encodes phone strings into token-id sequences via enum-vocab lookup."""

    def __init__(
        self,
        vocab: list[str],
        token_to_id: dict[str, int],
        unk_id: int,
        encoder_kind: str,
        syllable_set: frozenset[str] | None,
        chunk_pattern: list[int] | None = None,
        *,
        device: Device | None = None,
        pin_memory: bool = False,
    ) -> None:
        self._vocab = vocab
        self._tok2id = token_to_id
        self._unk = unk_id
        self._kind = encoder_kind
        self._syl_set = syllable_set or frozenset()
        self._chunk_pattern = tuple(chunk_pattern) if chunk_pattern else ()
        self._device = device
        self._pin_memory = pin_memory

    def _to_ids(self, text: str) -> list[int]:
        if self._kind == _KIND_P_SL:
            return self._encode_p_sl(text)
        if self._kind == _KIND_P_SL_FUSED:
            return self._encode_p_sl_fused(text)
        if self._kind == _KIND_PHI_CHUNK:
            return self._encode_phi_chunk(text)
        return self._encode_syl_phn(text)

    def _encode_phi_chunk(self, text: str) -> list[int]:
        """Deterministic chunking of the per-word phon stream by a cyclical
        pattern of chunk-lengths.

        Used by the phi-chunk tokenizer family (philovar / phihivar). The
        vocab itself is shared; only `chunk_pattern` in JSON metadata
        differs. The encoder splits each whitespace-separated word into
        atomic SAMPA phones (via _split_sampa), then emits successive
        chunks of length given by chunk_pattern[i % len(chunk_pattern)].
        Word-end remainder shorter than the next pattern entry is emitted
        at whatever length fits (1/2/3 — all admitted in the shared vocab).
        """
        ids: list[int] = []
        pattern = self._chunk_pattern
        if not pattern:
            return ids
        for word in text.split():
            phons = _split_sampa(word)
            i = 0
            chunk_idx = 0
            n = len(phons)
            while i < n:
                target_len = pattern[chunk_idx % len(pattern)]
                actual_len = min(target_len, n - i)
                chunk = "".join(phons[i:i + actual_len])
                ids.append(self._tok2id.get(chunk, self._unk))
                i += actual_len
                chunk_idx += 1
        return ids

    def _encode_p_sl(self, text: str) -> list[int]:
        """Longest-match: emit atomic phones and standalone `|` tokens.

        Vocab contains atomic phones (a, o:, aU, …) and a `|` token. The space
        between words is dropped — vocab has no space token.
        """
        ids: list[int] = []
        for word in text.split():
            i = 0
            n = len(word)
            while i < n:
                if i + 1 < n and word[i:i + 2] in SAMPA_MULTI_CHAR:
                    tok = word[i:i + 2]
                    i += 2
                else:
                    tok = word[i]
                    i += 1
                ids.append(self._tok2id.get(tok, self._unk))
        return ids

    def _encode_p_sl_fused(self, text: str) -> list[int]:
        """Phones with lookahead: if a `|` follows the phone, emit the fused
        variant (e.g. `o:|`); otherwise emit the bare phone.

        Vocab does NOT contain a standalone `|` token — every syllable-final
        phone has a fused variant. The word boundary acts as an implicit
        syllable boundary: the last phone of each word is also emitted in
        fused form, mirroring the legacy `format_phones_atom_fused` semantics
        where every syllable-final phone (including word-final) carries `|`.
        """
        ids: list[int] = []
        for word in text.split():
            # Append a virtual `|` so the word-final phone is treated as
            # syllable-final.
            if not word.endswith("|"):
                word = word + "|"
            i = 0
            n = len(word)
            while i < n:
                if i + 1 < n and word[i:i + 2] in SAMPA_MULTI_CHAR:
                    phone = word[i:i + 2]
                    i += 2
                else:
                    phone = word[i]
                    i += 1
                if i < n and word[i] == "|":
                    phone = phone + "|"
                    i += 1
                ids.append(self._tok2id.get(phone, self._unk))
        return ids

    def _encode_syl_phn(self, text: str) -> list[int]:
        """Syllable-or-phone: try whole syllable first, fallback to atomic phones.

        Split each word at `|` to get syllables. If the syllable is in the
        top-N set, emit it as a single token; otherwise split it into atomic
        phones via `_split_sampa`.
        """
        ids: list[int] = []
        for word in text.split():
            for syl in word.split("|"):
                if not syl:
                    continue
                if syl in self._syl_set:
                    ids.append(self._tok2id.get(syl, self._unk))
                else:
                    for p in _split_sampa(syl):
                        ids.append(self._tok2id.get(p, self._unk))
        return ids

    @override
    def __call__(self, text: str) -> Tensor:
        ids = self._to_ids(text)
        t = torch.tensor(ids, dtype=torch.int64, device=self._device)
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
class EnumPhoneDecoder(TokenDecoder):
    """Decodes token-id sequences back to a phone string.

    Output convention: tokens are joined as-is, separated by a single space —
    callers can post-process to strip space-around-`|` if desired. Specials
    (<unk>, <pad>) are dropped.
    """

    def __init__(self, vocab: list[str], unk_id: int, pad_id: int) -> None:
        self._vocab = vocab
        self._unk = unk_id
        self._pad = pad_id

    def _tokens_from_ids(self, ids: list[int]) -> list[str]:
        out: list[str] = []
        for i in ids:
            if i == self._pad or i == self._unk:
                continue
            if 0 <= i < len(self._vocab):
                out.append(self._vocab[i])
        return out

    @override
    def __call__(self, token_indices: Tensor) -> str:
        ids = token_indices.tolist() if token_indices.ndim == 1 else token_indices.view(-1).tolist()
        return " ".join(self._tokens_from_ids(ids))

    @override
    def decode_from_tokens(self, tokens: Sequence[str]) -> str:
        return " ".join(t for t in tokens if t not in {"<unk>", "<pad>"})


@final
class EnumPhoneTokenizer(Tokenizer):
    """Tokenizer wrapping an enum-style phone/syllable vocabulary."""

    def __init__(
        self,
        vocab: list[str],
        unk_id: int,
        pad_id: int,
        encoder_kind: str,
        syllable_set: frozenset[str] | None,
        chunk_pattern: list[int] | None = None,
    ) -> None:
        self._vocab = vocab
        self._tok2id = {t: i for i, t in enumerate(vocab)}
        self._unk_id = unk_id
        self._pad_id = pad_id
        self._encoder_kind = encoder_kind
        self._syl_set = syllable_set
        self._chunk_pattern = chunk_pattern
        self._vocab_info = VocabularyInfo(
            size=len(vocab),
            unk_idx=unk_id,
            bos_idx=None,
            eos_idx=None,
            pad_idx=pad_id,
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
        if task is not None:
            raise ValueError(f"`task` must be `None`, but is '{task}' instead.")
        if lang is not None:
            raise ValueError(f"`lang` must be `None`, but is '{lang}' instead.")
        if mode is not None:
            raise ValueError(f"`mode` must be `None`, but is '{mode}' instead.")
        return EnumPhoneEncoder(
            self._vocab,
            self._tok2id,
            self._unk_id,
            self._encoder_kind,
            self._syl_set,
            chunk_pattern=self._chunk_pattern,
            device=device,
            pin_memory=pin_memory,
        )

    @override
    def create_raw_encoder(
        self, *, device: Device | None = None, pin_memory: bool = False
    ) -> TokenEncoder:
        return EnumPhoneEncoder(
            self._vocab,
            self._tok2id,
            self._unk_id,
            self._encoder_kind,
            self._syl_set,
            chunk_pattern=self._chunk_pattern,
            device=device,
            pin_memory=pin_memory,
        )

    @override
    def create_decoder(self, *, skip_special_tokens: bool = False) -> TokenDecoder:
        return EnumPhoneDecoder(self._vocab, self._unk_id, self._pad_id)

    @property
    @override
    def vocab_info(self) -> VocabularyInfo:
        return self._vocab_info


def load_enum_phone_tokenizer(path: Path, config: None) -> Tokenizer:
    """Load an enum-phone tokenizer from a JSON vocab file."""
    with Path(path).open("r", encoding="utf-8") as f:
        meta = json.load(f)
    vocab: list[str] = meta["tokens"]
    unk_id: int = meta.get("unk_id", 0)
    pad_id: int = meta.get("pad_id", 1)
    encoder_kind = _detect_encoder_kind(meta)
    if encoder_kind == _KIND_SYL_PHN:
        # Top-N syllables are the entries right after the specials (<unk>, <pad>)
        # and BEFORE the phone-inventory tail. Reconstruct that slice from tokens.
        top_n = int(meta.get("top_n_syllables", 0))
        n_specials = sum(1 for t in vocab[:5] if t in {"<unk>", "<pad>"})
        syllable_set: frozenset[str] | None = frozenset(vocab[n_specials:n_specials + top_n])
    else:
        syllable_set = None
    chunk_pattern = meta.get("chunk_pattern") if encoder_kind == _KIND_PHI_CHUNK else None
    return EnumPhoneTokenizer(
        vocab=vocab,
        unk_id=unk_id,
        pad_id=pad_id,
        encoder_kind=encoder_kind,
        syllable_set=syllable_set,
        chunk_pattern=chunk_pattern,
    )
