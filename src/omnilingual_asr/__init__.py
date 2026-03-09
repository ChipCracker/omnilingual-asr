# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from types import NoneType

from fairseq2.composition.assets import register_package_assets
from fairseq2.composition.models import register_model_family
from fairseq2.composition.tokenizers import register_tokenizer_family
from fairseq2.data.tokenizers.tokenizer import Tokenizer
from fairseq2.runtime.dependency import DependencyContainer

from omnilingual_asr.models.wav2vec2_asr.config import (
    register_omnilingual_asr_wav2vec2_asr_configs,
)
from omnilingual_asr.models.wav2vec2_llama import (
    WAV2VEC2_LLAMA_FAMILY,
    Wav2Vec2LlamaConfig,
    Wav2Vec2LlamaModel,
    apply_fsdp_to_wav2vec2_llama,
    convert_wav2vec2_llama_state_dict,
    create_wav2vec2_llama_model,
    register_wav2vec2_llama_configs,
)
from omnilingual_asr.models.wav2vec2_ssl.config import (
    register_omnilingual_asr_wav2vec2_ssl_configs,
)

__version__ = "0.2.0"


def setup_fairseq2_extension(container: DependencyContainer) -> None:
    # Make sure that the default fairseq2 asset store can resolve cards under
    # the directory <omnilingual_asr>/cards.
    register_package_assets(container, "omnilingual_asr.cards")

    _register_models(container)
    _register_tokenizers(container)


def _register_tokenizers(container: DependencyContainer) -> None:
    from omnilingual_asr.tokenizers.syllable_tokenizer import (
        SYLLABLE_TOKENIZER_FAMILY,
        load_syllable_tokenizer,
    )

    register_tokenizer_family(
        container,
        SYLLABLE_TOKENIZER_FAMILY,
        kls=Tokenizer,
        config_kls=NoneType,
        loader=load_syllable_tokenizer,
    )


def _register_models(container: DependencyContainer) -> None:

    # Only adding custom wav2vec2 archs for wav2vec2_ssl model in fs2
    register_omnilingual_asr_wav2vec2_ssl_configs(container)

    # Only adding custom wav2vec2 archs for wav2vec2_asr model in fs2
    register_omnilingual_asr_wav2vec2_asr_configs(container)

    # wav2vec2 llama
    register_model_family(
        container,
        WAV2VEC2_LLAMA_FAMILY,
        kls=Wav2Vec2LlamaModel,
        config_kls=Wav2Vec2LlamaConfig,
        factory=create_wav2vec2_llama_model,
        fsdp_applier=apply_fsdp_to_wav2vec2_llama,
        state_dict_converter=convert_wav2vec2_llama_state_dict,
    )

    register_wav2vec2_llama_configs(container)
