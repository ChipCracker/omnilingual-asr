# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from fairseq2.models.wav2vec2.asr.config import Wav2Vec2AsrConfig
from fairseq2.models.wav2vec2.config import Wav2Vec2Config
from fairseq2.runtime.config_registry import ConfigRegistrar, get_config
from fairseq2.runtime.dependency import DependencyContainer, DependencyResolver


def register_omnilingual_asr_wav2vec2_asr_configs(
    container: DependencyContainer,
) -> None:
    arch = ConfigRegistrar(container, Wav2Vec2AsrConfig)

    @arch("300m", advanced=True)
    def _300m_asr(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        # base_10h and large_lv60k are original wav2vec2 configurations living in fs2
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config

        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 2475
        config.target_vocab_size = 9812

        return config

    @arch("1b", advanced=True)
    def _1b_asr(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "1b"
        ).encoder_config

        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 9812

        return config

    @arch("3b", advanced=True)
    def _3b_asr(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "3b"
        ).encoder_config

        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 9812

        return config

    @arch("7b", advanced=True)
    def _7b_asr(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "7b"
        ).encoder_config

        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 9812

        return config

    @arch("300m_48", advanced=True)
    def _300m_asr_48(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 48
        return config

    @arch("300m_64", advanced=True)
    def _300m_asr_64(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 64
        return config

    @arch("300m_96", advanced=True)
    def _300m_asr_96(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 96
        return config

    @arch("300m_128", advanced=True)
    def _300m_asr_128(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 128
        return config

    @arch("300m_192", advanced=True)
    def _300m_asr_192(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 192
        return config

    @arch("300m_256", advanced=True)
    def _300m_asr_256(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 256
        return config

    @arch("300m_512", advanced=True)
    def _300m_asr_512(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 512
        return config

    @arch("300m_514", advanced=True)
    def _300m_asr_514(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        # 512-vocab variants extended with the KSOF disfluency markers
        # <uf> (id 512) and <m> (id 513).
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 514
        return config

    @arch("300m_1024", advanced=True)
    def _300m_asr_1024(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 1024
        return config

    @arch("300m_2048", advanced=True)
    def _300m_asr_2048(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 2048
        return config

    @arch("300m_3072", advanced=True)
    def _300m_asr_3072(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 3072
        return config

    @arch("300m_3840", advanced=True)
    def _300m_asr_3840(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 3840
        return config

    @arch("300m_1500", advanced=True)
    def _300m_asr_1500(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        # de_syl_hybrid (SYL+PHN-BPE v2): SP-BPE vocab=1500
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 1500
        return config

    @arch("300m_4096", advanced=True)
    def _300m_asr_4096(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        # de_phon_free (PPC-BPE v2) and de_phon_bound (PP+SL-BPE v2): vocab=4096
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 4096
        return config

    @arch("300m_5000", advanced=True)
    def _300m_asr_5000(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        # de_ortho_bpe (TXT-BPE) and de_ortho_syl (PSC-BPE): vocab=5000
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 5000
        return config

    @arch("300m_103", advanced=True)
    def _300m_asr_103(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        # IPA CommonPhone: pad(0) + 101 IPA + unk(102) = 103 classes
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 103
        return config

    @arch("300m_44", advanced=True)
    def _300m_asr_44(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        # German SAMPA: CTC blank(0) + <pad>(1) + 33 base chars + 9 multi-char UDS = 44
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 44
        return config

    @arch("1b_512", advanced=True)
    def _1b_asr_512(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "1b"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 512
        return config

    @arch("1b_1024", advanced=True)
    def _1b_asr_1024(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "1b"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 1024
        return config

    @arch("1b_2048", advanced=True)
    def _1b_asr_2048(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "1b"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 2048
        return config

    @arch("1b_103", advanced=True)
    def _1b_asr_103(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        # IPA CommonPhone: pad(0) + 101 IPA + unk(102) = 103 classes
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "1b"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 103
        return config

    @arch("1b_44", advanced=True)
    def _1b_asr_44(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        # German SAMPA: 44 classes (see 300m_44)
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "1b"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 44
        return config

    @arch("1b_64", advanced=True)
    def _1b_asr_64(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        # 64 classes — buffer for SAMPA-D tokenizer with short tense vowels (~50 tokens)
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "1b"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 64
        return config

    @arch("3b_103", advanced=True)
    def _3b_asr_103(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        # IPA CommonPhone: pad(0) + 101 IPA + unk(102) = 103 classes
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "3b"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 103
        return config

    @arch("3b_64", advanced=True)
    def _3b_asr_64(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        # 64 classes — buffer for SAMPA-D tokenizer with short tense vowels (~50 tokens)
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "3b"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 64
        return config

    @arch("3b_512", advanced=True)
    def _3b_asr_512(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "3b"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 512
        return config

    @arch("3b_1024", advanced=True)
    def _3b_asr_1024(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "3b"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 1024
        return config

    @arch("3b_2048", advanced=True)
    def _3b_asr_2048(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "3b"
        ).encoder_config
        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1
        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 2048
        return config

    @arch("300m_v2", advanced=True)
    def _300m_asr_v2(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "large_lv60k"
        ).encoder_config

        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 2475
        config.target_vocab_size = 10288

        return config

    @arch("1b_v2", advanced=True)
    def _1b_asr_v2(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "1b"
        ).encoder_config

        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 10288

        return config

    @arch("3b_v2", advanced=True)
    def _3b_asr_v2(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "3b"
        ).encoder_config

        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 10288

        return config

    @arch("7b_v2", advanced=True)
    def _7b_asr_v2(resolver: DependencyResolver) -> Wav2Vec2AsrConfig:
        config = get_config(resolver, Wav2Vec2AsrConfig, "base_10h")
        config.encoder_config = get_config(
            resolver, Wav2Vec2Config, "7b"
        ).encoder_config

        config.encoder_config.dropout_p = 0.0
        config.encoder_config.attn_dropout_p = 0.0
        config.encoder_config.ffn_inner_dropout_p = 0.1
        config.encoder_config.layer_drop_p = 0.1

        config.use_masking = False
        config.max_temporal_mask_prob = 0.0
        config.max_spatial_mask_prob = 0.0
        config.target_vocab_size = 10288

        return config
