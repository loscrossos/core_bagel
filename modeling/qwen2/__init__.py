# Copyright 2024 The Qwen Team and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from transformers.utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_tokenizers_available,
    is_torch_available,
)


_import_structure = {
    "configuration_qwen2": ["Qwen2Config"],
    "tokenization_qwen2": ["Qwen2Tokenizer"],
}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_qwen2_fast"] = ["Qwen2TokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_qwen2"] = [
        "Qwen2ForCausalLM",
        "Qwen2Model",
        "Qwen2PreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_qwen2 import Qwen2Config
    from .tokenization_qwen2 import Qwen2Tokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_qwen2_fast import Qwen2TokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_qwen2 import (
            Qwen2ForCausalLM,
            Qwen2Model,
            Qwen2PreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
