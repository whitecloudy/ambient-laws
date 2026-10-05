# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

def __getattr__(name):
    if name in ('RenewSemanticCommunicationDataset', 'renewSemanticCommunicationDataset'):
        from training.semantic_comm_dataset import (
            RenewSemanticCommunicationDataset,
            renewSemanticCommunicationDataset,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
