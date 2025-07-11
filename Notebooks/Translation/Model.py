from transformers import PreTrainedModel
from ModelConfig import ModelConfig

import sys
import os

# Add Language/ as parent Directory
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))

from src.models.Translation import TranslationModel

class Model(PreTrainedModel):
    config_class = ModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = TranslationModel(config)

    def forward(self, src_input_ids, tgt_input_ids, Coupled=False):
        output, attention_output_1, attention_output_2 = self.model(src_input_ids, tgt_input_ids, Coupled)
        return output, attention_output_1, attention_output_2