from transformers import PreTrainedModel
from ModelConfig import ModelConfig

import sys
import os

# Add Language/ as parent Directory
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))

from src.models.Classification import ClassificationModel

class Model(PreTrainedModel):
    config_class = ModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = ClassificationModel(config)

    def forward(self, input_ids, mask=None):
        output, attention_output = self.model(input_ids, mask=mask)
        return output, attention_output