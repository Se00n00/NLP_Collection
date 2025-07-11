from transformers import PreTrainedModel
from wikitext_modelcofig import WikiText_ModelConfig
import sys
import os

# Add Language/ as parent Directory
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../..')))

from src.models.GenerativeModel import GenerativeModel

class Wikitext_Model(PreTrainedModel):
    config_class = WikiText_ModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.generative_model = GenerativeModel(config)

    def forward(self, input_ids):
        output, attention_output = self.generative_model(input_ids)
        return output, attention_output
