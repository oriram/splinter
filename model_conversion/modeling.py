import torch
from torch.nn import Module, Linear, LayerNorm
from torch.nn.parameter import Parameter

from transformers import BertPreTrainedModel, BertModel
from transformers.modeling_bert import BertLMPredictionHead


class FullyConnectedLayer(Module):
    def __init__(self, input_dim, output_dim):
        super(FullyConnectedLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dense = Linear(self.input_dim, self.output_dim)
        self.LayerNorm = LayerNorm(self.output_dim)

    def forward(self, inputs):
        return inputs


class QuestionAwareSpanSelectionHead(Module):
    def __init__(self, config):
        super().__init__()

        self.query_start_transform = FullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.query_end_transform = FullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.start_transform = FullyConnectedLayer(config.hidden_size, config.hidden_size)
        self.end_transform = FullyConnectedLayer(config.hidden_size, config.hidden_size)

        self.start_classifier = Parameter(torch.Tensor(config.hidden_size, config.hidden_size))
        self.end_classifier = Parameter(torch.Tensor(config.hidden_size, config.hidden_size))

    def forward(self, inputs):
        return inputs


class ClassificationHead(Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.span_predictions = QuestionAwareSpanSelectionHead(config)

    def forward(self, inputs):
        return inputs


class ModelWithQASS(BertPreTrainedModel):
    """
    Normalize start and end separately
    """

    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.cls = ClassificationHead(config)

        self.init_weights()

    def forward(self, inputs):
        return inputs
