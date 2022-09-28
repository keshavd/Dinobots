from torch.nn import Module, Linear
from dinobots.models.geometric_based.RevGNN import RevGNN
from dinobots.mixins.LinkPredictionMixin import LinkPredictionMixin


class RevGNNForLinkPrediction(LinkPredictionMixin, Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        num_groups=2,
        ignore_index=None,
        num_labels=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_groups = num_groups
        self.model = RevGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            num_groups=num_groups,
        )
        self.classification_head = Linear(hidden_channels, out_channels)
        self.ignore_index = ignore_index
        self.num_labels = num_labels

    def reset_parameters(self):
        self.model.reset_parameters()
        self.classification_head.reset_parameters()
