from torch.nn import Module, BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dinobots.outputs.LinkPredictionOutput import LinkPredictionOutput


class LinkPredictionMixin(Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, edge_index, y=None, **kwargs):
        x = self.model(x, edge_index, **kwargs)
        node_embeddings = self.classification_head(x)
        output = LinkPredictionOutput()
        output.output_x = node_embeddings
        logits = (node_embeddings[edge_index[0]] * node_embeddings[edge_index[1]]).sum(
            -1
        )  # dot product
        output.logits = logits  # Logit per link
        if y is not None:
            loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)
            output.loss = loss_fct(logits.view(-1, self.num_labels), y.view(-1))
        return output
