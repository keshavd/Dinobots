from torch.nn import Module, BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dinobots.outputs.LinkPredictionOutput import LinkPredictionOutput


class LinkPredictionMixin(Module):

    def forward(self, x, edge_index, y=None, **kwargs):
        x = self.model(x, edge_index, **kwargs)
        node_embeddings = self.classification_head(x)
        output = LinkPredictionOutput()
        output.x_embedding = node_embeddings
        # Logits are calculated as the Dot product of A and B
        A = edge_index[0]
        B = edge_index[1]
        logits = (node_embeddings[A] * node_embeddings[B]).sum(-1)
        output.logits = logits  # Logit per edge
        if y is not None:
            loss_fct = CrossEntropyLoss(ignore_index=self.ignore_index)
            output.loss = loss_fct(logits.view(-1, self.num_labels), y.view(-1))
        return output
