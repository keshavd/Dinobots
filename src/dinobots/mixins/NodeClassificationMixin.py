from torch.nn import Module, Linear


class NodeClassificationMixin(Module):
    def forward(self, x, edge_index, y=None, **kwargs):
        x = self.model(x, edge_index, **kwargs)
        node_embeddings = self.classification_head(x)
        return node_embeddings
