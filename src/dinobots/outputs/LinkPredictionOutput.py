from dataclasses import dataclass
import torch
from typing import Optional


@dataclass
class LinkPredictionOutput:
    x_embedding: Optional[torch.FloatTensor] = None
    logits: Optional[dict] = None
    loss: Optional[torch.FloatTensor] = None