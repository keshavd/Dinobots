from dataclasses import dataclass
import torch
from typing import Optional


@dataclass
class NodeClassificationOutput:
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[dict] = None
