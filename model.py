import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F


class BertBaseClassifier(nn.Module):
    def __init__(
        self,
        bertModel,
        num_classes,
        input_dim=768,
        hidden_dim=1024,
        dropout=0.1,
    ):
        super(BertBaseClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(bertModel)

        for param in self.bert.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        cls_token = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_token)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        else:
            return {"logits": logits}
