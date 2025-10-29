import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class RobertaTagalogEncoder(nn.Module):
    def __init__(self, model_name='jcblaise/roberta-tagalog-base', proj_dim=768):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        hidden = self.model.config.hidden_size
        if hidden != proj_dim:
            self.proj = nn.Linear(hidden, proj_dim)
        else:
            self.proj = nn.Identity()

    def encode_batch(self, texts, device='cuda', max_len=64):
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        for k in enc:
            enc[k] = enc[k].to(device)
        out = self.model(**enc, return_dict=True)
        cls = out.last_hidden_state[:, 0, :]
        return self.proj(cls)
