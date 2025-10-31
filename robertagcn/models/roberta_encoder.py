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
        """Encodes a batch of texts (documents or words) using BERT."""
        enc = self.tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        for k in enc:
            enc[k] = enc[k].to(device)
        out = self.model(**enc, return_dict=True)
        cls = out.last_hidden_state[:, 0, :]
        return self.proj(cls)

    def encode_words(self, words, device='cuda', max_len=16, batch_size=64):
        """Encodes a list of words using BERT, handling batches."""
        word_embeddings = []
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i+batch_size]
            # For single words, truncation might not be as critical, but max_len=16 is a reasonable default
            enc = self.tokenizer(batch_words, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
            for k in enc:
                enc[k] = enc[k].to(device)
            out = self.model(**enc, return_dict=True)
            cls = out.last_hidden_state[:, 0, :]
            word_embeddings.append(self.proj(cls).cpu())
        return torch.cat(word_embeddings, dim=0).to(device)
