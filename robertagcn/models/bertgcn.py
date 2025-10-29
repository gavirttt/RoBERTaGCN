import torch
import torch.nn as nn
from .roberta_encoder import RobertaTagalogEncoder
from .gcn import SimpleGCN


class BertGCN(nn.Module):
    """
    BertGCN model combining BERT and GCN for transductive text classification
    
    Paper: "BertGCN: Transductive Text Classification by Combining GCN and BERT"
    Lin et al., ACL 2021 Findings
    
    Architecture:
    - BERT encoder for document embeddings
    - Auxiliary classifier on BERT embeddings (Z_BERT)
    - GCN over heterogeneous graph (Z_GCN)
    - Final prediction: Z = λZ_GCN + (1-λ)Z_BERT (Equation 6)
    """
    def __init__(self, encoder_name='jcblaise/roberta-tagalog-base', 
                 feat_dim=768, gcn_hid=256, n_classes=2, dropout=0.5):
        super().__init__()
        self.encoder = RobertaTagalogEncoder(model_name=encoder_name, proj_dim=feat_dim)
        # Paper Equation 5: Z_BERT = softmax(WX)
        self.aux_clf = nn.Linear(feat_dim, n_classes)
        # Paper Equation 4: Z_GCN = softmax(g(X,A)) where g is GCN
        self.gcn = SimpleGCN(in_dim=feat_dim, hid_dim=gcn_hid, out_dim=n_classes, dropout=dropout)

    def bert_forward(self, texts, device='cuda', max_len=64):
        """Forward pass through BERT encoder and auxiliary classifier"""
        feats = self.encoder.encode_batch(texts, device=device, max_len=max_len)
        logits = self.aux_clf(feats)
        return feats, logits

    def gcn_forward(self, A_sparse, X_all):
        """
        Forward pass through GCN
        
        Args:
            A_sparse: Normalized adjacency matrix (N_total x N_total)
            X_all: Node features (N_total x feat_dim), includes both docs and words
        
        Returns:
            logits: (N_total x n_classes)
        """
        logits = self.gcn(A_sparse, X_all)
        return logits