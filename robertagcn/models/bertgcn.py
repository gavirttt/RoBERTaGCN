import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.aux_clf = nn.Linear(feat_dim, n_classes)
        self.gcn = SimpleGCN(in_dim=feat_dim, hid_dim=gcn_hid, out_dim=n_classes, dropout=dropout)
        
        # Memory bank for document embeddings (Section 3.3)
        self.memory_bank = None
        self.memory_initialized = False
        
        # REMOVED: Cache for word features (paper uses zeros)
        # self.word_features = None
        # self.vocab_cached = None

    def initialize_memory_bank(self, ndocs, feat_dim, device):
        """Initialize memory bank with zeros - FIXED GRADIENT FLOW"""
        self.memory_bank = torch.zeros((ndocs, feat_dim), device=device, requires_grad=False)
        self.memory_initialized = True

    def update_memory_batch(self, batch_idx, batch_embeddings):
        """Update memory bank with proper gradient handling - FIXED VERSION"""
        if self.memory_bank is not None:
            with torch.no_grad():
                self.memory_bank[batch_idx] = batch_embeddings.detach()

    def get_memory_bank(self):
        """Get current memory bank"""
        return self.memory_bank

    # REMOVED: encode_and_cache_words method
    # def encode_and_cache_words(self, vocab, device, config):
    #     """Encode all words once and cache them"""
    #     ...

    def bert_forward(self, texts, device='cuda', max_len=64):
        """Forward pass through BERT encoder and auxiliary classifier"""
        feats = self.encoder.encode_batch(texts, device=device, max_len=max_len)
        logits = self.aux_clf(feats)
        return feats, logits

    def gcn_forward(self, A_sparse, X_all):
        """Forward pass through GCN"""
        logits = self.gcn(A_sparse, X_all)
        return logits

    def forward(self, idx, A_torch, vocab, texts, device, config, update_memory=True):
        texts_batch = [texts[i] for i in idx]
        
        # Get BERT embeddings for current batch WITH gradients
        feats_batch = self.encoder.encode_batch(texts_batch, device=device, max_len=config.get('max_len', 64))
        
        # Update memory bank (no gradients stored)
        if update_memory and self.memory_initialized:
            with torch.no_grad():
                self.memory_bank[idx] = feats_batch.detach()
        
        # Build feature matrix for GCN
        # CRITICAL: Use detached memory bank + fresh batch embeddings with gradients
        X_docs = self.memory_bank.detach().clone()  # Stop gradients from memory
        X_docs[idx] = feats_batch  # Allow gradients for current batch
        
        # FIXED: Word features are ZEROS (matching paper Equation 2)
        # Paper: X = [X_doc; 0] where word nodes have zero features
        X_words = torch.zeros((len(vocab), config['feat_dim']), device=device)
        
        X_full = torch.cat([X_docs, X_words], dim=0)
        
        # Get predictions
        bert_logits = self.aux_clf(feats_batch)
        gcn_logits_all = self.gcn(A_torch, X_full)
        gcn_logits_batch = gcn_logits_all[idx, :]
        
        lambda_val = config.get('lmbda', 0.7)
        
        # Convert to probabilities (OFFICIAL IMPLEMENTATION APPROACH)
        Z_gcn = F.softmax(gcn_logits_batch, dim=1)
        Z_bert = F.softmax(bert_logits, dim=1)
        
        # Interpolate probabilities with epsilon for stability
        Z_final = (Z_gcn + 1e-10) * lambda_val + Z_bert * (1 - lambda_val)
        
        # Return log probabilities for NLLLoss
        return torch.log(Z_final)