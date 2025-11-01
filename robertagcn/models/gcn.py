import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Standard GCN layer as in Kipf & Welling (2016)
    Paper Equation 3: L^(i) = ρ(Ã L^(i-1) W^(i))
    """
    def __init__(self, in_dim, out_dim, dropout=0.5, activation=True, use_dropout=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = dropout
        self.use_dropout = use_dropout
        self.activation = activation

    def forward(self, A_sparse, X):
        # Ã X (message passing)
        H = torch.sparse.mm(A_sparse, X)
        # Linear transformation
        H = self.linear(H)
        # Activation and dropout (only if not final layer)
        if self.activation:
            H = F.relu(H)
        if self.use_dropout:
            H = F.dropout(H, p=self.dropout, training=self.training)
        return H


class SimpleGCN(nn.Module):
    """
    Two-layer GCN as used in BertGCN paper experiments
    Paper mentions using 2-layer GCN in experiments
    
    FIXED: Final layer has no dropout (standard practice)
    """
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.5):
        super().__init__()
        # First layer with activation and dropout
        self.l1 = GCNLayer(in_dim, hid_dim, dropout=dropout, activation=True, use_dropout=True)
        # Second layer WITHOUT activation or dropout (for classification logits)
        self.l2 = GCNLayer(hid_dim, out_dim, dropout=dropout, activation=False, use_dropout=False)

    def forward(self, A_sparse, X):
        H = self.l1(A_sparse, X)
        H = self.l2(A_sparse, H)
        return H  # Returns logits, not activations