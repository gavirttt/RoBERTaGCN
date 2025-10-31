import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """
    Standard GCN layer as in Kipf & Welling (2016)
    Paper Equation 3: L^(i) = ρ(Ã L^(i-1) W^(i))
    """
    def __init__(self, in_dim, out_dim, dropout=0.5, activation=True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, A_sparse, X):
        # Ã X (message passing)
        H = torch.sparse.mm(A_sparse, X)
        # Linear transformation
        H = self.linear(H)
        # Activation and dropout (only if not final layer)
        if self.activation:
            return F.relu(self.drop(H))
        else:
            return self.drop(H)


class SimpleGCN(nn.Module):
    """
    Two-layer GCN as used in BertGCN paper experiments
    Paper mentions using 2-layer GCN in experiments
    
    FIXED: Final layer returns raw logits without ReLU activation
    """
    def __init__(self, in_dim, hid_dim, out_dim, dropout=0.5):
        super().__init__()
        # First layer with activation
        self.l1 = GCNLayer(in_dim, hid_dim, dropout=dropout, activation=True)
        # Second layer WITHOUT activation (for classification logits)
        self.l2 = GCNLayer(hid_dim, out_dim, dropout=dropout, activation=False)

    def forward(self, A_sparse, X):
        H = self.l1(A_sparse, X)
        H = self.l2(A_sparse, H)
        return H  # Returns logits, not activations