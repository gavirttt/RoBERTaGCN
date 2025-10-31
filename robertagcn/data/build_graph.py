import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy import sparse
import re
from collections import defaultdict
import itertools


def build_text_graph(texts, max_features=20000, min_df=2, window_size=20):
    """
    OPTIMIZED: Build heterogeneous graph using sparse operations
    
    Build heterogeneous graph for text corpus following BertGCN paper methodology,
    with additional social interaction edges.
    
    Paper Equation 1 (Extended):
    A_i,j = {
        PPMI(i,j),      if i,j are words and i ≠ j
        TF-IDF(i,j),    if i is document, j is word
        1,              if i = j
        0,              otherwise
    }
    
    Args:
        texts: List of document strings
        max_features: Maximum vocabulary size
        min_df: Minimum document frequency for words
        window_size: Sliding window size for PMI calculation (paper: 20)
    
    Returns:
        A_norm: Normalized adjacency matrix (sparse CSR)
        vocab: List of vocabulary words
        X_tfidf: Document-word TF-IDF matrix (ndoc x nword)
    """
    ndocs = len(texts)
    
    # Step 1: Build vocabulary and TF-IDF matrix
    print("Building TF-IDF matrix...")
    tfidf = TfidfVectorizer(max_features=max_features, min_df=min_df,
                            token_pattern=r"(?u)\b\w+\b", lowercase=True)
    X_tfidf = tfidf.fit_transform(texts)
    vocab = np.array(tfidf.get_feature_names_out())
    nwords = len(vocab)
    
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Step 2: OPTIMIZED - Build word-word graph using sparse co-occurrence
    print("Building word-word graph with optimized PMI calculation...")
    
    # Tokenize all documents
    tokenized_docs = []
    for text in texts:
        tokens = re.findall(r'\b\w+\b', text.lower())
        # Filter tokens that are in vocabulary
        token_indices = [word_to_idx[token] for token in tokens if token in word_to_idx]
        tokenized_docs.append(token_indices)
    
    # Build co-occurrence matrix using sparse operations
    cooc_data = defaultdict(float)
    word_freq = np.zeros(nwords, dtype=np.float32)
    
    for doc_idx, tokens in enumerate(tokenized_docs):
        # Count word frequencies
        for token_idx in tokens:
            word_freq[token_idx] += 1
        
        # Sliding window co-occurrence - optimized
        doc_len = len(tokens)
        for i in range(doc_len):
            start = max(0, i - window_size // 2)
            end = min(doc_len, i + window_size // 2 + 1)
            
            for j in range(start, end):
                if i != j:
                    idx_i, idx_j = tokens[i], tokens[j]
                    if idx_i < idx_j:
                        cooc_data[(idx_i, idx_j)] += 1
                    else:
                        cooc_data[(idx_j, idx_i)] += 1
    
    # Convert to sparse matrix
    rows, cols, data = [], [], []
    for (i, j), count in cooc_data.items():
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([count, count])
    
    cooc_matrix = sparse.csr_matrix((data, (rows, cols)), shape=(nwords, nwords))
    
    # Step 3: OPTIMIZED - Calculate PPMI using sparse operations
    print("Calculating PPMI values (optimized)...")
    total_pairs = cooc_matrix.sum() / 2
    word_word_ppmi = cooc_matrix.copy().astype(np.float32)
    
    # Convert to COO for efficient iteration
    cooc_coo = cooc_matrix.tocoo()
    
    # Calculate PPMI
    new_data = []
    for i, j, count in zip(cooc_coo.row, cooc_coo.col, cooc_coo.data):
        if i < j and count > 0:  # Only process upper triangle
            p_ij = count / total_pairs
            p_i = word_freq[i] / word_freq.sum()
            p_j = word_freq[j] / word_freq.sum()
            
            pmi = np.log(p_ij / (p_i * p_j)) if (p_i * p_j) > 0 else 0
            ppmi = max(pmi, 0)
            
            new_data.append(ppmi)
        elif i == j:
            new_data.append(0.0)  # No self-loops in PPMI
    
    # Create symmetric PPMI matrix
    word_word_ppmi = sparse.coo_matrix(
        (new_data * 2, (list(cooc_coo.row) + list(cooc_coo.col), 
                        list(cooc_coo.col) + list(cooc_coo.row))),
        shape=(nwords, nwords)
    ).tocsr()
    
    # Add self-connections
    word_word_with_self = word_word_ppmi + sparse.eye(nwords, dtype=np.float32, format='csr')
    
    # Step 4: Build heterogeneous adjacency matrix
    print("Constructing heterogeneous graph...")
    
    # Document-document: identity matrix
    doc_doc = sparse.eye(ndocs, dtype=np.float32, format='csr')
    
    # Document-word edges (TF-IDF)
    doc_word = X_tfidf.tocsr()
    word_doc = doc_word.T.tocsr()
    
    # Build block adjacency matrix
    A = sparse.bmat([
        [doc_doc, doc_word],
        [word_doc, word_word_with_self]
    ], format='csr')
    
    # Step 5: Symmetric normalization
    print("Applying symmetric normalization...")
    A = eliminate_zeros(A.tocoo())
    rowsum = np.array(A.sum(1)).squeeze()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum>0)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    A_norm = D_inv_sqrt.dot(A).dot(D_inv_sqrt).tocsr()
    
    # Validation
    assert not np.isnan(A_norm.data).any(), "Graph contains NaN values"
    assert not np.isinf(A_norm.data).any(), "Graph contains inf values"
    
    print(f"Graph built: {A_norm.shape[0]} nodes ({ndocs} docs + {nwords} words)")
    print(f"Graph density: {A_norm.nnz / (A_norm.shape[0] * A_norm.shape[1]):.6f}")
    print(f"Average degree: {A_norm.nnz / A_norm.shape[0]:.2f}")
    print("="*70)
    
    return A_norm, vocab.tolist(), X_tfidf.tocsr()


def eliminate_zeros(coo_matrix):
    """Remove explicit zeros from COO matrix"""
    mask = coo_matrix.data != 0
    return sparse.coo_matrix(
        (coo_matrix.data[mask], (coo_matrix.row[mask], coo_matrix.col[mask])),
        shape=coo_matrix.shape
    )