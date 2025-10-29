import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import re


def build_text_graph(texts, max_features=20000, min_df=2, window_size=20):
    """
    Build heterogeneous graph for text corpus following BertGCN paper methodology.
    
    Paper Equation 1:
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
    X_tfidf = tfidf.fit_transform(texts)  # ndoc x nword
    vocab = np.array(tfidf.get_feature_names_out())
    nwords = len(vocab)
    
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    # Step 2: Build word-word graph using sliding window PMI
    print("Building word-word graph with sliding window PMI...")
    
    # Tokenize all documents
    tokenized_docs = []
    for text in texts:
        tokens = re.findall(r'\b\w+\b', text.lower())
        tokenized_docs.append(tokens)
    
    # Build co-occurrence matrix using sliding windows
    cooc_matrix = np.zeros((nwords, nwords), dtype=np.float32)
    word_freq = np.zeros(nwords, dtype=np.float32)
    
    for tokens in tokenized_docs:
        token_indices = [word_to_idx[token] for token in tokens if token in word_to_idx]
        
        for idx in token_indices:
            word_freq[idx] += 1
        
        # Sliding window co-occurrence (paper uses window_size=20)
        doc_len = len(token_indices)
        for i in range(doc_len):
            start = max(0, i - window_size // 2)
            end = min(doc_len, i + window_size // 2 + 1)
            
            for j in range(start, end):
                if i != j:
                    idx_i = token_indices[i]
                    idx_j = token_indices[j]
                    cooc_matrix[idx_i, idx_j] += 1
    
    # Step 3: Calculate PPMI (Positive Pointwise Mutual Information)
    # Paper Equation 1: Word-word edges use PPMI
    print("Calculating PPMI values...")
    total_pairs = np.sum(cooc_matrix) / 2
    word_word_ppmi = np.zeros((nwords, nwords), dtype=np.float32)
    
    for i in range(nwords):
        for j in range(i + 1, nwords):
            if cooc_matrix[i, j] > 0:
                p_ij = cooc_matrix[i, j] / total_pairs
                p_i = word_freq[i] / np.sum(word_freq)
                p_j = word_freq[j] / np.sum(word_freq)
                
                pmi = np.log(p_ij / (p_i * p_j))
                ppmi = max(pmi, 0)  # Positive PMI
                
                if ppmi > 0:
                    word_word_ppmi[i, j] = ppmi
                    word_word_ppmi[j, i] = ppmi  # Symmetric
    
    word_word = sparse.csr_matrix(word_word_ppmi)
    
    # Step 4: Build heterogeneous adjacency matrix
    # Paper structure: [N_doc x N_doc | N_doc x N_word]
    #                  [N_word x N_doc | N_word x N_word]
    print("Constructing heterogeneous graph...")
    
    # Paper Equation 1: A_i,i = 1 (self-connections)
    # For documents: identity matrix
    doc_doc = sparse.eye(ndocs, dtype=np.float32, format='csr')
    
    # Paper Equation 1: A_i,j = TF-IDF(i,j) if i is doc, j is word
    doc_word = X_tfidf.tocsr()
    
    # Transpose for word-document edges
    word_doc = doc_word.T.tocsr()
    
    # Paper Equation 1: A_i,j = PPMI(i,j) if i,j are words and i ≠ j
    # Add self-connections for words: A_i,i = 1
    word_word_with_self = word_word + sparse.eye(nwords, dtype=np.float32, format='csr')
    
    # Build block adjacency matrix
    A = sparse.bmat([
        [doc_doc, doc_word],
        [word_doc, word_word_with_self]
    ], format='csr')
    
    # Step 5: Symmetric normalization (GCN paper: Kipf & Welling 2016)
    # Ã = D^(-1/2) A D^(-1/2)
    print("Applying symmetric normalization...")
    A = A.tocoo()
    rowsum = np.array(A.sum(1)).squeeze()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum>0)
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    D_inv_sqrt = sparse.diags(d_inv_sqrt)
    A_norm = D_inv_sqrt.dot(A).dot(D_inv_sqrt).tocsr()
    
    print(f"Graph built: {A_norm.shape[0]} nodes ({ndocs} docs + {nwords} words)")
    print(f"Graph density: {A_norm.nnz / (A_norm.shape[0] * A_norm.shape[1]):.6f}")
    print(f"Average degree: {A_norm.nnz / A_norm.shape[0]:.2f}\n===================================")
    
    return A_norm, vocab.tolist(), X_tfidf.tocsr()