import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import re


def build_text_graph(texts, max_features=20000, min_df=2, window_size=20, 
                     social_edges=None, social_weight=1.0):
    """
    Build heterogeneous graph for text corpus following BertGCN paper methodology,
    with additional social interaction edges.
    
    Paper Equation 1 (Extended):
    A_i,j = {
        PPMI(i,j),      if i,j are words and i ≠ j
        TF-IDF(i,j),    if i is document, j is word
        social_weight,  if i,j are documents with social connection
        1,              if i = j
        0,              otherwise
    }
    
    Args:
        texts: List of document strings
        max_features: Maximum vocabulary size
        min_df: Minimum document frequency for words
        window_size: Sliding window size for PMI calculation (paper: 20)
        social_edges: Dict with social connection info:
            {
                'authors': List of author usernames (one per document)
                'reply_to': List of reply-to usernames (None if not a reply)
                'doc_ids': List of document IDs for mapping
            }
        social_weight: Weight for social edges (default: 1.0)
    
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
    
    # Step 3.5: Build social edges if provided
    doc_doc_social = sparse.csr_matrix((ndocs, ndocs), dtype=np.float32)
    
    if social_edges is not None:
        print("Building social interaction edges...")
        authors = social_edges.get('authors', [])
        reply_to = social_edges.get('reply_to', [])
        doc_ids = social_edges.get('doc_ids', list(range(ndocs)))
        
        # Build author -> document indices mapping
        author_to_docs = {}
        for doc_idx, author in enumerate(authors):
            if author and str(author).lower() not in ['', 'nan', 'none']:
                if author not in author_to_docs:
                    author_to_docs[author] = []
                author_to_docs[author].append(doc_idx)
        
        # Create doc_id -> index mapping
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        
        # Track edge statistics
        author_edges = 0
        reply_edges = 0
        
        # Build sparse matrix components
        row_indices = []
        col_indices = []
        values = []
        
        # 1. Author edges: Connect documents by the same author
        for author, doc_indices in author_to_docs.items():
            if len(doc_indices) > 1:
                # Connect all documents by this author
                for i in range(len(doc_indices)):
                    for j in range(i + 1, len(doc_indices)):
                        idx_i = doc_indices[i]
                        idx_j = doc_indices[j]
                        row_indices.extend([idx_i, idx_j])
                        col_indices.extend([idx_j, idx_i])
                        values.extend([social_weight, social_weight])
                        author_edges += 1
        
        # 2. Reply edges: Connect reply to original tweet
        for doc_idx, reply_to_user in enumerate(reply_to):
            if reply_to_user and str(reply_to_user).lower() not in ['', 'nan', 'none']:
                # Find documents by the replied-to author
                if reply_to_user in author_to_docs:
                    for target_idx in author_to_docs[reply_to_user]:
                        if target_idx != doc_idx:  # Don't self-connect
                            row_indices.extend([doc_idx, target_idx])
                            col_indices.extend([target_idx, doc_idx])
                            values.extend([social_weight, social_weight])
                            reply_edges += 1
                            break  # Connect to first matching author's tweet
        
        # Build sparse social edge matrix
        if row_indices:
            doc_doc_social = sparse.csr_matrix(
                (values, (row_indices, col_indices)), 
                shape=(ndocs, ndocs),
                dtype=np.float32
            )
        
        print(f"Social edges added: {author_edges} author connections, {reply_edges} reply connections")
        print(f"Total social edges: {doc_doc_social.nnz}")
    
    # Step 4: Build heterogeneous adjacency matrix
    print("Constructing heterogeneous graph...")
    
    # For documents: identity matrix + social edges
    doc_doc = sparse.eye(ndocs, dtype=np.float32, format='csr') + doc_doc_social
    
    # Document-word edges (TF-IDF)
    doc_word = X_tfidf.tocsr()
    
    # Transpose for word-document edges
    word_doc = doc_word.T.tocsr()
    
    # Word-word edges (PPMI) with self-connections
    word_word_with_self = word_word + sparse.eye(nwords, dtype=np.float32, format='csr')
    
    # Build block adjacency matrix
    A = sparse.bmat([
        [doc_doc, doc_word],
        [word_doc, word_word_with_self]
    ], format='csr')
    
    # Step 5: Symmetric normalization (GCN paper: Kipf & Welling 2016)
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