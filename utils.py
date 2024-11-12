def normalize_rows(P):
    return P / P.sum(axis=1, keepdims=True)
