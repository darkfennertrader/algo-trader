# from torch.linalg import eigh
# import torch


# def pca_factors(returns: torch.Tensor, K: int, vol_scale: bool = True):
#     """
#     returns: (T, N)
#     K: number of factors to keep
#     """
#     # 1) Demean each asset (across time)
#     mean = returns.mean(dim=0, keepdim=True)  # (1, N)
#     R = returns - mean

#     # 2) Optionally vol-scale to unit variance per asset
#     if vol_scale:
#         std = R.std(dim=0, unbiased=True, keepdim=True)  # (1, N)
#         std = torch.clamp(std, min=1e-6)
#         R = R / std

#     # 3) Covariance: Σ = (1/(T-1)) Rᵀ R, shape (N, N)
#     cov = (R.T @ R) / (R.shape[0] - 1.0)

#     # 4) Eigendecomposition: cov = V diag(vals) Vᵀ
#     vals, vecs = eigh(cov)  # vals ascending

#     # 5) Sort descending
#     idx = torch.argsort(vals, descending=True)
#     vals = vals[idx]
#     vecs = vecs[:, idx]

#     # 6) Take top-K eigenvectors as loadings
#     B_pca = vecs[:, :K]  # (N, K)

#     # 7) Factor returns: F = R * B
#     F_pca = R @ B_pca  # (T, K)

#     # 8) Explained variance
#     explained = vals / vals.sum()
#     cum_explained = torch.cumsum(explained, dim=0)

#     return B_pca, F_pca, vals, explained, cum_explained
