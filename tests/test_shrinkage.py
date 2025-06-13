import pytest
import torch

from bergson.math import optimal_linear_shrinkage


@pytest.mark.parametrize(
    "p,n",
    [
        # Test the n < p case
        (32, 16),
        (64, 32),
        (128, 64),
        # And the n > p case
        (16, 64),
        (32, 128),
        (64, 256),
    ],
)
def test_olse_shrinkage(p: int, n: int):
    # Use CUDA to speed things up if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    prng = torch.Generator(device=device).manual_seed(42)

    # Number of population matrices to test
    N = 100

    # Generate a random covariance matrix
    A = torch.randn(N, 1, p, p, device=device, generator=prng)
    S_true = A @ A.mH / p
    torch.linalg.diagonal(S_true).add_(1e-3)

    # Generate random means
    mu = torch.randn(N, 1, 1, p, device=device, generator=prng)

    # Number of draws of empirical samples per matrix
    m = 100

    # Generate random Gaussian vectors with this covariance matrix
    scale_tril = torch.linalg.cholesky(S_true)
    X = torch.randn(N, m, n, p, device=device, generator=prng) @ scale_tril.mH + mu

    # Compute the sample covariance matrix
    mu_hat = X.mean(dim=-2, keepdim=True)
    X_centered = X - mu_hat
    S_hat = (X_centered.mH @ X_centered) / n

    # Apply shrinkage
    S_olse = optimal_linear_shrinkage(S_hat, n)

    # Compute the average squared Frobenius error across the different draws
    norm_naive = torch.square(S_hat - S_true).sum(dim=(-1, -2)).mean(1)
    norm_olse = torch.square(S_olse - S_true).sum(dim=(-1, -2)).mean(1)

    # The MSE should be lower for all the population matrices
    assert torch.all(norm_olse <= norm_naive)

    # Compute the sample second moment matrix
    S_hat = S_hat + mu_hat.mT @ mu_hat
    S_olse = S_olse + mu_hat.mT @ mu_hat
    S_true = A + mu.mT @ mu

    # Compute the average squared Frobenius error across the different draws
    norm_naive = torch.square(S_hat - S_true).sum(dim=(-1, -2)).mean(1)
    norm_olse = torch.square(S_olse - S_true).sum(dim=(-1, -2)).mean(1)

    # The MSE should be lower for all the population matrices
    assert torch.all(norm_olse <= norm_naive)
