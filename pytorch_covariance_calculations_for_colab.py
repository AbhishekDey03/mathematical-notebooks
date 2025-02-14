import torch
import math
import time

# Select device: GPU if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def find_correlation_matrix(image_size, sigma, device):
    """
    Constructs a Gaussian correlation matrix.
    
    Args:
        image_size (int): The image dimension (assumed square).
        sigma (float): Standard deviation of the target distribution.
        device (torch.device): Device to perform computations on.
        
    Returns:
        torch.Tensor: A (image_size**2 x image_size**2) correlation matrix.
    """
    # Create grid coordinates (as floats)
    x = torch.arange(image_size, device=device, dtype=torch.float32)
    y = torch.arange(image_size, device=device, dtype=torch.float32)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    # Flatten and stack the coordinates: shape (n, 2) where n=image_size^2
    pixel_coords = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
    i = pixel_coords[:, 0]
    j = pixel_coords[:, 1]
    
    # Compute pairwise differences
    di = i.unsqueeze(1) - i.unsqueeze(0)  # shape (n, n)
    dj = j.unsqueeze(1) - j.unsqueeze(0)
    d = 1.8 * torch.sqrt(di**2 + dj**2)
    
    # Compute the Gaussian kernel
    # Use math.sqrt to compute the constant because sigma is a float.
    const = 1 / math.sqrt(2 * math.pi * sigma**2)
    C = const * torch.exp(-d**2 / (2 * sigma**2))
    
    # Set the diagonal elements to 1
    n = C.shape[0]
    C[torch.arange(n), torch.arange(n)] = 1.0
    return C


def negative_log_likelihood_full(x, mu, cov):
    """
    Computes the negative log likelihood (NLL) for a Gaussian distribution 
    using the full covariance matrix.
    
    Args:
        x (torch.Tensor): Data sample (vector).
        mu (torch.Tensor): Mean vector.
        cov (torch.Tensor): Covariance matrix.
    
    Returns:
        torch.Tensor: The computed negative log likelihood.
    """
    diff = x - mu
    n = diff.numel()
    sign, logdet = torch.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix is not positive definite!")
    inv_cov = torch.linalg.inv(cov)
    quad_term = diff @ (inv_cov @ diff)
    const_term = n * torch.log(torch.tensor(2 * math.pi, device=diff.device, dtype=diff.dtype))
    nll = 0.5 * (logdet + quad_term + const_term)
    return nll


def negative_log_likelihood_svd(x, mu, cov):
    """
    Computes the negative log likelihood using an SVD-based formulation.
    
    Args:
        x (torch.Tensor): Data sample (vector).
        mu (torch.Tensor): Mean vector.
        cov (torch.Tensor): Covariance matrix.
        
    Returns:
        torch.Tensor: The computed negative log likelihood.
    """
    U, s, Vh = torch.linalg.svd(cov)
    diff = x - mu
    n = diff.numel()
    logdet = torch.sum(torch.log(s))
    # Transform the difference vector into the SVD basis.
    z = Q.T @ diff
    quad_term = torch.sum((z**2) / s)
    const_term = n * torch.log(torch.tensor(2 * math.pi, device=diff.device, dtype=diff.dtype))
    nll = 0.5 * (logdet + quad_term + const_term)
    return nll


def negative_log_likelihood_cholesky(x, mu, cov):
    """
    Computes the negative log likelihood using the Cholesky decomposition.
    
    Args:
        x (torch.Tensor): Data sample.
        mu (torch.Tensor): Mean vector.
        cov (torch.Tensor): Covariance matrix.
    
    Returns:
        torch.Tensor: The computed negative log likelihood.
    """
    L = torch.linalg.cholesky(cov)  # cov = L @ L.T
    diff = (x - mu).unsqueeze(1)  # Convert to column vector: shape (n, 1)
    # Solve L * y = diff via forward substitution.
    y = torch.triangular_solve(diff, L, upper=False)[0].squeeze()
    quad_term = torch.dot(y, y)
    log_det = 2 * torch.sum(torch.log(torch.diag(L)))
    n = x.numel()
    const_term = n * torch.log(torch.tensor(2 * math.pi, device=x.device, dtype=x.dtype))
    nll = 0.5 * (log_det + quad_term + const_term)
    return nll


# ----- Main computation -----

image_size = 150
sigma = 1.0  # You can adjust sigma as needed.
cov = find_correlation_matrix(image_size, sigma, device)
n = image_size ** 2  # Dimensionality

# Create random data and mean vectors on the selected device.
x = torch.randn(n, device=device)
mu = torch.randn(n, device=device)

# Compute negative log likelihood (NLL) using the full covariance method.
nll_full = negative_log_likelihood_full(x, mu, cov)

nll_svd = negative_log_likelihood_svd(x, mu, cov)

nll_chol = negative_log_likelihood_cholesky(x, mu, cov)

print("Negative Log Likelihood (Cholesky):", nll_chol.item())

print("\nReprinting for clarity:")
print("Full covariance NLL:     ", nll_full.item())
print("SVD formulation NLL:     ", nll_svd.item())
print("Cholesky NLL:            ", nll_chol.item())
print("Difference (Full - Chol):", abs(nll_full.item() - nll_chol.item()))


# ----- Timing Comparisons -----

num_iter = 100

# Warm-up (to ensure GPU kernels are loaded)
_ = negative_log_likelihood_full(x, mu, cov)
_ = negative_log_likelihood_svd(x, mu, cov)
_ = negative_log_likelihood_cholesky(x, mu, cov)
torch.cuda.synchronize() if device.type == "cuda" else None

# Time the full covariance method.
start_full = time.perf_counter()
for _ in range(num_iter):
    nll_full = negative_log_likelihood_full(x, mu, cov)
    if device.type == "cuda":
        torch.cuda.synchronize()
time_full = (time.perf_counter() - start_full) / num_iter

# Time the SVD method.
start_svd = time.perf_counter()
for _ in range(num_iter):
    nll_svd = negative_log_likelihood_svd(x, mu, U, s)
    if device.type == "cuda":
        torch.cuda.synchronize()
time_svd = (time.perf_counter() - start_svd) / num_iter

# Time the Cholesky method.
start_chol = time.perf_counter()
for _ in range(num_iter):
    nll_chol = negative_log_likelihood_cholesky(x, mu, cov)
    if device.type == "cuda":
        torch.cuda.synchronize()
time_chol = (time.perf_counter() - start_chol) / num_iter

print("\nAverage time per call:")
print("Full covariance:     {:.2e} seconds".format(time_full))
print("SVD formulation:     {:.2e} seconds".format(time_svd))
print("Cholesky approach:   {:.2e} seconds".format(time_chol))

times = {"Full": time_full, "SVD": time_svd, "Cholesky": time_chol}
fastest = min(times, key=times.get)
print("Fastest method:", fastest)
