import torch

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms.outcome import Standardize

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel


def suggest_next_points(
    X,
    y,
    strengths,
    num_candidates=1,
    maximize=False,
    trust_region_size=None,  # NEW
):
    """
    Suggest next candidate(s) using GP + qLogExpectedImprovement.

    Args:
        X (array-like): shape (n, d)
        y (array-like): shape (n,)
        strengths (array-like): shape (d,)
        num_candidates (int): number of points to suggest
        maximize (bool): maximize or minimize objective
        trust_region_size (float or None):
            None → global search
            float in (0,1] → size of trust region in normalized space

    Returns:
        candidates (torch.Tensor): shape (num_candidates, d)
    """

    # Convert to tensors
    X = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64).unsqueeze(-1)
    strengths = torch.tensor(strengths, dtype=torch.float64)

    d = X.shape[1]

    # Safety
    assert strengths.shape[0] == d, "strengths must have length d"
    strengths = torch.clamp(strengths, min=1e-6)
    strengths = strengths / strengths.mean()

    # Normalize inputs
    bounds = torch.stack([
        X.min(dim=0).values,
        X.max(dim=0).values
    ])
    X_norm = normalize(X, bounds)

    # Strength → lengthscale
    lengthscales = (1.0 / strengths).unsqueeze(0)

    # GP model
    covar_module = ScaleKernel(
        RBFKernel(ard_num_dims=d)
    )

    model = SingleTaskGP(
        X_norm,
        y,
        covar_module=covar_module,
        outcome_transform=Standardize(m=1),
    )

    model.covar_module.base_kernel.lengthscale = lengthscales.clone()

    # Fit
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    # Acquisition
    best_f = y.max() if maximize else y.min()

    acq = qLogExpectedImprovement(
        model=model,
        best_f=best_f,
    )

    # =========================
    # Trust region logic (NEW)
    # =========================
    if trust_region_size is None:
        # Global search
        bounds_norm = torch.stack([
            torch.zeros(d),
            torch.ones(d)
        ])
    else:
        # Local trust region
        assert 0 < trust_region_size <= 1, "trust_region_size must be in (0,1]"

        best_idx = torch.argmax(y) if maximize else torch.argmin(y)
        best_x = X_norm[best_idx]

        radius = trust_region_size

        lower = torch.clamp(best_x - radius, 0.0, 1.0)
        upper = torch.clamp(best_x + radius, 0.0, 1.0)

        bounds_norm = torch.stack([lower, upper])

    # Optimize
    candidates_norm, _ = optimize_acqf(
        acq,
        bounds=bounds_norm,
        q=num_candidates,
        num_restarts=10,
        raw_samples=100,
    )

    # Back to original space
    candidates = unnormalize(candidates_norm, bounds)

    return candidates


# =========================
# Main
# =========================
def main():
    X = [
        [0.1, 0.5],
        [0.3, 0.2],
        [0.8, 0.7],
    ]
    y = [1.0, 0.5, 2.0]

    strengths = [1.0, 10.0]
    num_candidates = 10

    # Try different modes:
    # trust_region_size = None     → global
    # trust_region_size = 0.2      → moderate local
    # trust_region_size = 0.05     → tight local
    trust_region_size = 1.0

    candidates = suggest_next_points(
        X,
        y,
        strengths,
        num_candidates=num_candidates,
        maximize=True,
        trust_region_size=trust_region_size,
    )

    print("Suggested candidates:")
    print(candidates)


if __name__ == "__main__":
    main()
