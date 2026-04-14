
***

# Advanced Sampling Methods for Physics-Informed Boundary Inversion

## Overview
This repository/paper explores methodologies for inverting physics-informed uncertainty bounds (Conformal Prediction bounds) from the residual space back into the physical state-space. Standard Monte Carlo rejection sampling suffers from the "curse of dimensionality" in high-resolution spatiotemporal PDEs, leading to near-zero acceptance rates. 

To resolve this, we present three advanced sampling methodologies designed to efficiently navigate the high-dimensional, highly-constrained physical manifold defined by a physics operator $\mathcal{D}$, a surrogate prediction $\hat{u}$, and a calibrated residual bound $\hat{q}$.

---

## Method 1: Posterior / Manifold Sampling (MCMC)

**Concept:** Rather than blindly sampling from a prior distribution and rejecting failures, this method frames the problem as sampling from a target posterior distribution. By relaxing the strict conformal boundaries into a smooth probability density, we can use gradient-guided algorithms like the Unadjusted Langevin Algorithm (ULA) or Hamiltonian Monte Carlo (HMC) to "walk" along the high-probability manifold of valid physical states.

**Mathematical Formulation:**
We define a Prior distribution over the physical perturbations, $p(\eta)$ (e.g., a Gaussian Process). The Conformal Bound constraint is formulated as a Likelihood function, $p(R \mid \eta)$, where $R = \mathcal{D}(\hat{u} + \eta)$. 

The target Posterior distribution of valid perturbations is:
$$p(\eta \mid R_{\text{bound}}) \propto p(\eta) \cdot p(R \mid \eta)$$

To sample from this posterior, we use Langevin dynamics, which utilizes the gradient of the log-posterior to guide the random walk into the valid bounded regions:
$$\eta^{(k+1)} = \eta^{(k)} + \frac{\epsilon}{2} \nabla_{\eta} \log p(\eta^{(k)} \mid R_{\text{bound}}) + \sqrt{\epsilon} z^{(k)}$$
where $\epsilon$ is the step size and $z^{(k)} \sim \mathcal{N}(0, I)$ is injected thermal noise to ensure proper statistical sampling of the entire valid volume, rather than collapsing to a single point.

---

## Method 2: Differentiable Rejection (Inference-Time Optimization)

**Concept:**
This method replaces the statistical rejection step with an explicit optimization loop at inference time. It treats the binary acceptance mask as a differentiable loss landscape. If a sampled noise field fails the residual check, we backpropagate the error through the frozen physics operator $\mathcal{D}$ and directly update the noise field via gradient descent until it strictly satisfies the bound constraint.

**Mathematical Formulation:**
We define a continuous penalty function that is exactly zero when the residual is within the conformal bound $\hat{q}$, and grows smoothly when violated. 

The Boundary Loss is defined as:
$$\mathcal{L}_{\text{bound}}(\eta) = \frac{1}{N} \sum_{t,x} \max\left(0, \left| \mathcal{D}(\hat{u} + \eta)_{t,x} \right| - \hat{q}_{t,x}\right)^2$$

To prevent the optimizer from exploiting the physics operator and generating non-physical, high-frequency "adversarial" noise, we regularize the update with a Prior Loss (e.g., the Mahalanobis distance for a Gaussian prior with covariance $\Sigma$):
$$\mathcal{L}_{\text{prior}}(\eta) = \eta^T \Sigma^{-1} \eta$$

The total objective optimized for *each rejected sample* is:
$$\mathcal{L}_{\text{total}}(\eta) = \lambda_{\text{bound}} \mathcal{L}_{\text{bound}}(\eta) + \lambda_{\text{prior}} \mathcal{L}_{\text{prior}}(\eta)$$

**Update Rule:**
$$\eta^{(k+1)} = \eta^{(k)} - \alpha \nabla_{\eta} \mathcal{L}_{\text{total}}(\eta^{(k)})$$
*(Iterated until $\mathcal{L}_{\text{bound}} = 0$)*

---

## Method 3: Generative Modeling (Training a Generator)

**Concept:**
While Differentiable Rejection is highly effective, running gradient descent for every sample at inference time is computationally expensive. This method shifts the computational burden to a training phase by training a secondary, lightweight Generative Model (e.g., a Normalizing Flow, VAE, or Guided Diffusion model) to map standard Gaussian noise directly onto the valid physical manifold.

**Mathematical Formulation:**
Let $G_{\phi}(z)$ be a neural network parameterized by weights $\phi$, taking standard normal noise $z \sim \mathcal{N}(0, I)$ as input and outputting a physical perturbation $\eta$.

The neural surrogate ($\hat{u}$) and physics operator ($\mathcal{D}$) remain entirely frozen. We train the generator weights $\phi$ to minimize the expected constraint violation over the standard normal distribution:
$$\min_{\phi} \mathbb{E}_{z \sim \mathcal{N}(0,I)} \left[ \mathcal{L}_{\text{total}}(G_{\phi}(z)) \right]$$

where $\mathcal{L}_{\text{total}}$ is the same composite boundary and prior loss defined in Method 2:
$$\mathcal{L}_{\text{total}}(\eta) = \lambda_{\text{bound}} \mathcal{L}_{\text{bound}}(G_{\phi}(z)) + \lambda_{\text{prior}} \mathcal{L}_{\text{prior}}(G_{\phi}(z))$$

**Inference:**
Once trained, generating valid physical bounds requires zero rejection and zero optimization. We simply sample $z \sim \mathcal{N}(0, I)$ and compute the valid perturbation via a single forward pass: $\eta = G_{\phi}(z)$.

---

## Method Comparison

| Feature | Posterior Sampling (MCMC) | Differentiable Rejection | Generative Modeling |
| :--- | :--- | :--- | :--- |
| **Paradigm** | Statistical / Bayesian | Optimization | Deep Learning |
| **Acceptance Rate** | High (controlled by step size) | $100\%$ (Guaranteed by loop) | $100\%$ (By construction) |
| **Inference Cost** | High (many gradient evaluations) | Medium (optimization per sample) | **Low** (single forward pass) |
| **Training Cost** | None | None | **High** (requires training phase) |
| **Best For** | Rigorous probabilistic UQ | Out-of-the-box accuracy on frozen models | Fast, real-time bounding required at inference |

---

## Design Note: Coverage Guarantee vs. Residual Containment

A key design point is that all three advanced methods (MCMC, Optimisation, Generative) optimise for **100% residual containment per sample** — not directly for `1 - alpha` coverage. This is intentional, not a bug.

The `1 - alpha` probabilistic coverage guarantee is provided entirely by **conformal prediction during the calibration step**, which produces a `qhat` such that:

$$P\bigl(|\mathcal{D}(\hat{u})| \leq \hat{q}\bigr) \geq 1 - \alpha$$

The role of perturbation sampling is then to answer a *deterministic* question: given this calibrated `qhat`, what is the **feasible set** of all fields `u` satisfying `|D(u)| <= qhat` everywhere? The physical-space bounds are the envelope of this set.

A sample that only *partially* satisfies the residual constraint does not belong to the feasible set, so the hard accept/reject gate (`|residual| <= qhat` at every point) is applied after MCMC refinement and after optimisation rescue. The advanced methods simply make it far more likely that samples land inside this set — they do not relax the set itself.

In summary:

- **`qhat`** encodes the `1 - alpha` coverage (from CP calibration).
- **Sampling methods** characterise the feasible set defined by that `qhat` (requiring 100% residual containment per accepted sample).
- **Coverage** is validated empirically across held-out trajectories, confirming that the envelope achieves `>= 1 - alpha` coverage as guaranteed by conformal prediction theory.

---

## Design Note: Optimisation Space and Noise Priors

### What is actually optimised?

All three gradient-based methods (MCMC, Optimisation, Generative) update the perturbation field $\eta$. A subtle but important detail is **which parameterisation** of $\eta$ the gradients act on.

Currently, regardless of the noise generation strategy, the optimised variables are always the **raw per-grid-point values** $\eta_i$ for $i = 0, \dots, N-1$. The code path is:

```
1. noise = noise_generator.{spatial|white|gp|bspline}(...)   # [batch, N]
2. noise = noise.clone().detach().requires_grad_(True)         # optimise this tensor
3. Langevin / Adam updates noise at every grid point
```

The noise generators are called **once** to produce an initialisation. After that, the correlation structure is not enforced — MCMC and Optimisation operate on $N$ independent degrees of freedom.

### Per noise type

| Noise type | Generation process | Latent dimension | Optimised variables |
| :--- | :--- | :--- | :--- |
| `white` | $\eta_i \sim \mathcal{N}(0, \sigma^2)$ independently | $N$ | $\eta_i$ (grid values) |
| `spatial` | $\eta = \text{conv1d}(\xi, G_L)$ where $\xi$ is white noise and $G_L$ is a Gaussian kernel | $N$ | $\eta_i$ (grid values) |
| `gp` | $\eta = L z$ where $L$ is the Cholesky factor of the kernel matrix $K$ and $z \sim \mathcal{N}(0, I)$ | $N$ | $\eta_i$ (grid values) |
| `bspline` | $\eta = \Phi c$ where $\Phi \in \mathbb{R}^{N \times K}$ is the B-spline basis and $c \sim \mathcal{N}(0, \sigma^2 I)$ | $K \ll N$ | $\eta_i$ (grid values) |

In every case, the noise type affects the **initialisation only**. The gradient updates act on the full $N$-dimensional grid-point representation, not on the latent parameters that generated it.

### Consequence: loss of structure during optimisation

After even a few gradient steps, the perturbation **no longer respects the original correlation structure**:

- **Spatial**: the Gaussian kernel correlation $C(r) \propto \exp(-r^2 / 2L^2)$ is not preserved.
- **GP**: the exact covariance from the kernel matrix is broken.
- **B-spline**: the perturbation leaves the $K$-dimensional spline subspace and is no longer $C^2$-continuous.

The only regularisation currently applied is the prior loss $\lambda_{\text{prior}} \cdot \text{mean}(\eta^2)$, which penalises magnitude uniformly across all grid points but does not enforce any spatial structure.

### Potential improvement: latent-space optimisation

To preserve the noise structure throughout optimisation, one could optimise in the **latent space** of each generator rather than in grid space. This would:

1. **Maintain physical smoothness** — perturbations stay within the generator's function class.
2. **Reduce dimensionality** — particularly for B-splines ($K$ coefficients instead of $N$ grid values).
3. **Provide a more informative prior** — the prior loss on latent variables directly penalises deviation from the generative model.

Concretely, for each noise type:

**B-spline** — optimise the $K$ control-point coefficients $c$ instead of the $N$ grid values:
$$c^{(k+1)} = c^{(k)} - \frac{\epsilon}{2} \nabla_c \mathcal{L}(c^{(k)}) + \sqrt{\epsilon}\,\sigma\, z^{(k)}, \quad \eta = \Phi c$$

Every iterate $\eta = \Phi c$ is guaranteed to be a valid cubic spline ($C^2$-continuous). The optimisation dimension drops from $N$ to $K$ (e.g. 100 → 16).

**GP** — optimise in the whitened space $z$ before the Cholesky transform:
$$z^{(k+1)} = z^{(k)} - \frac{\epsilon}{2} \nabla_z \mathcal{L}(z^{(k)}) + \sqrt{\epsilon}\,\sigma\, w^{(k)}, \quad \eta = L z$$

Every iterate $\eta = Lz$ has exactly the GP covariance structure. The prior loss on $z$ becomes $\|z\|^2$, which is the natural GP log-prior.

**Spatial** — optimise the pre-convolution white noise $\xi$:
$$\xi^{(k+1)} = \xi^{(k)} - \frac{\epsilon}{2} \nabla_\xi \mathcal{L}(\xi^{(k)}) + \sqrt{\epsilon}\,\sigma\, w^{(k)}, \quad \eta = \text{conv1d}(\xi, G_L)$$

Every iterate maintains the Gaussian kernel correlation structure.

This latent-space approach is not yet implemented but would be a natural extension, particularly for scaling to high-dimensional PDE fields where preserving physical smoothness and reducing the optimisation dimension become critical.