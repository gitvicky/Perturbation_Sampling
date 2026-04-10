
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