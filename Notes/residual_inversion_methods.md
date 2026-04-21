# Advanced Methods for Inverting Physics-Informed Residual Bounds

## 1. Introduction and Problem Formulation

In high-resolution, physics-informed machine learning, models output a surrogate prediction $\hat{u}$ for a PDE. Using Conformal Prediction (CP), we can calibrate a residual bound $\hat{q}$ such that the true physical state $u$ satisfies the governing PDE operator $\mathcal{D}$ within this bound:
$$ P\bigl(|\mathcal{D}(u)| \le \hat{q}\bigr) \ge 1 - \alpha $$
The inverse problem is to sample perturbations $\eta$ such that the modified physical state $u = \hat{u} + \eta$ strictly lies in the **feasible set**:
$$ \mathcal{F} = \left\{ \eta : |\mathcal{D}(\hat{u} + \eta)| \le \hat{q} \right\} $$
Due to the "curse of dimensionality", standard Monte Carlo rejection sampling yields near-zero acceptance rates. Below, we review four advanced methodologies to solve this, analyze their strengths and weaknesses, and finally propose a synthesized theoretical solution for complex, higher-order non-linear PDEs like the Navier-Stokes equations.

---

## 2. Method 1: Posterior / Manifold Sampling (Langevin Dynamics)

### Mathematical Structure
Rather than blind rejection, we treat the residual bound constraint as a likelihood $p(R \mid \eta)$ and the physical constraints as a prior $p(\eta)$. We sample from the posterior $p(\eta \mid R_{\text{bound}})$ using the Unadjusted Langevin Algorithm (ULA). The gradient of the log-posterior guides the random walk into the valid bounded regions:
$$ \eta^{(k+1)} = \eta^{(k)} + \frac{\epsilon}{2} \nabla_{\eta} \log p(\eta^{(k)} \mid R_{\text{bound}}) + \sqrt{\epsilon} z^{(k)} $$
where $z^{(k)} \sim \mathcal{N}(0, I)$ is thermal noise.

### Code Implementation
```python
import torch

def langevin_step(eta, operator, u_hat, q_hat, step_size, prior_inv_cov):
    eta.requires_grad_(True)
    
    # 1. Boundary Log-Likelihood (Penalty)
    residual = torch.abs(operator(u_hat + eta))
    loss_bound = torch.mean(torch.relu(residual - q_hat)**2)
    
    # 2. Prior Log-Likelihood
    loss_prior = torch.einsum('b...i,ij,b...j->b...', eta, prior_inv_cov, eta).mean()
    
    # Total Energy (Negative Log-Posterior)
    energy = loss_bound + loss_prior
    grad_eta = torch.autograd.grad(energy, eta)[0]
    
    # Langevin Update
    noise = torch.randn_like(eta)
    with torch.no_grad():
        eta_new = eta - 0.5 * step_size * grad_eta + torch.sqrt(torch.tensor(step_size)) * noise
    return eta_new
```

### Advantages & Disadvantages
*   **Pros:** Statistically rigorous; makes no structural assumptions about the feasible set $\mathcal{F}$ (non-parametric); can handle multi-modal distributions.
*   **Cons:** High inference cost due to many gradient evaluations per sample.

### Citations
*   Roberts, G. O., & Tweedie, R. L. (1996). *Exponential convergence of Langevin distributions and their discrete approximations*. Bernoulli.
*   Cotter, S. L., et al. (2013). *MCMC methods for functions: modifying old algorithms to make them faster*. Statistical Science.

---

## 3. Method 2: Differentiable Rejection (Inference-Time Optimization)

### Mathematical Structure
This method replaces stochastic sampling with an explicit optimization loop. We define a continuous penalty function $\mathcal{L}_{\text{bound}}$ that is exactly zero when the residual is within the conformal bound $\hat{q}$.
$$ \mathcal{L}_{\text{total}}(\eta) = \lambda_{\text{bound}} \mathcal{L}_{\text{bound}}(\eta) + \lambda_{\text{prior}} \mathcal{L}_{\text{prior}}(\eta) $$
where $\mathcal{L}_{\text{bound}}(\eta) = \frac{1}{N} \sum \max\left(0, |\mathcal{D}(\hat{u} + \eta)| - \hat{q}\right)^2$. Gradient descent is iterated until $\mathcal{L}_{\text{bound}} = 0$.

### Code Implementation
```python
import torch

def optimize_perturbation(eta_init, operator, u_hat, q_hat, lr=1e-2, max_iters=100):
    eta = eta_init.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([eta], lr=lr)
    
    for _ in range(max_iters):
        optimizer.zero_grad()
        residual = torch.abs(operator(u_hat + eta))
        violation = torch.relu(residual - q_hat)
        
        if violation.max() == 0:
            break # 100% contained
            
        loss_bound = torch.mean(violation**2)
        loss_prior = torch.mean(eta**2) # Simple isotropic prior
        loss = loss_bound + 0.1 * loss_prior
        
        loss.backward()
        optimizer.step()
        
    return eta.detach()
```

### Advantages & Disadvantages
*   **Pros:** Guarantees 100% residual containment for the drawn sample if optimization converges; highly robust out-of-the-box.
*   **Cons:** Optimization can destroy the underlying spatial correlation of the initial noise prior; still requires inference-time gradient steps.

### Citations
*   Tarantola, A. (2005). *Inverse Problem Theory and Methods for Model Parameter Estimation*. SIAM.

---

## 4. Method 3: Generative Modeling (Amortized Generator)

### Mathematical Structure
We shift the computational burden to a training phase by learning a parameterized map (e.g., Normalizing Flow or continuous implicit network) $G_\phi(z)$ from a base Gaussian distribution to the valid physical manifold.
$$ \min_{\phi} \mathbb{E}_{z \sim \mathcal{N}(0,I)} \left[ \lambda_{\text{bound}} \mathcal{L}_{\text{bound}}(G_{\phi}(z)) + \lambda_{\text{prior}} \mathcal{L}_{\text{prior}}(G_{\phi}(z)) \right] $$

### Code Implementation
```python
import torch
import torch.nn as nn

class PerturbationGenerator(nn.Module):
    def __init__(self, latent_dim, spatial_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, spatial_dim)
        )
        
    def forward(self, z):
        return self.net(z)

# Training loop sketch
# optimizer = Adam(generator.parameters())
# for z in dataloader:
#    eta = generator(z)
#    loss = compute_boundary_loss(eta, operator, u_hat, q_hat)
#    loss.backward(); optimizer.step()
```

### Advantages & Disadvantages
*   **Pros:** **Extremely fast at inference** (single forward pass); can learn complex topological manifolds if trained well.
*   **Cons:** High upfront training cost; susceptible to mode collapse; out-of-distribution generalisation is difficult; no strict guarantee of boundary satisfaction without a fallback mechanism.

### Citations
*   Jacobsen, J.-H., et al. (2023). *CoCoGen: Physically-Consistent and Conditioned Score-based Generative Models for Forward and Inverse Problems*. 
*   Song, Y., et al. (2020). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR.

---

## 5. Method 4: Variational Inference (Per-Trajectory Posterior)

### Mathematical Structure
Instead of amortizing across all predictions, Variational Inference (VI) fits a parametric Gaussian posterior $q_\phi(z) = \mathcal{N}(\mu, \Sigma)$ in the latent space of a noise prior *for each trajectory independently*. It maximizes the Evidence Lower Bound (ELBO):
$$ \mathrm{ELBO}(\phi) = \mathbb{E}_{z \sim q_{\phi}}\!\bigl[-\mathcal{L}_{\text{bound}}(\text{decode}(z))\bigr] - \mathrm{KL}\!\bigl(q_{\phi} \parallel \mathcal{N}(0, I)\bigr) $$

### Code Implementation
```python
import torch

class MeanFieldPosterior(torch.nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.zeros(latent_dim))
        self.log_sigma = torch.nn.Parameter(torch.zeros(latent_dim))
        
    def sample(self, num_samples):
        sigma = torch.exp(self.log_sigma)
        eps = torch.randn(num_samples, len(self.mu), device=self.mu.device)
        return self.mu + eps * sigma
        
    def kl_divergence(self):
        sigma2 = torch.exp(2 * self.log_sigma)
        return 0.5 * torch.sum(self.mu**2 + sigma2 - 1.0 - 2*self.log_sigma)

# Optimization loop optimizes the VI parameters (mu, log_sigma) 
# instead of the latent noise itself.
```

### Advantages & Disadvantages
*   **Pros:** Balances expressivity and speed; generates an entire valid distribution (smooth envelopes) parameterized compactly (mean-field or low-rank); latent space structure preserves spatial continuity guarantees.
*   **Cons:** Bounded by the expressiveness of the variational family (Gaussian); struggles if the feasible set $\mathcal{F}$ is highly non-convex or disconnected.

### Citations
*   Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). *Variational Inference: A Review for Statisticians*. JASA.
*   Kingma, D. P., & Welling, M. (2013). *Auto-Encoding Variational Bayes*. ICLR.

---

## 6. Theoretical Proposal: Inverting Residual Bounds for Multivariate Nonlinear PDEs (Navier-Stokes)

### The Challenge
Higher-order, nonlinear, multivariate PDEs such as the Navier-Stokes (NS) equations present extreme difficulties for conformal bound inversion:
1.  **High Dimensionality and Non-Gaussianity:** Fluid turbulence is highly structured and non-Gaussian. Standard Gaussian priors (`WhitePrior`, `SpatialPrior`) fail to capture the underlying physics manifold.
2.  **Differentiability and $C^k$ Continuity:** NS requires computation of 2nd order spatial derivatives. Spline priors provide basic smoothness but lack the semantic physics structure of turbulence.
3.  **Non-convex Feasible Sets:** The non-linear convective term $(u \cdot \nabla) u$ causes the feasible set $\mathcal{F}$ to be highly non-convex.

### Proposed Method: Score-Guided Latent Variational Optimization (SGL-VO)

We propose combining **Score-Based Generative Priors** with **Latent Variational Inference** and a final **Differentiable Rejection Projection**.

**Step 1: Learned Physics-Informed Score Prior (Pre-training)**
Instead of using analytical priors (GP or B-Splines), train a Score-Based Generative Model (SGM) $s_\theta(u, t) \approx \nabla_u \log p_t(u)$ on a dataset of turbulent PDE solutions. 
*Why?* The score network perfectly captures the highly non-Gaussian spatial correlations of turbulence (vortices, eddies).

**Step 2: Score-Guided Variational Inference (Inference Optimization)**
For a specific prediction $\hat{u}$ and conformal bound $\hat{q}$, we apply VI not in a standard normal latent space, but via Score-based Data Assimilation (e.g., DiLO or SMDP frameworks). We parameterize a distribution $q_\phi(z_0)$ over the diffusion latent space. The ELBO is modified such that the KL divergence is measured against the physical SGM prior:
$$ \mathrm{ELBO}(\phi) = \mathbb{E}_{u \sim \text{SDE\_Solve}(q_\phi)}\!\bigl[-\lambda_{\text{bound}} \mathcal{L}_{\text{bound}}(u - \hat{u})\bigr] + \text{Prior Score Matching Term} $$
Because the score network enforces $C^\infty$ physical consistency implicitly, the generated fields are extremely likely to naturally satisfy the NS operator constraints.

**Step 3: Projected Differentiable Rejection (Guaranteeing 100% Containment)**
Because VI only bounds the residual *in expectation*, we draw samples $u_i \sim q_\phi$, and for those that slightly violate $\hat{q}$, we apply Differentiable Rejection (Method 2). 
However, to prevent the Adam optimizer from generating high-frequency artifacts that destroy the turbulence structures, we **project the gradient updates onto the score network's manifold**:
$$ \eta^{(k+1)} = \eta^{(k)} - \alpha \nabla_{\eta} \mathcal{L}_{\text{bound}}(\eta^{(k)}) + \beta s_\theta(\hat{u} + \eta^{(k)}) $$
The gradient of $\mathcal{L}_{\text{bound}}$ strictly enforces the constraint, while the score guidance term $s_\theta$ pulls the perturbation back into the physically viable manifold, preserving fluid dynamics continuity.

### Theoretical Advantages for Navier-Stokes
*   **Physics-Aware Continuity:** The score guidance replaces naive smoothness priors, natively understanding multi-scale turbulence structures.
*   **Computational Tractability:** By operating in the latent space of the diffusion model (VI), the dimensionality is compressed, mitigating the massive cost of optimizing a full 3D spatiotemporal grid.
*   **Strict Conformal Validity:** The final projected differentiable rejection step ensures the rigorous theoretical `1 - alpha` coverage guarantee derived from CP calibration is upheld deterministically for every sample in the envelope.

### Supporting Citations for the Proposal
*   Rozet, F., & Louppe, G. (2023). *Score-based Data Assimilation*. NeurIPS.
*   Hong, S., et al. (2024). *A Score-based Generative Solver for PDE-constrained Inverse Problems with Complex Priors*.
*   Yang, X., & Sommer, S. (2023). *FluidDiff: A diffusion model based surrogate for the Navier-Stokes equations*. SIAM.

---

## 7. Physics-Informed (PDE-Dependent) Priors

A fundamental weakness of standard methods is their reliance on "physics-blind" priors (e.g., standard Gaussian Processes, B-splines, or white noise). These priors generate generic smooth fields, and the optimization/sampling algorithms must do the heavy lifting to forcefully mold that generic noise into a physics-compliant shape. 

By explicitly incorporating the PDE operator into the prior, we can generate perturbations that natively live on the physical manifold, drastically reducing the required optimization steps and improving the physical consistency of the sampled envelope.

### 7.1 The Linearized SPDE Prior (Tangent Space Prior)

Instead of drawing $\eta$ from a generic Gaussian process, we define the prior implicitly via a Stochastic Partial Differential Equation (SPDE) centered around the surrogate prediction $\hat{u}$.

For a nonlinear PDE operator $\mathcal{D}$, we compute its Fréchet derivative (Jacobian) $\mathcal{J}_{\mathcal{D}}$ linearized at $\hat{u}$. The perturbation prior is defined as the solution to:
$$ \mathcal{J}_{\mathcal{D}}(\hat{u}) \eta = \xi $$
where $\xi \sim \mathcal{N}(0, I)$ is basic spatial white noise.

**Mechanism:** 
Mathematically, $\eta = \mathcal{J}_{\mathcal{D}}(\hat{u})^{-1} \xi$. The inverse operator $\mathcal{J}_{\mathcal{D}}(\hat{u})^{-1}$ acts as the Green's function of the local PDE. By pushing generic noise *through* the linearized physics operator, the resulting perturbation $\eta$ automatically inherits the dispersion relations, wave speeds, and fundamental physics (e.g., divergence-free conditions in fluids) of the governing equations. 

**Application to Navier-Stokes:**
If $\mathcal{D}$ is the Navier-Stokes operator, $\mathcal{J}_{\mathcal{D}}$ corresponds to the Oseen equations (linearized Navier-Stokes). An SPDE prior would inherently generate valid turbulent structures (eddies advecting with the base flow $\hat{u}$) rather than non-physical Gaussian blobs.

### 7.2 Adjoint-Mapped Perturbations

Computing the forward inverse operator $\mathcal{J}_{\mathcal{D}}^{-1}$ is often computationally prohibitive for large grids. A cheaper alternative is to use the **Adjoint Operator** $\mathcal{J}^*_{\mathcal{D}}(\hat{u})$.

We generate a latent noise field $z$ and map it to the physical perturbation space using the adjoint:
$$ \eta = \mathcal{J}^*_{\mathcal{D}}(\hat{u}) z $$

**Mechanism:**
The adjoint operator maps from the residual (error) space back into the physical state space. By constructing $\eta$ via the adjoint, we restrict our sampling exclusively to the "active" directions that demonstrably affect the residual. This ensures zero computational effort is wasted generating noise in the null space of the PDE (i.e., perturbations that change the state but do not affect the physical residual).

### 7.3 Residual-Conditioned Heteroskedastic Prior

We can use the magnitude of the initial residual $R_0 = \mathcal{D}(\hat{u})$ to dynamically shape the variance of our prior across the spatial domain. The conformal constraint dictates:
$$ -\hat{q} \le R_0 + \mathcal{J}_{\mathcal{D}}(\hat{u})\eta \approx \mathcal{D}(\hat{u} + \eta) \le \hat{q} $$

We construct a prior covariance matrix $\Sigma(x)$ that varies spatially based on how close the base prediction is to the conformal boundary:
$$ p(\eta) = \mathcal{N}(0, \Sigma(R_0, \hat{q})) $$
where the local variance is defined as:
$$ \Sigma_{ii} \propto \text{ReLU}(\hat{q}_i - |R_0|_i) $$

**Mechanism:**
*   In regions where the model prediction is poor ($|R_0(x)| \approx \hat{q}(x)$), the allowable variance of $\eta$ is squeezed to near-zero. This prevents the sampling algorithm from accidentally pushing a borderline valid state out of bounds.
*   In regions where the prediction is perfect ($R_0(x) \approx 0$), the prior variance is maximized, allowing the sampler to aggressively explore the envelope of valid solutions.

### Code Implementation (Conceptual SPDE Prior)
```python
import torch

def generate_spde_prior_sample(u_hat, operator, base_noise):
    """
    Conceptual implementation of a Tangent Space (SPDE) Prior.
    Requires a differentiable physics operator to compute the Jacobian-vector product.
    """
    u_hat = u_hat.clone().requires_grad_(True)
    
    # Forward pass to get base residual
    R0 = operator(u_hat)
    
    # We want to solve J * eta = base_noise. 
    # In practice, this requires an iterative solver (e.g., GMRES or CG) 
    # using Jacobian-Vector Products (JVPs) provided by PyTorch autograd.
    
    def jvp(v):
        # Computes J * v
        return torch.autograd.grad(R0, u_hat, grad_outputs=v, retain_graph=True)[0]
    
    # eta = iterative_linear_solve(A_func=jvp, b=base_noise)
    # return eta
    pass
```

### 7.4 Generalization to Complex Coupled Systems (MHD & Plasma Physics)

A key advantage of physics-informed prior methods (specifically the **Linearized SPDE Prior** and **Adjoint-Mapped Perturbations**) is their mathematical agnosticism to the specific PDE. By interacting with the PDE solely through its Fréchet derivative (Jacobian) $\mathcal{J}_{\mathcal{D}}$ or its adjoint $\mathcal{J}^*_{\mathcal{D}}$, the methods inherently adapt to the underlying physics.

#### Magnetohydrodynamics (MHD)
MHD couples fluid dynamics (Navier-Stokes) with electromagnetism (Maxwell's equations) across a multivariate state vector $u = [\mathbf{v}, \mathbf{B}, p, \rho]$.
*   **The Problem with Standard Priors:** Independent perturbations of $\mathbf{v}$ and $\mathbf{B}$ using GPs will violate the divergence-free constraint ($\nabla \cdot \mathbf{B} = 0$), creating unphysical magnetic monopoles and massive residual spikes.
*   **The Physics-Informed Solution:** The Jacobian $\mathcal{J}_{\text{MHD}}$ contains the linearized cross-coupling terms (e.g., the Lorentz force and magnetic induction). Solving $\mathcal{J}_{\text{MHD}} \eta = \xi$ generates a *jointly coupled* perturbation. The inverse operator intrinsically enforces $\nabla \cdot (\delta\mathbf{B}) = 0$, naturally generating Alfvén waves rather than isotropic noise.

#### Hasegawa-Wakatani (HW) Equations
The HW equations model electrostatic drift-wave turbulence in magnetized edge plasmas (crucial for tokamak fusion reactors), coupling plasma density fluctuation $n$ and electrostatic potential $\phi$.
*   **The Problem with Standard Priors:** HW turbulence is highly anisotropic, generating "zonal flows" and sheared eddies. Standard spatial GPs generate isotropic, circular noise blobs that require massive optimization effort to stretch into zonal flows.
*   **The Physics-Informed Solution:** Linearizing the HW operator around a base state $\hat{u}$ containing a density gradient yields primary eigenmodes corresponding to drift-wave instabilities. Pushing white noise $\xi$ through $\mathcal{J}_{\text{HW}}^{-1}$ naturally amplifies noise along these eigenmodes, spontaneously generating anisotropic, elongated drift-wave structures.

#### Computational Challenges in Practice
While mathematically seamless, generalizing to systems like MHD or HW introduces severe computational challenges:
1.  **Ill-Conditioned Jacobians:** Plasmas or MHD models often exhibit widely separated time/length scales (e.g., fast Alfvén waves vs. slow resistive diffusion), leading to "stiff" Jacobians. Iterative solvers (like GMRES) for the SPDE prior $\mathcal{J} \eta = \xi$ will require robust preconditioners to converge.
2.  **Memory Costs of the Adjoint:** While PyTorch's reverse-mode autodiff (`torch.autograd.grad(..., grad_outputs=z)`) provides the discrete adjoint "for free", it requires storing the entire forward computation graph of the PDE solver in GPU memory, which scales poorly for large 3D spatiotemporal grids.
3.  **Boundary Conditions:** Complex PDEs feature intricate boundary conditions (e.g., perfectly conducting walls in MHD). The Jacobian-Vector Product (JVP) must strictly enforce these linearized boundaries, which can be challenging to implement robustly in PyTorch if the boundary enforcement isn't perfectly differentiable in the forward pass.

---

## 8. First-Principles Derivation of the Linearized SPDE Prior

To show that $\mathcal{J}_{\mathcal{D}}(\hat{u}) \eta = \xi$ is the natural Gaussian prior induced by a linearized PDE operator, we derive it from first principles.

### 8.1. Linearization and the SPDE
We start with a deterministic PDE constraint $\mathcal{D}(u) = 0$. Given a surrogate solution $\hat{u}$, we model uncertainty as $u = \hat{u} + \eta$.
Using a first-order expansion:
$$ \mathcal{D}(\hat{u} + \eta) \approx \mathcal{D}(\hat{u}) + \mathcal{J}_{\mathcal{D}}(\hat{u})\eta $$
Let $R_0 = \mathcal{D}(\hat{u})$ (the residual) and $L = \mathcal{J}_{\mathcal{D}}(\hat{u})$ (the linear operator).
To introduce uncertainty, we assume the linearized PDE residual behaves like random noise:
$$ L\eta = \xi $$
where $\xi \sim \mathcal{N}(0, I)$ is spatial white noise. This defines a stochastic PDE (SPDE).

### 8.2. Green's Functions and Induced Gaussian Process
Let $G(x,y)$ be the Green's function of $L$, satisfying $LG(x,y) = \delta(x-y)$. The solution to $L\eta = \xi$ is:
$$ \eta(x) = \int G(x,y)\,\xi(y)\,dy $$
Since $\xi$ is Gaussian and the mapping is linear, $\eta$ is also a Gaussian process with zero mean.
Its covariance is $\text{Cov}(\eta) = L^{-1}(L^{-1})^*$. Thus, the induced prior is:
$$ \eta \sim \mathcal{N}\bigl(0, (L^*L)^{-1}\bigr) $$
This SPDE prior is the Gaussian measure whose covariance is the inverse of the PDE energy operator.

---

## 9. Connection to Preconditioned Langevin Dynamics

Standard Unadjusted Langevin Algorithm (ULA) can be stiff because $L^*L$ has a large spectrum due to the multi-scale nature of PDEs.
Instead, we can use a **Preconditioned Langevin** approach.

### 9.1. The Preconditioner
We use $M = (L^*L)^{-1}$ as the preconditioner, which exactly matches the covariance of the SPDE prior. The energy functional for the sampling objective combines the constraint and the prior:
$$ E(\eta) = \mathcal{L}_{\text{bound}}(\eta) + \frac{1}{2}\langle \eta, L^*L\eta \rangle $$

### 9.2. Preconditioned Update
Applying the preconditioner $M$ to the gradient of $E$ gives the preconditioned Langevin update:
$$ \eta^{(k+1)} = \eta^{(k)} - \frac{\epsilon}{2} \left[ (L^*L)^{-1} \nabla_{\eta} \mathcal{L}_{\text{bound}}(\eta^{(k)}) + \eta^{(k)} \right] + \sqrt{\epsilon} (L^*L)^{-1/2} z^{(k)} $$
where $z^{(k)} \sim \mathcal{N}(0, I)$.

### 9.3. Interpretation
*   **Prior Term:** The prior shrinkage becomes isotropic, perfectly conditioning the prior.
*   **Physics-Filtered Gradient:** The constraint gradients are projected through the PDE inverse ($(L^*L)^{-1}$), meaning updates respect the PDE structure and avoid high-frequency artifacts.
*   **Structured Noise:** The noise injected is "physics-shaped" ($(L^*L)^{-1/2} z$), directly sampling the SPDE prior natively.

---

## 10. PDE-Specific Mappings

The SPDE prior and preconditioned Langevin approach apply differently depending on the nature of the PDE:

*   **Burgers' Equation:** The linearized operator generates perturbations that are advected and smoothed. It works excellently for smooth/laminar flows but requires regularization or hybrid methods in shock-dominated regimes where the Jacobian becomes ill-conditioned.
*   **Wave Equation:** The operator $\partial_{tt} - c^2\Delta$ generates traveling waves. It correctly respects finite propagation speed, but the purely hyperbolic nature lacks diffusion, meaning preconditioned Langevin might suffer from persistent oscillations and requires light damping.
*   **Navier–Stokes (Incompressible):** The Oseen operator naturally generates divergence-free turbulence structures and transported eddies. While perfect for laminar flows, highly turbulent (chaotic) regimes make the linearization only locally valid, necessitating a combination with Score-Based Priors.
*   **Ideal MHD:** The SPDE prior brilliantly preserves cross-field physics (e.g., generating coupled Alfvén waves and enforcing divergence-free magnetic fields natively). However, it is computationally heavy due to extreme stiffness and requires robust preconditioning.

---

## 11. End-to-End Workflow Summary

The overall workflow transforms the inverse problem from "sample generically then enforce physics" into "sample directly within the physics-induced geometry".

1.  **Start with Surrogate & Bound:** Given surrogate $\hat{u}$, PDE operator $\mathcal{D}$, and conformal bound $\hat{q}$.
2.  **Linearize Physics Operator:** Compute $L = \mathcal{J}_{\mathcal{D}}(\hat{u})$ using Jacobian-vector products (autodiff), without explicitly forming the matrix.
3.  **Define SPDE Prior:** Conceptually use $\eta \sim \mathcal{N}(0, (L^*L)^{-1})$ so perturbations inherently follow PDE structure.
4.  **Set Up Objective:** Define the energy functional $E(\eta)$ consisting of the constraint violation penalty $\mathcal{L}_{\text{bound}}$ and the SPDE prior.
5.  **Preconditioned Langevin Sampling:** Iteratively compute the constraint gradient, solve $(L^*L)x = \nabla_{\eta}\mathcal{L}_{\text{bound}}$ via Conjugate Gradient (CG), generate structured noise via $(L^*L)^{-1/2}z$, and update $\eta$.
6.  **Feasibility Check:** Verify if $|\mathcal{D}(\hat{u} + \eta)| \le \hat{q}$. If needed, apply a final differentiable projection.
7.  **Re-linearize (Optional):** For highly nonlinear systems, periodically update $\hat{u} = \hat{u} + \eta$ and recompute $L$.
8.  **Generate Envelope:** The chain of generated samples forms a valid conformal envelope that maintains physical consistency.