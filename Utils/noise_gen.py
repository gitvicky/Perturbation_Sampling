# %% 
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class PDENoiseGenerator:
    """
    A comprehensive noise generator for PDE simulations in PyTorch.
    
    This class provides various methods for generating different types of noise
    that are commonly needed in stochastic PDE simulations, including white noise,
    colored noise, spatially/temporally correlated noise, and boundary-compatible noise.
    
    The implementation is GPU-friendly and handles proper scaling for numerical
    discretizations.
    """
    
    def __init__(self, device='cpu', dtype=torch.float32):
        """
        Initialize the noise generator.
        
        Args:
            device: str or torch.device, computation device ('cpu', 'cuda', etc.)
            dtype: torch.dtype, data type for generated tensors
        """
        self.device = device
        self.dtype = dtype
    
    def white_noise(self, shape, std=1.0, seed=None):
        """
        Generate basic white noise (independent Gaussian samples).
        
        White noise has:
        - Zero mean: E[ξ(x)] = 0
        - Delta-correlated: E[ξ(x)ξ(y)] = σ²δ(x-y)
        - Gaussian distribution at each point
        
        This is the fundamental building block for other noise types.
        Used in stochastic PDEs like: du/dt = F(u) + σξ(x,t)
        
        Args:
            shape: tuple, shape of the noise tensor 
                   Can be (H, W) for 2D spatial noise
                   or (batch, channels, H, W) for batched operations
            std: float, standard deviation of the noise (σ in mathematical notation)
            seed: int, random seed for reproducibility in testing/debugging
            
        Returns:
            torch.Tensor: White noise tensor with specified shape and statistics
            
        Notes:
            - Each grid point is independent
            - Satisfies central limit theorem for large ensembles
            - Memory efficient - no correlations to store
            - Fundamental ingredient for more complex noise types
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        return torch.randn(shape, device=self.device, dtype=self.dtype) * std
    
    def colored_noise_spectral(self, shape, alpha=0.0, std=1.0):
        """
        Generate colored noise using spectral filtering method.
        
        Colored noise has power spectral density S(k) ∝ k^(-α), where:
        - α = 0: White noise (flat spectrum)
        - α = 1: Pink noise (1/f noise, common in nature)
        - α = 2: Brown noise (Brownian motion, integrated white noise)
        
        Method: Generate white noise in Fourier space, then filter by frequency
        This preserves Gaussian statistics while introducing spatial correlations.
        
        Physical applications:
        - Pink noise: Models natural fluctuations (weather, turbulence)
        - Brown noise: Models diffusion processes, accumulated random walks
        
        Args:
            shape: tuple, spatial dimensions (H, W) only
            alpha: float, spectral exponent controlling correlation structure
                   Higher alpha = more low-frequency content = smoother noise
            std: float, overall noise amplitude after normalization
            
        Returns:
            torch.Tensor: Spatially correlated noise with specified spectral properties
            
        Notes:
            - Requires FFT operations - more expensive than white noise
            - Naturally periodic due to FFT (good for periodic boundary conditions)
            - Amplitude is renormalized to maintain desired std deviation
            - Higher alpha values create smoother, more correlated fields
        """
        H, W = shape
        
        # Create frequency grids for 2D FFT
        # fftfreq gives frequencies in cycles per sample, centered at 0
        freqs_h = torch.fft.fftfreq(H, device=self.device)
        freqs_w = torch.fft.fftfreq(W, device=self.device)
        freq_grid_h, freq_grid_w = torch.meshgrid(freqs_h, freqs_w, indexing='ij')
        
        # Compute radial frequency: |k| = sqrt(kx² + ky²)
        # This gives the magnitude of the wave vector
        freq_radial = torch.sqrt(freq_grid_h**2 + freq_grid_w**2)
        freq_radial[0, 0] = 1e-10  # Avoid division by zero at DC component
        
        # Generate white noise in Fourier space
        # Complex noise ensures proper statistics after inverse FFT
        noise_fft = torch.randn(H, W, device=self.device, dtype=torch.complex64)
        
        # Apply spectral shaping: multiply by k^(-α/2) to get S(k) ∝ k^(-α)
        # Factor of 1/2 because power spectrum involves |F(k)|²
        if alpha != 0:
            noise_fft = noise_fft / (freq_radial**(alpha/2))
        
        # Transform back to spatial domain
        # Real part gives spatially correlated Gaussian field
        noise = torch.fft.ifft2(noise_fft).real
        
        # Normalize to desired standard deviation
        # FFT operations can change the amplitude, so we renormalize
        noise = noise * std / torch.std(noise)
        
        return noise
    
    def spatially_correlated_noise(self, shape, correlation_length=5.0, std=1.0):
        """
        Generate spatially correlated noise using Gaussian kernel convolution.
        
        This method creates noise with exponential spatial correlation:
        C(r) ∝ exp(-r²/(2L²)) where L is the correlation length
        
        Method: Convolve white noise with a Gaussian kernel
        - Preserves Gaussian statistics (convolution of Gaussians is Gaussian)
        - Creates isotropic (rotation-invariant) correlations
        - Correlation length controls the smoothness scale
        
        Physical interpretation:
        - Models local averaging effects in physical systems
        - Represents finite-size effects of measurement instruments
        - Mimics diffusion-limited processes
        
        Args:
            shape: tuple, spatial dimensions (H, W)
            correlation_length: float, spatial correlation length in grid units
                              Controls how far correlations extend
                              Larger values = smoother, more correlated noise
            std: float, noise amplitude after normalization
            
        Returns:
            torch.Tensor: Spatially correlated Gaussian noise
            
        Notes:
            - Uses conv2d for efficiency (can leverage GPU convolution kernels)
            - Kernel size automatically chosen based on correlation length
            - Proper padding maintains field size
            - More intuitive control than spectral methods for specific correlation lengths
            - Computational cost scales with correlation length
        """
        H, W = shape
        
        # Generate white noise with batch/channel dimensions for conv2d
        # Shape: (1, 1, H, W) for single batch, single channel
        white_noise = torch.randn(1, 1, H, W, device=self.device, dtype=self.dtype)
        
        # Create Gaussian kernel for convolution
        # Kernel size: 4σ covers ~99.99% of Gaussian mass, ensure odd size
        kernel_size = max(3, int(4 * correlation_length) // 2 * 2 + 1)  
        sigma = correlation_length / 2.0  # Convert correlation length to Gaussian σ
        
        # Create 2D Gaussian kernel: G(x,y) = exp(-(x² + y²)/(2σ²))
        x = torch.arange(kernel_size, device=self.device, dtype=self.dtype)
        x = x - kernel_size // 2  # Center the kernel
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        
        # Normalize kernel to preserve noise amplitude
        kernel = kernel / torch.sum(kernel)  
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # Add batch/channel dims
        
        # Apply convolution with padding to maintain spatial dimensions
        padding = kernel_size // 2
        correlated_noise = F.conv2d(white_noise, kernel, padding=padding)
        
        # Remove batch/channel dimensions and normalize amplitude
        correlated_noise = correlated_noise.squeeze()
        correlated_noise = correlated_noise * std / torch.std(correlated_noise)
        
        return correlated_noise
    
    def temporal_noise_sequence(self, spatial_shape, n_timesteps, dt=0.01, 
                               tau=1.0, std=1.0, noise_type='white'):
        """
        Generate temporally correlated noise sequence using Ornstein-Uhlenbeck process.
        
        The Ornstein-Uhlenbeck (OU) process satisfies:
        dX = -θX dt + σ dW(t)
        
        Where:
        - θ = 1/τ is the relaxation rate
        - τ is the correlation time
        - σ controls the noise strength
        - dW(t) is the Wiener process increment
        
        Properties:
        - Mean-reverting: noise tends to return to zero
        - Exponential temporal correlation: C(t) ∝ exp(-|t|/τ)
        - Stationary statistics in the long-time limit
        
        Physical applications:
        - Models thermal fluctuations with finite correlation time
        - Represents environmental noise with memory
        - Used in Langevin equations for particle motion
        
        Args:
            spatial_shape: tuple, spatial dimensions (H, W)
            n_timesteps: int, number of time steps in the sequence
            dt: float, time step size (affects discretization accuracy)
            tau: float, correlation time (τ in equations)
                 Controls how quickly correlations decay
                 Large τ = long memory, slow decorrelation
            std: float, stationary standard deviation of the process
            noise_type: str, type of driving noise ('white' or 'colored')
                       'colored' uses pink noise as driving force
            
        Returns:
            torch.Tensor: Shape (n_timesteps, H, W) temporal noise sequence
            
        Notes:
            - Euler-Maruyama discretization of the SDE
            - First timestep is zero (initial condition)
            - Each spatial point evolves independently
            - Driving noise strength adjusted to maintain correct stationary variance
            - Can use colored driving noise for more complex temporal structure
        """
        dt = torch.tensor(dt, device=self.device)
        H, W = spatial_shape
        
        # Initialize noise sequence - starts from zero
        noise_sequence = torch.zeros(n_timesteps, H, W, device=self.device, dtype=self.dtype)
        
        # OU process parameters
        theta = torch.tensor(1.0 / tau, device=self.device)  # Relaxation rate
        # Noise strength: σ = std * sqrt(2θ) for correct stationary variance
        sigma = std * torch.sqrt(2 * theta)
        
        # Generate sequence using Euler-Maruyama scheme
        for t in range(1, n_timesteps):
            # Previous noise state
            prev_noise = noise_sequence[t-1]
            
            # Generate driving noise increment
            if noise_type == 'white':
                # Standard Wiener process increment: dW = N(0,1) * sqrt(dt)
                dW = torch.randn(H, W, device=self.device, dtype=self.dtype) * torch.sqrt(dt)
            else:
                # Use colored noise as driving force (non-standard but interesting)
                dW = self.colored_noise_spectral((H, W), alpha=1.0) * torch.sqrt(dt)
            
            # OU update: X(t+dt) = X(t)*exp(-θ*dt) + σ*dW
            # Exact solution for linear SDE with additive noise
            noise_sequence[t] = prev_noise * torch.exp(-theta * dt) + sigma * dW
        
        return noise_sequence
    
    def mesh_scaled_noise(self, shape, dx=1.0, dy=1.0, std=1.0):
        """
        Generate noise properly scaled for finite difference mesh discretization.
        
        In numerical PDEs, noise must be scaled with mesh size to ensure:
        1. Convergence as mesh is refined (dx, dy → 0)
        2. Correct physical units and amplitude
        3. Proper weak/strong convergence for stochastic PDEs
        
        For white noise in 2D, proper scaling is:
        ξ_discrete = ξ_continuous / sqrt(dx * dy)
        
        This ensures that ∫∫ ξ²(x,y) dx dy remains finite as mesh is refined.
        
        Physical reasoning:
        - Noise represents point sources distributed in space
        - Smaller mesh cells → higher point density → larger amplitude per cell
        - Maintains total noise power independent of discretization
        
        Args:
            shape: tuple, spatial dimensions (H, W)
            dx, dy: float, mesh spacing in x and y directions
                   Should match the spacing used in your PDE discretization
            std: float, physical noise amplitude (before mesh scaling)
                This should be the amplitude you want in continuous units
                
        Returns:
            torch.Tensor: Mesh-scaled white noise
            
        Notes:
            - Critical for convergence studies and multi-resolution simulations
            - Factor scales as 1/sqrt(cell_area) in 2D
            - In 1D: scale by 1/sqrt(dx), in 3D: scale by 1/sqrt(dx*dy*dz)
            - Ignore this at your own peril - results won't converge!
        """
        # Scale noise amplitude with mesh size for proper discretization
        # Factor of 1/sqrt(area) ensures proper continuous limit
        mesh_factor = 1.0 / torch.sqrt(torch.tensor(dx * dy, device=self.device))
        effective_std = std * mesh_factor
        
        return self.white_noise(shape, std=effective_std)
    
    def boundary_compatible_noise(self, shape, boundary_type='periodic'):
        """
        Generate noise compatible with specific boundary conditions.
        
        Different PDEs require different boundary conditions, and the noise
        should respect these same conditions for physical consistency and
        numerical stability.
        
        Boundary types:
        1. Periodic: ξ(0) = ξ(L), ∂ξ/∂x(0) = ∂ξ/∂x(L)
           - Used in problems with periodic domains (e.g., turbulence in boxes)
           - Naturally satisfied by FFT-based noise generation
           
        2. Dirichlet: ξ = 0 at boundaries
           - Used when the field is fixed at boundaries
           - Common in heat equation with fixed temperature boundaries
           
        3. Neumann: ∂ξ/∂n = 0 at boundaries (zero normal derivative)
           - Used when flux is specified at boundaries
           - Common in diffusion with insulating boundaries
        
        Args:
            shape: tuple, spatial dimensions (H, W)
            boundary_type: str, type of boundary condition
                          'periodic', 'dirichlet', or 'neumann'
                          
        Returns:
            torch.Tensor: Boundary-compatible noise field
            
        Notes:
            - Boundary conditions should match those used in your PDE solver
            - Improper boundary conditions can cause numerical instabilities
            - Periodic noise naturally arises from FFT-based generation
            - Dirichlet: simply zero out boundary values
            - Neumann: reflect values to satisfy zero-derivative condition
        """
        H, W = shape
        noise = torch.randn(H, W, device=self.device, dtype=self.dtype)
        
        if boundary_type == 'periodic':
            # Ensure periodicity by using only compatible Fourier modes
            # FFT naturally creates periodic fields, so we just process through FFT
            noise_fft = torch.fft.fft2(noise)
            # Keep all modes for truly periodic noise
            noise = torch.fft.ifft2(noise_fft).real
            
        elif boundary_type == 'dirichlet':
            # Zero noise at boundaries: ξ(boundary) = 0
            # This is appropriate when the PDE solution is also zero at boundaries
            noise[0, :] = 0      # Top boundary
            noise[-1, :] = 0     # Bottom boundary  
            noise[:, 0] = 0      # Left boundary
            noise[:, -1] = 0     # Right boundary
            
        elif boundary_type == 'neumann':
            # Reflect noise at boundaries to satisfy zero-derivative condition
            # ∂ξ/∂n = 0 ⟹ ξ(boundary) = ξ(interior_neighbor)
            noise[0, :] = noise[1, :]       # ∂ξ/∂y = 0 at top
            noise[-1, :] = noise[-2, :]     # ∂ξ/∂y = 0 at bottom
            noise[:, 0] = noise[:, 1]       # ∂ξ/∂x = 0 at left  
            noise[:, -1] = noise[:, -2]     # ∂ξ/∂x = 0 at right
        
        return noise

# Example usage and demonstration
def demonstrate_noise_generation():
    """
    Comprehensive demonstration of all noise generation methods.
    
    This function shows how to use each method and provides examples
    of typical parameters and use cases.
    """
    # Initialize generator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    noise_gen = PDENoiseGenerator(device=device)
    
    # Grid parameters - typical for a moderate resolution 2D simulation
    H, W = 64, 64
    dx, dy = 0.1, 0.1  # Physical mesh spacing
    
    print(f"Using device: {device}")
    print(f"Grid size: {H}x{W}, dx={dx}, dy={dy}")
    
    # Generate different types of noise
    print("\nGenerating different noise types...")
    
    # 1. Basic white noise - fundamental building block
    white_noise = noise_gen.white_noise((H, W), std=1.0, seed=42)
    
    # 2. Colored noise (pink noise, alpha=1) - commented out in original
    # pink_noise = noise_gen.colored_noise_spectral((H, W), alpha=1.0, std=1.0)
    
    # 3. Spatially correlated noise - smoother than white noise
    corr_noise = noise_gen.spatially_correlated_noise((H, W), correlation_length=3.0, std=1.0)
    
    # 4. Mesh-scaled noise - proper for numerical convergence
    mesh_noise = noise_gen.mesh_scaled_noise((H, W), dx=dx, dy=dy, std=0.1)
    
    # 5. Temporal sequence - for time-dependent stochastic PDEs
    n_steps = 10
    temporal_noise = noise_gen.temporal_noise_sequence((H, W), n_steps, dt=0.01, tau=0.1)
    
    # 6. Boundary-compatible noise - respects PDE boundary conditions
    dirichlet_noise = noise_gen.boundary_compatible_noise((H, W), boundary_type='dirichlet')
    
    print("✓ White noise generated")
    # print("✓ Pink noise generated") 
    print("✓ Spatially correlated noise generated")
    print("✓ Mesh-scaled noise generated")
    print(f"✓ Temporal sequence generated ({n_steps} timesteps)")
    print("✓ Boundary-compatible noise generated")
    
    # Example: Adding noise to a PDE solution
    print("\nExample: Adding noise to a 2D solution...")
    
    # Create a simple 2D function (e.g., solution to a PDE)
    # This could represent temperature, concentration, wave amplitude, etc.
    x = torch.linspace(0, 2*np.pi, W, device=device)
    y = torch.linspace(0, 2*np.pi, H, device=device)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Example PDE solution (2D sine wave) - could be solution to wave equation
    clean_solution = torch.sin(X) * torch.cos(Y)
    
    # Add different types of noise - demonstrate the effect
    noisy_solution_white = clean_solution + 0.1 * white_noise
    noisy_solution_corr = clean_solution + 0.1 * corr_noise
    
    print("✓ Noise added to PDE solution")
    
    # Return results for potential visualization
    return {
        'white_noise': white_noise.cpu(),
        # 'pink_noise': pink_noise.cpu(),  # Commented out in original
        'corr_noise': corr_noise.cpu(),
        'mesh_noise': mesh_noise.cpu(),
        'temporal_noise': temporal_noise.cpu(),
        'dirichlet_noise': dirichlet_noise.cpu(),
        'clean_solution': clean_solution.cpu(),
        'noisy_solution_white': noisy_solution_white.cpu(),
        'noisy_solution_corr': noisy_solution_corr.cpu()
    }

# if __name__ == "__main__":
#     results = demonstrate_noise_generation()
# %%