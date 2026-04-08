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



# %% 
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class PDENoiseGenerator1D:
    """
    A 1D noise generator for batched PDE simulations in PyTorch.
    
    This class generates noise with shape [batch_size, n_points] where:
    - Each batch element is independent (no correlation across batch dimension)
    - Spatial correlations exist within each 1D sequence (along n_points)
    - Optimized for batched neural PDE solvers and parallel simulations
    
    Common applications:
    - Neural operators processing multiple 1D PDEs simultaneously
    - Monte Carlo simulations with many realizations
    - Time series of spatial fields where each batch is a different time
    - Parameter sweeps where each batch has different PDE parameters
    """
    
    def __init__(self, device='cpu', dtype=torch.float32):
        """
        Initialize the 1D noise generator.
        
        Args:
            device: str or torch.device, computation device ('cpu', 'cuda', etc.)
            dtype: torch.dtype, data type for generated tensors
        """
        self.device = device
        self.dtype = dtype
    
    def white_noise(self, batch_size, n_points, std=1.0, seed=None):
        """
        Generate batched 1D white noise.
        
        Creates independent white noise for each batch element and each spatial point.
        White noise properties:
        - Zero mean: E[ξ(x)] = 0
        - Delta-correlated in space: E[ξ(x)ξ(y)] = σ²δ(x-y)
        - Independent across batch: E[ξᵢ(x)ξⱼ(y)] = 0 for i ≠ j
        - Gaussian distribution at each point
        
        This is the fundamental building block for stochastic PDEs of the form:
        ∂u/∂t = F(u, ∂u/∂x, ∂²u/∂x², ...) + σξ(x,t)
        
        Args:
            batch_size: int, number of independent noise realizations
                       Each batch element represents a different:
                       - PDE realization for Monte Carlo
                       - Time snapshot in a sequence
                       - Parameter configuration
            n_points: int, number of spatial discretization points
                     Should match the spatial resolution of your PDE grid
            std: float, standard deviation of the noise (σ in mathematical notation)
                The physical amplitude of fluctuations
            seed: int, random seed for reproducibility
                 Useful for debugging and comparing different methods
                 
        Returns:
            torch.Tensor: Shape [batch_size, n_points] of independent white noise
            
        Notes:
            - Each (batch, point) pair is independent
            - Memory efficient - no correlations stored
            - Scales linearly with batch_size and n_points
            - Foundation for building more complex noise types
            - Satisfies central limit theorem for ensemble averages
            
        Example:
            # For neural operator training with 32 examples, 128 grid points
            noise = generator.white_noise(32, 128, std=0.1)
            
            # Add to PDE solution: u_noisy = u_clean + noise
            u_noisy = u_clean + noise
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        return torch.randn(batch_size, n_points, device=self.device, dtype=self.dtype) * std
    
    def spatially_correlated_noise(self, batch_size, n_points, correlation_length=5.0, std=1.0, seed=None):
        """
        Generate batched 1D spatially correlated noise using Gaussian kernel convolution.
        
        Creates noise with exponential spatial correlation within each batch element:
        C(r) ∝ exp(-r²/(2L²)) where L is the correlation length
        
        Key properties:
        - Spatial correlations within each 1D sequence (along n_points)
        - No correlations across batch dimension (each batch independent)
        - Preserves Gaussian statistics (convolution of Gaussians is Gaussian)
        - Isotropic correlations in 1D (symmetric around each point)
        
        Method: 
        1. Generate independent white noise for each batch
        2. Apply 1D convolution with Gaussian kernel to each batch element
        3. The convolution operates along the spatial dimension only
        
        Physical interpretation:
        - Models measurement noise with finite instrument resolution
        - Represents local averaging effects in physical systems
        - Mimics diffusion-limited processes or finite-size effects
        - Common in experimental data where sensors have spatial extent
        
        Applications:
        - Smoothed initial conditions for PDE ensembles
        - Measurement noise in inverse problems
        - Subgrid-scale fluctuations in turbulence models
        - Uncertainty quantification with correlated errors
        
        Args:
            batch_size: int, number of independent noise realizations
                       Each batch has its own independent spatial correlation pattern
            n_points: int, number of spatial discretization points
                     Determines the spatial resolution of correlations
            correlation_length: float, spatial correlation length in grid units
                              Controls the spatial smoothness scale
                              - Small values (< 2): nearly white noise
                              - Medium values (2-10): visible correlations
                              - Large values (> n_points/4): very smooth
            std: float, noise amplitude after normalization
                Physical amplitude of the correlated fluctuations
            seed: int, random seed for reproducibility
                 
        Returns:
            torch.Tensor: Shape [batch_size, n_points] of spatially correlated noise
            
        Notes:
            - Uses conv1d for computational efficiency
            - Kernel size automatically chosen based on correlation length
            - Proper padding maintains spatial dimension size
            - Each batch element has independent but statistically identical correlations
            - Computational cost scales with correlation length
            - For very large correlation lengths, consider spectral methods instead
            
        Implementation details:
            - Gaussian kernel: G(x) = exp(-x²/(2σ²)) where σ = correlation_length/2
            - Kernel size: 4σ captures ~99.99% of Gaussian mass
            - Normalization preserves total noise power
            - Padding='same' maintains n_points dimension
            
        Example:
            # Generate smooth noise for 16 PDE realizations, 64 grid points
            # with correlation length of 3 grid spacings
            noise = generator.spatially_correlated_noise(16, 64, 
                                                       correlation_length=3.0, 
                                                       std=0.05)
            
            # Each row is an independent correlated noise realization
            assert noise.shape == (16, 64)
            
            # Verify independence across batches
            correlation_across_batches = torch.corrcoef(noise)  # Should be ~identity
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        # Generate independent white noise for each batch element
        # Shape: [batch_size, 1, n_points] for conv1d (needs channel dimension)
        white_noise = torch.randn(batch_size, 1, n_points, device=self.device, dtype=self.dtype)
        
        # Create 1D Gaussian kernel for convolution
        # Kernel size: 4σ covers ~99.99% of Gaussian mass, ensure odd size
        kernel_size = max(3, int(4 * correlation_length) // 2 * 2 + 1)
        sigma = correlation_length / 2.0  # Convert correlation length to Gaussian σ
        
        # Create 1D Gaussian kernel: G(x) = exp(-x²/(2σ²))
        x = torch.arange(kernel_size, device=self.device, dtype=self.dtype)
        x = x - kernel_size // 2  # Center the kernel around zero
        kernel = torch.exp(-x**2 / (2 * sigma**2))
        
        # Normalize kernel to preserve noise amplitude
        # This ensures that convolution doesn't change the total power
        kernel = kernel / torch.sum(kernel)
        
        # Reshape kernel for conv1d: [out_channels, in_channels, kernel_size]
        # We want: [1, 1, kernel_size] for single channel convolution
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        # Apply 1D convolution with padding to maintain spatial dimensions
        # Each batch element is convolved independently
        padding = kernel_size // 2
        correlated_noise = F.conv1d(white_noise, kernel, padding=padding)
        
        # Remove channel dimension: [batch_size, 1, n_points] → [batch_size, n_points]
        correlated_noise = correlated_noise.squeeze(1)
        
        # Normalize to desired standard deviation
        # Convolution can change amplitude, so we renormalize to maintain std
        current_std = torch.std(correlated_noise)
        if current_std > 1e-10:  # Avoid division by zero
            correlated_noise = correlated_noise * std / current_std
        
        return correlated_noise
    

    def pre_correlated_noise(self, batch_size, n_points, kernel, std=1.0, seed=None):
        """
        Generate batched 1D spatially correlated noise using the PRE kernel
        
        
        Args:
            batch_size: int, number of independent noise realizations
                       Each batch has its own independent spatial correlation pattern
            n_points: int, number of spatial discretization points
                     Determines the spatial resolution of correlations
            kernel: Additive PRE Kernel. 

            std: float, noise amplitude after normalization
                Physical amplitude of the correlated fluctuations
            seed: int, random seed for reproducibility
                 
        Returns:
            torch.Tensor: Shape [batch_size, n_points] of spatially correlated noise
            
        
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        # Generate independent white noise for each batch element
        # Shape: [batch_size, 1, n_points] for conv1d (needs channel dimension)
        white_noise = torch.randn(batch_size, 1, n_points, device=self.device, dtype=self.dtype)
        
        
        
        # Reshape kernel for conv1d: [out_channels, in_channels, kernel_size]
        # We want: [1, 1, kernel_size] for single channel convolution
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        kernel_size = kernel.shape[-1]
        
        # Apply 1D convolution with padding to maintain spatial dimensions
        # Each batch element is convolved independently
        padding = kernel_size // 2
        correlated_noise = F.conv1d(white_noise, kernel, padding=padding)
        
        # Remove channel dimension: [batch_size, 1, n_points] → [batch_size, n_points]
        correlated_noise = correlated_noise.squeeze(1)
        
        # Normalize to desired standard deviation
        # Convolution can change amplitude, so we renormalize to maintain std
        current_std = torch.std(correlated_noise)
        if current_std > 1e-10:  # Avoid division by zero
            correlated_noise = correlated_noise * std / current_std
        
        return correlated_noise

    # ----- B-spline noise (C2-continuous by construction) -----

    @staticmethod
    def _cubic_bspline_1d(t):
        """Uniform cubic B-spline kernel, support on [-2, 2], C2 continuous.

        Adapted from the SplineFNO implementation.
        """
        abs_t = torch.abs(t)
        result = torch.zeros_like(t)

        mask1 = abs_t < 1
        a1 = abs_t[mask1]
        result[mask1] = (2.0 / 3.0) - a1 ** 2 + 0.5 * a1 ** 3

        mask2 = (abs_t >= 1) & (abs_t < 2)
        a2 = abs_t[mask2]
        result[mask2] = (1.0 / 6.0) * (2.0 - a2) ** 3

        return result

    def _build_bspline_basis_1d(self, n_points, n_knots, device, dtype):
        """Build 1D B-spline basis matrix Phi [n_points, n_knots].

        Parameters
        ----------
        n_points : int
            Number of evaluation points (the signal length).
        n_knots : int
            Number of uniformly-spaced control points.

        Returns
        -------
        Phi : Tensor [n_points, n_knots]
        """
        eval_pts = torch.linspace(0.0, 1.0, n_points, device=device, dtype=dtype)
        knots = torch.linspace(0.0, 1.0, n_knots, device=device, dtype=dtype)
        dk = knots[1] - knots[0]

        # Relative distances: [n_points, n_knots]
        t = (eval_pts.unsqueeze(1) - knots.unsqueeze(0)) / dk
        Phi = self._cubic_bspline_1d(t)
        return Phi

    def bspline_noise(self, batch_size, n_points, n_knots=16, std=1.0, seed=None):
        """Generate C2-continuous noise via cubic B-spline basis.

        Samples random control-point coefficients and evaluates the spline,
        producing noise that is smooth by construction.  The spatial scale is
        controlled by ``n_knots``: fewer knots produce smoother (longer-
        wavelength) perturbations.

        Parameters
        ----------
        batch_size : int
            Number of independent noise realizations.
        n_points : int
            Number of spatial discretization points.
        n_knots : int
            Number of uniform B-spline control points.  Typical range 8-32.
            Lower values give smoother noise; higher values approach white noise.
        std : float
            Desired standard deviation of the output noise.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        Tensor [batch_size, n_points]
        """
        if seed is not None:
            torch.manual_seed(seed)

        Phi = self._build_bspline_basis_1d(n_points, n_knots, self.device, self.dtype)

        # Random control-point coefficients: [batch_size, n_knots]
        coeffs = torch.randn(batch_size, n_knots, device=self.device, dtype=self.dtype)

        # Evaluate spline: [batch_size, n_knots] @ [n_knots, n_points] -> [batch_size, n_points]
        noise = coeffs @ Phi.t()

        # Normalize to target std
        current_std = torch.std(noise)
        if current_std > 1e-10:
            noise = noise * std / current_std

        return noise


    def gp_noise(self, batch_size, n_points, correlation_length=5.0, std=1.0, 
                                 kernel_type='rbf', nu=1.5, seed=None):
        """
        Generate batched 1D spatially correlated noise using Gaussian Process sampling.
        
        This method uses GPyTorch to sample from a Gaussian Process with specified
        covariance structure. This provides more flexible and theoretically principled
        spatial correlations compared to convolution-based methods.
        
        Advantages over convolution method:
        - Exact covariance structure (not approximate via convolution)
        - Multiple kernel types available (RBF, Matérn, Periodic, etc.)
        - Better theoretical foundation from Gaussian Process theory
        - Naturally handles boundary conditions and non-uniform grids
        - Can easily extend to more complex covariance functions
        
        Available kernel types:
        - 'rbf': Radial Basis Function (Gaussian) kernel
          k(x,x') = σ² exp(-|x-x'|²/(2l²))
          Infinitely differentiable, very smooth samples
          
        - 'matern': Matérn kernel with parameter ν
          k(x,x') = σ²(2^(1-ν)/Γ(ν))(√(2ν)|x-x'|/l)^ν K_ν(√(2ν)|x-x'|/l)
          Controls smoothness: ν=0.5 (rough), ν=1.5 (medium), ν=2.5 (smooth)
          
        - 'periodic': Periodic kernel for cyclic domains
          k(x,x') = σ² exp(-2sin²(π|x-x'|/p)/l²)
          
        Physical interpretation:
        - RBF: Very smooth processes, good for temperature fields, smooth flows
        - Matérn: More realistic for natural phenomena, finite smoothness
        - Periodic: For processes with known periodicity
        
        Args:
            batch_size: int, number of independent GP realizations
                       Each batch samples independently from the same GP prior
            n_points: int, number of spatial discretization points
                     These will be the input locations for GP sampling
            correlation_length: float, length scale of the covariance function
                              Controls spatial correlation distance
                              Larger values = smoother, more correlated samples
            std: float, marginal standard deviation of the GP
                Physical amplitude of fluctuations (σ in kernel formulas)
            kernel_type: str, type of covariance kernel
                        'rbf', 'matern', or 'periodic'
            nu: float, smoothness parameter for Matérn kernel (ignored for other kernels)
               Common values: 0.5 (Ornstein-Uhlenbeck), 1.5, 2.5, ∞ (RBF)
            seed: int, random seed for reproducibility
                 
        Returns:
            torch.Tensor: Shape [batch_size, n_points] of GP samples
            
        Notes:
            - Requires GPyTorch installation: pip install gpytorch
            - Each batch is an independent sample from the same GP prior
            - Computational complexity: O(n_points³) for exact sampling
            - For large n_points (>1000), consider approximate methods
            - Covariance matrix is positive definite by construction
            - Handles numerical stability via Cholesky decomposition
            
        Mathematical details:
            - Samples f ~ GP(0, k(x,x')) where k is the chosen kernel
            - Uses Cholesky decomposition: f = L @ z where z ~ N(0,I)
            - L is Cholesky factor of covariance matrix K
            - Ensures exact covariance structure (up to numerical precision)
            
        Example:
            # Smooth RBF samples
            noise_rbf = generator.spatially_correlated_noise(
                16, 64, correlation_length=5.0, kernel_type='rbf'
            )
            
            # Rougher Matérn samples  
            noise_matern = generator.spatially_correlated_noise(
                16, 64, correlation_length=5.0, kernel_type='matern', nu=0.5
            )
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        try:
            import gpytorch
        except ImportError:
            print("Warning: GPyTorch not available. Falling back to convolution method.")
            return self.spatially_correlated_noise(batch_size, n_points,
                                                    correlation_length=correlation_length,
                                                    std=std, seed=seed)
        
        # Create input locations (spatial grid)
        # Normalize to [0, 1] interval for better numerical stability
        x_locations = torch.linspace(0, 1, n_points, device=self.device, dtype=self.dtype)
        
        # Create covariance kernel based on type
        if kernel_type.lower() == 'rbf':
            # RBF (Gaussian) kernel: k(x,x') = σ² exp(-|x-x'|²/(2l²))
            covar_module = gpytorch.kernels.RBFKernel()
            
        elif kernel_type.lower() == 'matern':
            # Matérn kernel with smoothness parameter nu
            covar_module = gpytorch.kernels.MaternKernel(nu=nu)
            
        elif kernel_type.lower() == 'periodic':
            # Periodic kernel for cyclic domains
            covar_module = gpytorch.kernels.PeriodicKernel()
            
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}. Use 'rbf', 'matern', or 'periodic'")
        
        # Set length scale to control correlation distance
        # Convert correlation_length to appropriate scale for [0,1] domain
        normalized_length_scale = correlation_length / n_points
        covar_module.lengthscale = normalized_length_scale
        
        # Scale kernel output to desired standard deviation
        scaled_covar_module = gpytorch.kernels.ScaleKernel(covar_module)
        scaled_covar_module.outputscale = std**2
        
        # Compute covariance matrix
        # Shape: [n_points, n_points]
        # Need to provide both sets of inputs for cross-covariance
        with torch.no_grad():
            covar_matrix = scaled_covar_module(x_locations, x_locations).evaluate()
        
        # Add small diagonal term for numerical stability (jitter)
        jitter = 1e-6
        covar_matrix += jitter * torch.eye(n_points, device=self.device, dtype=self.dtype)
        
        # Cholesky decomposition for sampling: K = L @ L^T
        try:
            L = torch.linalg.cholesky(covar_matrix)
        except RuntimeError as e:
            print(f"Warning: Cholesky decomposition failed ({e}). Adding more jitter.")
            jitter = 1e-4
            covar_matrix += jitter * torch.eye(n_points, device=self.device, dtype=self.dtype)
            L = torch.linalg.cholesky(covar_matrix)
        
        # Sample standard normal random variables
        # Shape: [batch_size, n_points]
        z = torch.randn(batch_size, n_points, device=self.device, dtype=self.dtype)
        
        # Transform to get GP samples: f = L @ z^T
        # L: [n_points, n_points], z^T: [n_points, batch_size]
        # Result: [n_points, batch_size] -> transpose to [batch_size, n_points]
        gp_samples = (L @ z.T).T
        
        return gp_samples
    

    
    def mesh_scaled_noise(self, batch_size, n_points, dx=1.0, std=1.0, correlation_length=None, seed=None):
        """
        Generate batched mesh-scaled noise for numerical PDE convergence.
        
        In numerical PDEs, noise must be scaled with mesh size to ensure:
        1. Convergence as mesh is refined (dx → 0)
        2. Correct physical units and amplitude
        3. Proper weak/strong convergence for stochastic PDEs
        
        For white noise in 1D, proper scaling is:
        ξ_discrete = ξ_continuous / sqrt(dx)
        
        For correlated noise, the scaling depends on whether the correlation
        length is fixed in physical units or grid units.
        
        Args:
            batch_size: int, number of independent noise realizations
            n_points: int, number of spatial discretization points
            dx: float, mesh spacing in physical units
               Should match the spacing used in your PDE discretization
            std: float, physical noise amplitude (before mesh scaling)
                This should be the amplitude you want in continuous units
            correlation_length: float, optional, if provided generates correlated noise
                               If None, generates white noise
                               Can be in grid units or physical units (specify clearly)
            seed: int, random seed for reproducibility
                 
        Returns:
            torch.Tensor: Shape [batch_size, n_points] of properly scaled noise
            
        Notes:
            - Critical for convergence studies and multi-resolution simulations
            - Factor scales as 1/sqrt(dx) in 1D
            - For correlation_length in physical units: keep fixed as dx changes
            - For correlation_length in grid units: scale with mesh refinement
        """
        # Scale noise amplitude with mesh size for proper discretization
        mesh_factor = 1.0 / torch.sqrt(torch.tensor(dx, device=self.device))
        effective_std = std * mesh_factor
        
        if correlation_length is None:
            # Generate mesh-scaled white noise
            return self.white_noise(batch_size, n_points, std=effective_std, seed=seed)
        else:
            # Generate mesh-scaled correlated noise
            return self.spatially_correlated_noise(batch_size, n_points, 
                                                 correlation_length=correlation_length, 
                                                 std=effective_std, seed=seed)

# # Example usage and demonstration
# def demonstrate_1d_noise_generation():
#     """
#     Comprehensive demonstration of 1D batched noise generation methods.
    
#     Shows typical use cases for neural PDE solvers and batched simulations.
#     """
#     # Initialize generator
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     noise_gen = PDENoiseGenerator1D(device=device)
    
#     # Typical parameters for neural operator training
#     batch_size = 16    # Number of training examples or Monte Carlo realizations
#     n_points = 128     # Spatial resolution (e.g., 128 grid points)
#     dx = 0.1          # Physical mesh spacing
    
#     print(f"Using device: {device}")
#     print(f"Batch size: {batch_size}, Grid points: {n_points}, dx: {dx}")
    
#     # Generate different types of 1D batched noise
#     print("\nGenerating 1D batched noise types...")
    
#     # 1. Basic white noise - independent at each point and batch
#     white_noise = noise_gen.white_noise(batch_size, n_points, std=1.0, seed=42)
#     print(f"✓ White noise generated: shape {white_noise.shape}")
    
#     # 2. Spatially correlated noise - smooth within each batch, independent across batches
#     corr_noise = noise_gen.spatially_correlated_noise(batch_size, n_points, 
#                                                      correlation_length=5.0, std=1.0, seed=42)
#     print(f"✓ Spatially correlated noise generated: shape {corr_noise.shape}")
    
#     # 3. Mesh-scaled white noise - proper for convergence studies
#     mesh_white = noise_gen.mesh_scaled_noise(batch_size, n_points, dx=dx, std=0.1, seed=42)
#     print(f"✓ Mesh-scaled white noise generated: shape {mesh_white.shape}")
    
#     # 4. Mesh-scaled correlated noise
#     mesh_corr = noise_gen.mesh_scaled_noise(batch_size, n_points, dx=dx, std=0.1, 
#                                           correlation_length=3.0, seed=42)
#     print(f"✓ Mesh-scaled correlated noise generated: shape {mesh_corr.shape}")
    
#     # Verification: Check independence across batches
#     print("\nVerifying statistical properties...")
    
#     # Check that batches are independent (correlation should be near zero)
#     if batch_size > 1:
#         batch_correlation = torch.corrcoef(white_noise)
#         off_diagonal = batch_correlation[torch.triu(torch.ones_like(batch_correlation), diagonal=1) == 1]
#         mean_cross_corr = torch.mean(torch.abs(off_diagonal))
#         print(f"Mean cross-batch correlation (should be ~0): {mean_cross_corr:.4f}")
    
#     # Check spatial correlations within batches
#     if n_points > 10:
#         # Compute spatial autocorrelation for first batch element
#         first_batch = corr_noise[0]  # Shape: [n_points]
#         # Simple lag-1 correlation
#         lag1_corr = torch.corrcoef(torch.stack([first_batch[:-1], first_batch[1:]]))[0,1]
#         print(f"Spatial lag-1 correlation in correlated noise: {lag1_corr:.4f}")
        
#         # Same for white noise (should be much smaller)
#         first_batch_white = white_noise[0]
#         lag1_corr_white = torch.corrcoef(torch.stack([first_batch_white[:-1], first_batch_white[1:]]))[0,1]
#         print(f"Spatial lag-1 correlation in white noise: {lag1_corr_white:.4f}")
    
#     # Example: Using with a simple 1D PDE solution
#     print("\nExample: Adding noise to 1D PDE solutions...")
    
#     # Create a batch of 1D solutions (e.g., sine waves with different frequencies)
#     x = torch.linspace(0, 2*np.pi, n_points, device=device)
#     frequencies = torch.linspace(1, 3, batch_size, device=device).unsqueeze(1)  # [batch_size, 1]
#     clean_solutions = torch.sin(frequencies * x)  # [batch_size, n_points]
    
#     # Add different types of noise
#     noisy_white = clean_solutions + 0.1 * white_noise
#     noisy_corr = clean_solutions + 0.1 * corr_noise
    
#     print(f"✓ Noise added to {batch_size} different 1D PDE solutions")
#     print(f"  Clean solutions shape: {clean_solutions.shape}")
#     print(f"  Noisy solutions shape: {noisy_white.shape}")
    
#     # Return results for potential visualization or further analysis
#     return {
#         'white_noise': white_noise.cpu(),
#         'corr_noise': corr_noise.cpu(),
#         'mesh_white': mesh_white.cpu(),
#         'mesh_corr': mesh_corr.cpu(),
#         'clean_solutions': clean_solutions.cpu(),
#         'noisy_white': noisy_white.cpu(),
#         'noisy_corr': noisy_corr.cpu(),
#         'x_grid': x.cpu(),
#         'frequencies': frequencies.cpu()
#     }

# if __name__ == "__main__":
#     results = demonstrate_1d_noise_generation()
    
#     # Optional: Quick visualization if matplotlib is available
#     try:
#         import matplotlib.pyplot as plt
        
#         print("\nCreating sample visualization...")
#         fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
#         # Plot first few batch elements
#         n_show = min(4, results['white_noise'].shape[0])
        
#         # Convert all tensors to numpy for matplotlib compatibility
#         x_np = results['x_grid'].numpy()
#         white_noise_np = results['white_noise'][:n_show].numpy()
#         corr_noise_np = results['corr_noise'][:n_show].numpy()
#         clean_solutions_np = results['clean_solutions'][:n_show].numpy()
#         noisy_corr_np = results['noisy_corr'][:n_show].numpy()
        
#         # White noise
#         axes[0,0].plot(x_np, white_noise_np.T)
#         axes[0,0].set_title('White Noise (first 4 batches)')
#         axes[0,0].set_xlabel('x')
#         axes[0,0].set_ylabel('ξ(x)')
#         axes[0,0].grid(True, alpha=0.3)
        
#         # Correlated noise  
#         axes[0,1].plot(x_np, corr_noise_np.T)
#         axes[0,1].set_title('Spatially Correlated Noise')
#         axes[0,1].set_xlabel('x')
#         axes[0,1].set_ylabel('ξ(x)')
#         axes[0,1].grid(True, alpha=0.3)
        
#         # Clean solutions
#         axes[1,0].plot(x_np, clean_solutions_np.T)
#         axes[1,0].set_title('Clean PDE Solutions')
#         axes[1,0].set_xlabel('x')
#         axes[1,0].set_ylabel('u(x)')
#         axes[1,0].grid(True, alpha=0.3)
        
#         # Noisy solutions
#         axes[1,1].plot(x_np, noisy_corr_np.T)
#         axes[1,1].set_title('Noisy PDE Solutions')
#         axes[1,1].set_xlabel('x')
#         axes[1,1].set_ylabel('u(x) + ξ(x)')
#         axes[1,1].grid(True, alpha=0.3)
        
#         plt.tight_layout()
#         plt.savefig('1d_noise_demo.png', dpi=150, bbox_inches='tight')
#         print("✓ Visualization saved as '1d_noise_demo.png'")
        
#     except ImportError:
#         print("Matplotlib not available for visualization")
#     except Exception as e:
#         print(f"Visualization failed: {e}")
#         print("This is likely due to tensor/numpy conversion issues")

# # %%