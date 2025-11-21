
# %% 
#FC example case is borrowed from: https://neuraloperator.github.io/dev/auto_examples/layers/plot_fourier_continuation.html#sphx-glr-auto-examples-layers-plot-fourier-continuation-py
import sys 
sys.path.append("..")
import torch 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from neuralop.layers.fourier_continuation import FCLegendre, FCGram


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating an example of 1D non-periodic function
# ----------------------------------------------
# We consider f(x) = sin(16x) - cos(8x) on the interval [0,1].
# This function is not periodic on [0,1], making it a good test case
# for Fourier continuation methods.

length_signal = 101  # Length of the original 1D signal
add_pts = 50  # Number of additional points for continuation
batch_size = 3  # Batch size for processing multiple signals

# Create the input signal
x = torch.linspace(0, 1, length_signal).repeat(batch_size, 1)
dx = x[1] - x[0]
f = torch.sin(16 * x) - torch.cos(8 * x)
f_x = lambda x: 16*torch.cos(16*x) + 8*torch.sin(8*x)
f_x_analytical = f_x(x)
# %% 
#Differentiating using ConvOps
from Utils.PRE.ConvOps_0d import ConvOperator
#Direct Convolution
D_x = ConvOperator(order=1)
f_x_conv = D_x(f)
#Spectral Convolution
f_x_spectral = D_x.spectral_convolution(f)

#Custom
f_x = D_x.differentiate(f)

plt.figure()
plt.plot(f_x_conv[0], label='Conv')
plt.plot(f_x_spectral[0], label='Spectral')
plt.plot(f_x[0], label='Custom')
plt.legend()
plt.title('f_x')

# %% 
#Integrating using ConvOps
f_integrated = D_x.integrate(f_x_conv)
plt.figure()
plt.plot(f[0], label='Signal')
plt.plot(f_integrated[0], label='retreived')
plt.legend()
plt.title('f')

# %%
# Extending the signal
# -----------------------------------------
# We use the FC-Legendre and FC-Gram Fourier continuation layers to extend the signal.
# FC-Legendre: Uses Legendre polynomial basis for continuation
Extension_Legendre = FCLegendre(d=2, n_additional_pts=add_pts)
f_extend_Legendre = Extension_Legendre(f, dim=1)

f_x_FC = D_x.differentiate(f_extend_Legendre)


plt.figure()
plt.plot(f_extend_Legendre[0], label='Analytical')
plt.plot(f_x_FC[0], label='Diff')
plt.legend()
plt.title('f_x')

# %% 
#Integrating using ConvOps
f_integrated_FC = D_x.integrate(f_x_FC)
# %%
