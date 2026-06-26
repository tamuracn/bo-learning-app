# This is the toy function
# x, y are variables
# z will represent different qw's axis
# (2/n) : [(n+1)/n]

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def stoich_function(x, y, n, sigma=0.01, A=1.0):
    x0 = 2 / n
    y0 = (n + 1) / n

    return A * np.exp(
        -((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)
    )

def toy_function(x, y, z, sigma=0.1, A=2.5, noise_scale=1.0):
    """Evaluate the stoichiometry peak in the x-y plane at discrete layer z (integer n).

    noise_scale: std of additive Gaussian noise relative to A (0 = no noise).
    Keep noise_scale < 1 to preserve the peak.
    """
    n = int(z)
    signal = stoich_function(x, y, n, sigma=sigma, A=A)
    raw_noise = np.random.normal(0, noise_scale * A, size=np.shape(signal))
    noise = gaussian_filter(raw_noise, sigma=2)
    return signal + noise

#-- Example usage: 3D plot where each integer n is a surface at z=n
x = np.linspace(0, 3, 80)
y = np.linspace(0, 3, 80)
X, Y = np.meshgrid(x, y)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

cmap = plt.get_cmap('tab10')
for n in range(1, 11):
    Z_n = toy_function(X, Y, n)
    # plot the surface at height z=n; add n as a flat offset so layers are separated
    ax.plot_surface(X, Y, Z_n + n, alpha=0.6, color=cmap(n / 10), linewidth=0)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z (stoich layer n)')
ax.set_zticks(range(1, 11))
ax.set_zticklabels([f'n={n}' for n in range(1, 11)])
ax.set_title('toy_function(x, y, z): peak in x-y plane per stoich layer')
plt.tight_layout()
# plt.show()
