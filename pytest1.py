import numpy as np
import matplotlib.pyplot as plt

# Parameters
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(x, y)

# Define two base oscillatory modes (standing waves)
wave1 = np.sin(2 * X + 1.5 * Y)
wave2 = np.sin(1.2 * X - 2.0 * Y + 1)

# Superposition (interference pattern)
Psi = wave1 + wave2

# Gradient descent surface = interference energy landscape
E = (wave1 - wave2)**2

fig = plt.figure(figsize=(10, 6))

# Plot 1: Interference pattern
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Psi, cmap='viridis', linewidth=0, alpha=0.9)
ax1.set_title('Interference Pattern (Memory Wave Field)')
ax1.set_xlabel('Axis 1 (Sensory)')
ax1.set_ylabel('Axis 2 (Affective)')
ax1.set_zlabel('Amplitude')

# Plot 2: Energy surface (Gradient Descent Landscape)
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X, Y, E, cmap='inferno', linewidth=0, alpha=0.9)
ax2.set_title('Gradient Descent Energy Surface')
ax2.set_xlabel('Phase shift X')
ax2.set_ylabel('Phase shift Y')
ax2.set_zlabel('Interference Energy')

plt.tight_layout()
plt.show()
