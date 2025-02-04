#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi, linspace, meshgrid, full_like

# Stereographic projection function
def stereographic_projection(x, y, z):
    denom = 1 - z
    x_proj = np.full_like(x, np.nan, dtype=float)
    y_proj = np.full_like(y, np.nan, dtype=float)
    mask = denom != 0
    x_proj[mask] = x[mask] / denom[mask]
    y_proj[mask] = y[mask] / denom[mask]
    return x_proj, y_proj

# Function to generate the unit sphere mesh
def generate_sphere_mesh(theta_steps=50, phi_steps=50):
    theta = np.linspace(0, np.pi, theta_steps)
    phi = np.linspace(0, 2 * np.pi, phi_steps)
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z, theta, phi

# Generate the sphere mesh
x, y, z, theta, phi = generate_sphere_mesh()

# Define Curve 1: Latitude at 45 degrees (constant theta)
t = np.linspace(0, 2 * np.pi, 300)
curve1_x = np.sin(np.pi / 4) * np.cos(t)
curve1_y = np.sin(np.pi / 4) * np.sin(t)
curve1_z = np.cos(np.pi / 4) * np.ones_like(t)

# Define Curve 2: A helical-like path on the sphere
curve2_x = np.cos(t)
curve2_y = np.sin(t)
curve2_z = (3 * curve2_x + 3 * curve2_y) / 5
magnitude = np.sqrt(curve2_x**2 + curve2_y**2 + curve2_z**2)
curve2_x /= magnitude
curve2_y /= magnitude
curve2_z /= magnitude

# Project curves using stereographic projection
curve1_proj_x, curve1_proj_y = stereographic_projection(curve1_x, curve1_y, curve1_z)
curve2_proj_x, curve2_proj_y = stereographic_projection(curve2_x, curve2_y, curve2_z)

# Plot the curves on the sphere
fig1 = plt.figure(figsize=(10, 7))
ax3d = fig1.add_subplot(111, projection='3d')
ax3d.plot_surface(x, y, z, color='lightblue', alpha=0.3, edgecolor='grey')
ax3d.plot(curve1_x, curve1_y, curve1_z, label='Latitude (Curve 1)', color='blue')
ax3d.plot(curve2_x, curve2_y, curve2_z, label='Helical Path (Curve 2)', color='red')
ax3d.set_title('Curves on the Unit Sphere', fontsize=14)
ax3d.set_xlabel('$x$', fontsize=12)
ax3d.set_ylabel('$y$', fontsize=12)
ax3d.set_zlabel('$z$', fontsize=12)
ax3d.legend()
plt.show()
plt.savefig("a1.png")

# Plot the curves in the projection plane
fig2 = plt.figure(figsize=(10, 7))
ax2 = fig2.add_subplot(111)
ax2.plot(curve1_proj_x, curve1_proj_y, label='Projected Latitude (Curve 1)', color='blue')
ax2.plot(curve2_proj_x, curve2_proj_y, label='Projected Helical Path (Curve 2)', color='red')
ax2.set_title('Stereographic Projection of Curves', fontsize=14)
ax2.set_xlabel("$x'$", fontsize=12)
ax2.set_ylabel("$y'$", fontsize=12)
ax2.set_aspect('equal')
ax2.legend()
plt.show()
plt.savefig("a1.png")