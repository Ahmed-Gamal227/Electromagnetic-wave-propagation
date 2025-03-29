import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
c = 1.0  
dx = 0.01  
dt = dx / (2 * c) 
L = 2.0  
N = int(L / dx)  

# Fields
Ez = np.zeros((N, N))  # z-component of the electric field
Hx = np.zeros((N, N))  # x-component of the magnetic field
Hy = np.zeros((N, N))  # y-component of the magnetic field

# Source parameters
source_position = (N // 2, N // 2)  # Source at the center of the grid
frequency = 50.0  # Source frequency
time = 0.0

# Update equations for Maxwell's curl

def update_fields(Ez, Hx, Hy):
    global time
    # Update magnetic fields
    Hx[:-1, :] -= (dt / dx) * (Ez[1:, :] - Ez[:-1, :])
    Hy[:, :-1] += (dt / dx) * (Ez[:, 1:] - Ez[:, :-1])

    # Update electric field
    Ez[1:, 1:] += (dt / dx) * ((Hy[1:, 1:] - Hy[:-1, 1:]) - (Hx[1:, 1:] - Hx[1:, :-1]))

    # Apply source
    Ez[source_position] += np.sin(2 * np.pi * frequency * time) * dt
    time += dt

# Visualization setup
fig, ax = plt.subplots()
im = ax.imshow(Ez, extent=[-L/2, L/2, -L/2, L/2], origin='lower', cmap='RdBu', vmin=-0.1, vmax=0.1)
ax.set_title("Electromagnetic Wave Propagation")
plt.colorbar(im, label='Ez Field Intensity')

def update(frame):
    update_fields(Ez, Hx, Hy)
    im.set_data(Ez)
    return [im]

ani = animation.FuncAnimation(fig, update, frames=200, interval=30, blit=True)
plt.show()