import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PARAMETERS
# =========================================================

Nx = 200
Ny = 60

v_char = 0.01 # характеристична швидкість
Re = 20 # задане числом Рейнольдса

L = Ny - 2 # верхній і нижній ряд — це стінки

# viscosity from Reynolds number
viscosity = v_char * L / Re

# relaxation time
tau = 3.0 * viscosity + 0.5

print("viscosity  =", viscosity)
print("tau =", tau)

steps = 5000
plot_every = 200

# =========================================================
# D2Q9 LATTICE
# =========================================================


# allowed directions
e = np.array([
    [ 0, 0],
    [ 1, 0],
    [ 0, 1],
    [-1, 0],
    [ 0,-1],
    [ 1, 1],
    [-1, 1],
    [-1,-1],
    [ 1,-1]
])

w = np.array([
    4/9,
    1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36
])

# opposite directions
opp = np.array([0,3,4,1,2,7,8,5,6])

# =========================================================
# INITIAL MACROSCOPIC FIELDS
# =========================================================

rho = np.ones((Ny, Nx))

ux = np.zeros((Ny, Nx))
uy = np.zeros((Ny, Nx))

# =========================================================
# INITIAL DISTRIBUTION
# f[y, x, i]
# =========================================================

f = np.zeros((Ny, Nx, 9))

u2 = ux**2 + uy**2

for i in range(9):

    eu = e[i,0]*ux + e[i,1]*uy

    f[:,:,i] = w[i] * rho * (
        1
        + 3*eu
        + 4.5*eu**2
        - 1.5*u2
    )

# =========================================================
# WALLS
# =========================================================

solid = np.zeros((Ny, Nx), dtype=bool)

solid[0,:] = True
solid[-1,:] = True

# =====================================================
# circle obstacle
# =====================================================

cx = Nx // 2
cy = Ny // 2

radius = 10

Y, X = np.ogrid[:Ny, :Nx]

circle = (
    (X - cx)**2 +
    (Y - cy)**2
) <= radius**2

solid[circle] = True

# =========================================================
# MAIN LOOP
# =========================================================

plt.ion()

for step in range(steps):

    # -----------------------------------------------------
    # MACROSCOPIC VARIABLES
    # -----------------------------------------------------

    rho = np.sum(f, axis=2)

    ux = np.sum(f * e[:,0], axis=2) / rho
    uy = np.sum(f * e[:,1], axis=2) / rho

    # -----------------------------------------------------
    # EQUILIBRIUM
    # -----------------------------------------------------

    feq = np.zeros_like(f)

    u2 = ux**2 + uy**2

    for i in range(9):

        eu = e[i,0]*ux + e[i,1]*uy

        feq[:,:,i] = w[i] * rho * (
            1
            + 3*eu
            + 4.5*eu**2
            - 1.5*u2
        )

    # -----------------------------------------------------
    # COLLISION
    # -----------------------------------------------------

    f += -(f - feq) / tau

    # -----------------------------------------------------
    # STREAMING
    # -----------------------------------------------------

    for i in range(9):

        f[:,:,i] = np.roll(
            np.roll(f[:,:,i], e[i,0], axis=1),
            e[i,1], axis=0
        )

    # =====================================================
    # ZOU-HE INLET (left boundary)
    # =====================================================

    u_in = v_char

    # known populations
    f0 = f[1:-1, 0, 0]
    f2 = f[1:-1, 0, 2]
    f3 = f[1:-1, 0, 3]
    f4 = f[1:-1, 0, 4]
    f6 = f[1:-1, 0, 6]
    f7 = f[1:-1, 0, 7]

    # compute density
    rho_in = (
                     f0 + f2 + f4
                     + 2 * (f3 + f6 + f7)
             ) / (1 - u_in)

    # reconstruct unknown populations
    f[1:-1, 0, 1] = (
            f3 + (2 / 3) * rho_in * u_in
    )

    f[1:-1, 0, 5] = (
            f7
            - 0.5 * (f2 - f4)
            + (1 / 6) * rho_in * u_in
    )

    f[1:-1, 0, 8] = (
            f6
            + 0.5 * (f2 - f4)
            + (1 / 6) * rho_in * u_in
    )

    # =====================================================
    # OUTLET (right boundary)
    # =====================================================

    for i in range(9):
        f[1:-1, -1, i] = f[1:-1, -2, i]

    # -----------------------------------------------------
    # BOUNCE BACK
    # -----------------------------------------------------

    for i in range(9):

        bounced = f[:,:,i][solid]

        f[:,:,opp[i]][solid] = bounced

    # -----------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------

    if step % plot_every == 0:

        speed = np.sqrt(ux**2 + uy**2)

        plt.clf()

        plt.imshow(speed, origin="lower")
        plt.colorbar()

        plt.title(
            f"step={step}  tau={tau:.3f}  Re={Re}"
        )

        plt.pause(0.01)

plt.ioff()
plt.show()