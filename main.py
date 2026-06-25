import numpy as np
import matplotlib.pyplot as plt
import cv2

# =========================================================
# PARAMETERS
# =========================================================

Nx = 200
Ny = 60

v_char = 0.01 # характеристична швидкість
Re = 7 # задане числом Рейнольдса

L = Ny - 2 # верхній і нижній ряд — це стінки

# viscosity from Reynolds number
viscosity = v_char * L / Re

# relaxation time
tau = 3.0 * viscosity + 0.5

print("viscosity  =", viscosity)
print("tau =", tau)

steps = 3000
plot_every = 1
ramp_steps = 200

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

radius = 8

Y, X = np.ogrid[:Ny, :Nx]

circle = (
    (X - cx)**2 +
    (Y - cy)**2
) <= radius**2

solid[circle] = True

# =========================================================
# MAIN LOOP
# =========================================================
cv2.namedWindow("LBM", cv2.WINDOW_NORMAL)
cv2.resizeWindow("LBM", 1200, 400)

mouse_x = 0
mouse_y = 0

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y

cv2.setMouseCallback("LBM", mouse_callback)

for step in range(steps):

    # -----------------------------------------------------
    # MACROSCOPIC VARIABLES
    # -----------------------------------------------------

    rho = np.sum(f, axis=2)

    ux = np.sum(f * e[:,0], axis=2) / rho
    uy = np.sum(f * e[:,1], axis=2) / rho
    ux[solid] = 0
    uy[solid] = 0

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

    # if step % 250 == 0:
    #     print("after STREAMING", np.mean(np.sum(f, axis=2)))
    # =====================================================
    # ZOU-HE INLET (left boundary)
    # =====================================================

    u_in = v_char
    # u_in = v_char * min(1.0, step / ramp_steps)

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
    # if step % 250 == 0:
    #     print("after ZOU-HE INLET (left boundary)", np.mean(np.sum(f, axis=2)))

    # =====================================================
    # OUTLET (right boundary)
    # =====================================================

    rho_out = 1.0

    f0 = f[1:-1, -1, 0]
    f1 = f[1:-1, -1, 1]
    f2 = f[1:-1, -1, 2]
    f4 = f[1:-1, -1, 4]
    f5 = f[1:-1, -1, 5]
    f8 = f[1:-1, -1, 8]

    ux_out = -1 + (f0 + f2 + f4 + 2 * (f1 + f5 + f8)) / rho_out

    f[1:-1, -1, 3] = f1 - (2 / 3) * rho_out * ux_out
    f[1:-1, -1, 6] = f8 - 0.5 * (f2 - f4) - (1 / 6) * rho_out * ux_out
    f[1:-1, -1, 7] = f5 + 0.5 * (f2 - f4) - (1 / 6) * rho_out * ux_out

    # for i in range(9):
    #     f[1:-1, -1, i] = f[1:-1, -2, i]

    # =====================================================
    # OUTLET (right boundary) - Anti-bounce-back pressure
    # =====================================================

    # rho = np.sum(f, axis=2)
    #
    # ux = np.sum(f * e[:, 0], axis=2) / rho
    # uy = np.sum(f * e[:, 1], axis=2) / rho
    # ux[solid] = 0
    # uy[solid] = 0
    #
    # rho_out = 1.0
    #
    # cs2 = 1.0 / 3.0
    # cs4 = cs2 ** 2
    #
    # # оцінка швидкості на межі через екстраполяцію зсередини
    # ux_w = ux[1:-1, -2] + 0.5 * (ux[1:-1, -2] - ux[1:-1, -3])
    # uy_w = uy[1:-1, -2] + 0.5 * (uy[1:-1, -2] - uy[1:-1, -3])
    #
    # u2_w = ux_w ** 2 + uy_w ** 2
    #
    # # unknown populations at right outlet: f3, f6, f7
    # # their opposite known populations: f1, f8, f5
    #
    # # f3 opposite f1
    # cu = e[3, 0] * ux_w + e[3, 1] * uy_w
    # f[1:-1, -1, 3] = (
    #         -f[1:-1, -1, 1]
    #         + 2 * w[3] * rho_out * (
    #                 1
    #                 + (cu ** 2) / (2 * cs4)
    #                 - u2_w / (2 * cs2)
    #         )
    # )
    #
    # # f6 opposite f8
    # cu = e[6, 0] * ux_w + e[6, 1] * uy_w
    # f[1:-1, -1, 6] = (
    #         -f[1:-1, -1, 8]
    #         + 2 * w[6] * rho_out * (
    #                 1
    #                 + (cu ** 2) / (2 * cs4)
    #                 - u2_w / (2 * cs2)
    #         )
    # )
    #
    # # f7 opposite f5
    # cu = e[7, 0] * ux_w + e[7, 1] * uy_w
    # f[1:-1, -1, 7] = (
    #         -f[1:-1, -1, 5]
    #         + 2 * w[7] * rho_out * (
    #                 1
    #                 + (cu ** 2) / (2 * cs4)
    #                 - u2_w / (2 * cs2)
    #         )
    # )

    # if step % 250 == 0:
    #     print("after OUTLET (right boundary) ", np.mean(np.sum(f, axis=2)))
    # -----------------------------------------------------
    # BOUNCE BACK
    # -----------------------------------------------------

    f_old = f.copy()

    for i in range(9):
        f[:, :, opp[i]][solid] = f_old[:, :, i][solid]

    # if step % 250 == 0:
    #     print("after BC", np.mean(np.sum(f, axis=2)))

    # -----------------------------------------------------
    # VISUALIZATION
    # -----------------------------------------------------

    speed = np.sqrt(ux ** 2 + uy ** 2)
    speed[solid] = 0

    img = speed / (2.0 * v_char)
    img = np.clip(img, 0, 1)
    img = (255 * img).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

    grid_x = int(mouse_x - 1)
    grid_y = int(mouse_y - 1)

    U_mouse = speed[grid_y, grid_x]

    text = f"x={grid_x}, y={grid_y}, u={U_mouse:.3f}"

    cv2.putText(
        img,
        text,
        (0, Ny - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )

    cv2.imshow("LBM", img)

    # field = rho.copy()
    # field[solid] = 1.0
    #
    # img = (field - 0.95) / (1.10 - 0.95)
    # img = np.clip(img, 0, 1)
    # img = (255 * img).astype(np.uint8)
    # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    #
    # grid_x = int(mouse_x - 1)
    # grid_y = int(mouse_y - 1)
    #
    # rho_mouse = rho[grid_y, grid_x]
    #
    # text = f"x={grid_x}, y={grid_y}, rho={rho_mouse:.4f}"
    #
    # cv2.putText(
    #     img,
    #     text,
    #     (0, Ny - 2),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.4,
    #     (255, 255, 255),
    #     1
    # )
    #
    # cv2.imshow("LBM", img)

    # 1 мс
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.getWindowProperty("LBM", cv2.WND_PROP_VISIBLE) < 1:
        break

    if step % 50 == 0:
        rho_now = np.sum(f, axis=2)
        print(
            step,
            "rho mean/min/max",
            np.mean(rho_now), np.min(rho_now), np.max(rho_now),
            "ux min/max",
            np.min(ux), np.max(ux),
            "rho right mean",
            np.mean(rho_now[1:-1, -2]),
            "ux right mean/max",
            np.mean(ux[1:-1, -2]), np.max(ux[1:-1, -2])
        )

    # if step % 250 == 0:
    #     rho_now = np.sum(f, axis=2)
    #     print("в кінці", step,"степу", np.mean(rho_now), np.min(rho_now), np.max(rho_now), np.max(ux), np.min(ux))
cv2.destroyAllWindows()
