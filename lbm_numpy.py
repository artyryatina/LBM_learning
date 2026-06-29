import cv2
import numpy as np


class LBM:
    def __init__(
        self,
        Nx,
        Ny,
        v_char,
        Re,
    ):
        # =====================================================
        # PARAMETERS
        # =====================================================

        self.Nx = Nx
        self.Ny = Ny

        self.v_char = v_char # характеристична швидкість
        self.Re = Re # задане числом Рейнольдса

        self.L = Ny - 2 # верхній і нижній ряд — це стінки

        self.viscosity = self.v_char * self.L / self.Re # viscosity from Reynolds number
        self.tau = 3.0 * self.viscosity + 0.5 # relaxation time

        self.steps = 30000
        self.ramp_steps = 200

        print("viscosity =", self.viscosity)
        print("tau =", self.tau)

        # =========================================================
        # D2Q9 LATTICE
        # =========================================================

        # allowed directions
        self.e = np.array([
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

        self.w = np.array([
            4/9,
            1/9, 1/9, 1/9, 1/9,
            1/36, 1/36, 1/36, 1/36
        ])

        # opposite directions
        self.opp = np.array([
            0, 3, 4, 1, 2, 7, 8, 5, 6
        ])

        self.solid = None
        self.initialize_solid()

        self.ux = None
        self.uy = None
        self.rho = None
        self.initialize_macroscopic()

        self.f = None
        self.initialize_distribution()

        cv2.namedWindow("LBM", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("LBM", 1200, 400)

        self.mouse_x = 0
        self.mouse_y = 0

        cv2.setMouseCallback("LBM", self.mouse_callback)


    def initialize_solid(self):
        # =====================================================
        # WALLS
        # =====================================================

        self.solid = np.zeros((self.Ny, self.Nx), dtype=bool)

        self.solid[0, :] = True
        self.solid[-1, :] = True

        # =====================================================
        # CIRCLE OBSTACLE
        # =====================================================

        cx = self.Nx // 2
        cy = self.Ny // 2
        radius = 2

        Y, X = np.ogrid[:self.Ny, :self.Nx]

        circle = (
                         (X - cx) ** 2 +
                         (Y - cy) ** 2
                 ) <= radius ** 2

        self.solid[circle] = True

    def initialize_macroscopic(self):
        # =====================================================
        # INITIAL MACROSCOPIC FIELDS
        # =====================================================

        self.rho = np.ones((self.Ny, self.Nx))

        self.ux = np.full((self.Ny, self.Nx), self.v_char)
        self.uy = np.zeros((self.Ny, self.Nx))

        self.ux[self.solid] = 0.0
        self.uy[self.solid] = 0.0

    def initialize_distribution(self):

        # =====================================================
        # INITIAL DISTRIBUTION
        # self.f[y, x, i]
        # =====================================================

        self.f = np.zeros((self.Ny, self.Nx, 9))

        u2 = self.ux ** 2 + self.uy ** 2

        for i in range(9):
            eu = self.e[i, 0] * self.ux + self.e[i, 1] * self.uy

            self.f[:, :, i] = self.w[i] * self.rho * (
                    1
                    + 3 * eu
                    + 4.5 * eu ** 2
                    - 1.5 * u2
            )

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y

    def compute_macroscopic(self):
        self.rho = np.sum(self.f, axis=2)

        self.ux = np.sum(self.f * self.e[:, 0], axis=2) / self.rho
        self.uy = np.sum(self.f * self.e[:, 1], axis=2) / self.rho

        self.ux[self.solid] = 0.0
        self.uy[self.solid] = 0.0

    def collide(self):
        feq = np.zeros_like(self.f)

        u2 = self.ux ** 2 + self.uy ** 2

        for i in range(9):
            eu = self.e[i, 0] * self.ux + self.e[i, 1] * self.uy

            feq[:, :, i] = self.w[i] * self.rho * (
                    1
                    + 3 * eu
                    + 4.5 * eu ** 2
                    - 1.5 * u2
            )
        self.f += -(self.f - feq) / self.tau

    def stream(self):
        for i in range(9):
            self.f[:, :, i] = np.roll(
                np.roll(self.f[:, :, i], self.e[i, 0], axis=1),
                self.e[i, 1], axis=0
            )

    def bounce_back(self):
        f_old = self.f.copy()

        for i in range(9):
            self.f[:, :, self.opp[i]][self.solid] = f_old[:, :, i][self.solid]

    def inlet_zou_he_velocity(self, u_in=None):
        if u_in is None:
            u_in = self.v_char

        f0 = self.f[1:-1, 0, 0]
        f2 = self.f[1:-1, 0, 2]
        f3 = self.f[1:-1, 0, 3]
        f4 = self.f[1:-1, 0, 4]
        f6 = self.f[1:-1, 0, 6]
        f7 = self.f[1:-1, 0, 7]

        rho_in = (
                         f0 + f2 + f4
                         + 2 * (f3 + f6 + f7)
                 ) / (1 - u_in)

        self.f[1:-1, 0, 1] = f3 + (2 / 3) * rho_in * u_in

        self.f[1:-1, 0, 5] = (
                f7
                - 0.5 * (f2 - f4)
                + (1 / 6) * rho_in * u_in
        )

        self.f[1:-1, 0, 8] = (
                f6
                + 0.5 * (f2 - f4)
                + (1 / 6) * rho_in * u_in
        )

    def outlet_zou_he_pressure(self, rho_out=1.0):
        f0 = self.f[1:-1, -1, 0]
        f1 = self.f[1:-1, -1, 1]
        f2 = self.f[1:-1, -1, 2]
        f4 = self.f[1:-1, -1, 4]
        f5 = self.f[1:-1, -1, 5]
        f8 = self.f[1:-1, -1, 8]

        ux_out = -1 + (
                f0 + f2 + f4
                + 2 * (f1 + f5 + f8)
        ) / rho_out

        self.f[1:-1, -1, 3] = f1 - (2 / 3) * rho_out * ux_out

        self.f[1:-1, -1, 6] = (
                f8
                - 0.5 * (f2 - f4)
                - (1 / 6) * rho_out * ux_out
        )

        self.f[1:-1, -1, 7] = (
                f5
                + 0.5 * (f2 - f4)
                - (1 / 6) * rho_out * ux_out
        )

    def outlet_copy(self):
        for i in range(9):
            self.f[1:-1, -1, i] = self.f[1:-1, -2, i]

    def outlet_anti_bounce_back_pressure(self, rho_out=1.0):
        rho = np.sum(self.f, axis=2)

        ux = np.sum(self.f * self.e[:, 0], axis=2) / rho
        uy = np.sum(self.f * self.e[:, 1], axis=2) / rho

        ux[self.solid] = 0.0
        uy[self.solid] = 0.0

        cs2 = 1.0 / 3.0
        cs4 = cs2 ** 2

        ux_w = ux[1:-1, -2] + 0.5 * (ux[1:-1, -2] - ux[1:-1, -3])
        uy_w = uy[1:-1, -2] + 0.5 * (uy[1:-1, -2] - uy[1:-1, -3])

        u2_w = ux_w ** 2 + uy_w ** 2

        cu = self.e[3, 0] * ux_w + self.e[3, 1] * uy_w

        self.f[1:-1, -1, 3] = (
                -self.f[1:-1, -1, 1]
                + 2 * self.w[3] * rho_out * (
                        1
                        + (cu ** 2) / (2 * cs4)
                        - u2_w / (2 * cs2)
                )
        )

        cu = self.e[6, 0] * ux_w + self.e[6, 1] * uy_w

        self.f[1:-1, -1, 6] = (
                -self.f[1:-1, -1, 8]
                + 2 * self.w[6] * rho_out * (
                        1
                        + (cu ** 2) / (2 * cs4)
                        - u2_w / (2 * cs2)
                )
        )

        cu = self.e[7, 0] * ux_w + self.e[7, 1] * uy_w

        self.f[1:-1, -1, 7] = (
                -self.f[1:-1, -1, 5]
                + 2 * self.w[7] * rho_out * (
                        1
                        + (cu ** 2) / (2 * cs4)
                        - u2_w / (2 * cs2)
                )
        )

    def visualize_velocity(self):
        speed = np.sqrt(self.ux ** 2 + self.uy ** 2)
        speed[self.solid] = 0.0

        img = speed / (2.0 * self.v_char)
        img = np.clip(img, 0, 1)
        img = (255 * img).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

        grid_x = int(self.mouse_x - 1)
        grid_y = int(self.mouse_y - 1)

        grid_x = np.clip(grid_x, 0, self.Nx - 1)
        grid_y = np.clip(grid_y, 0, self.Ny - 1)

        u_mouse = speed[grid_y, grid_x]

        text = f"x={grid_x}, y={grid_y}, u={u_mouse:.3f}"

        cv2.putText(
            img,
            text,
            (0, self.Ny - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )

        cv2.imshow("LBM", img)

    def visualize_density(self):
        field = self.rho.copy()
        field[self.solid] = 1.0

        img = (field - 0.95) / (1.10 - 0.95)
        img = np.clip(img, 0, 1)
        img = (255 * img).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

        grid_x = int(self.mouse_x - 1)
        grid_y = int(self.mouse_y - 1)

        grid_x = np.clip(grid_x, 0, self.Nx - 1)
        grid_y = np.clip(grid_y, 0, self.Ny - 1)

        rho_mouse = self.rho[grid_y, grid_x]

        text = f"x={grid_x}, y={grid_y}, rho={rho_mouse:.4f}"

        cv2.putText(
            img,
            text,
            (0, self.Ny - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )

        cv2.imshow("LBM", img)

    def should_stop(self):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True

        if cv2.getWindowProperty("LBM", cv2.WND_PROP_VISIBLE) < 1:
            return True

        return False

    def print_stats(self, step):
        rho_now = np.sum(self.f, axis=2)

        print(
            step,
            "rho mean/min/max",
            np.mean(rho_now), np.min(rho_now), np.max(rho_now),
            "ux mean/min/max",
            np.mean(self.ux), np.min(self.ux), np.max(self.ux),
            "rho right mean",
            np.mean(rho_now[1:-1, -2]),
            "ux right mean/max",
            np.mean(self.ux[1:-1, -2]), np.max(self.ux[1:-1, -2])
        )

if __name__ == "__main__":

    lbm = LBM(
        Nx=200,
        Ny=60,
        v_char=0.01,
        Re=7,
    )

    for step in range(10000):
        lbm.compute_macroscopic()
        lbm.collide()
        lbm.stream()

        lbm.inlet_zou_he_velocity()
        lbm.outlet_copy()
        # lbm.outlet_zou_he_pressure(rho_out=1.0)
        # lbm.outlet_anti_bounce_back_pressure(rho_out=1.0)

        lbm.bounce_back()

        lbm.compute_macroscopic()

        # lbm.visualize_velocity()
        lbm.visualize_density()

        if step % 50 == 0:
            lbm.print_stats(step)

        if lbm.should_stop():
            break

    cv2.destroyAllWindows()