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

        self.v_char = v_char  # характеристична швидкість
        self.Re = Re  # задане числом Рейнольдса

        self.L = Ny - 2  # верхній і нижній ряд — це стінки

        self.viscosity = self.v_char * self.L / self.Re  # viscosity from Reynolds number
        self.tau = 3.0 * self.viscosity + 0.5  # relaxation time

        self.steps = 30000
        self.ramp_steps = 200

        print("viscosity =", self.viscosity)
        print("tau =", self.tau)

        # =========================================================
        # D2Q9 LATTICE
        # =========================================================

        # allowed directions
        self.e = np.array([
            [0, 0],
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1]
        ])

        self.w = np.array([
            4 / 9,
            1 / 9, 1 / 9, 1 / 9, 1 / 9,
            1 / 36, 1 / 36, 1 / 36, 1 / 36
        ])

        # opposite directions
        self.opp = np.array([
            0, 3, 4, 1, 2, 7, 8, 5, 6
        ])

        self.solid_mask = None

        self.fluid_mask = None
        self.cbc_fluid_mask = None

        self.inlet_mask = None
        self.outlet_mask = None

        self.initialize_masks()

        self.ux = None
        self.uy = None
        self.rho = None
        self.initialize_macroscopic()

        self.f = None
        self.initialize_distribution()

        self.cbc_rho_right = np.ones(self.Ny - 2)
        self.cbc_ux_right = np.zeros(self.Ny - 2)
        self.cbc_uy_right = np.zeros(self.Ny - 2)

        cv2.namedWindow("LBM", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("LBM", 1200, 400)

        self.mouse_x = 0
        self.mouse_y = 0

        cv2.setMouseCallback("LBM", self.mouse_callback)

    def initialize_masks(self):
        # =====================================================
        # SOLID
        # =====================================================

        # walls
        self.solid_mask = np.zeros((self.Ny, self.Nx), dtype=bool)

        self.solid_mask[0, :] = True
        self.solid_mask[-1, :] = True

        # circle obstacle
        cx = self.Nx // 2
        cy = self.Ny // 2
        radius = 2

        Y, X = np.ogrid[:self.Ny, :self.Nx]

        circle = (
                         (X - cx) ** 2 +
                         (Y - cy) ** 2
                 ) <= radius ** 2

        self.solid_mask[circle] = True

        # =====================================================
        # FLUID
        # =====================================================
        self.fluid_mask = ~self.solid_mask

        # =====================================================
        # INLET
        # =====================================================
        self.inlet_mask = np.zeros_like(self.solid_mask)
        self.inlet_mask[1:-1, 0] = True

        # =====================================================
        # OUTLET
        # =====================================================
        self.outlet_mask = np.zeros_like(self.solid_mask)
        self.outlet_mask[1:-1, -1] = True

        # =====================================================
        # FLUID CBC
        # =====================================================
        self.cbc_fluid_mask = (
                self.fluid_mask &
                ~self.outlet_mask
        )

    def initialize_macroscopic(self):
        # =====================================================
        # INITIAL MACROSCOPIC FIELDS
        # =====================================================

        self.rho = np.ones((self.Ny, self.Nx))
        self.ux = np.zeros((self.Ny, self.Nx))
        self.uy = np.zeros((self.Ny, self.Nx))

        # self.ux[self.fluid_mask] = self.v_char
        # self.ux[self.inlet_mask] = self.v_char

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
        mask = self.fluid_mask

        self.rho[:, :] = 1.0
        self.ux[:, :] = 0.0
        self.uy[:, :] = 0.0

        rho = np.sum(self.f[mask, :], axis=1)

        self.rho[mask] = rho

        self.ux[mask] = np.sum(self.f[mask, :] * self.e[:, 0], axis=1) / rho
        self.uy[mask] = np.sum(self.f[mask, :] * self.e[:, 1], axis=1) / rho

    def compute_macroscopic_cbc(self):
        mask = self.cbc_fluid_mask

        self.rho[:, :] = 1.0
        self.ux[:, :] = 0.0
        self.uy[:, :] = 0.0

        rho = np.sum(self.f[mask], axis=1)

        self.rho[mask] = rho
        self.ux[mask] = np.sum(self.f[mask] * self.e[:, 0], axis=1) / rho
        self.uy[mask] = np.sum(self.f[mask] * self.e[:, 1], axis=1) / rho

        self.rho[self.outlet_mask] = self.cbc_rho_right
        self.ux[self.outlet_mask] = self.cbc_ux_right
        self.uy[self.outlet_mask] = self.cbc_uy_right

    def collide(self):
        """
        Collision для звичайних outlet BC.
        """

        mask = self.fluid_mask

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

        self.f[mask] += -(self.f[mask] - feq[mask]) / self.tau

    def collide_cbc(self):
        """
        Collision для CBC outlet.
        """

        mask = self.cbc_fluid_mask

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

        self.f[mask] += -(self.f[mask] - feq[mask]) / self.tau

    def stream_old(self):
        for i in range(9):
            self.f[:, :, i] = np.roll(
                np.roll(self.f[:, :, i], self.e[i, 0], axis=1),
                self.e[i, 1], axis=0
            )

    def stream(self):
        f_old = self.f.copy()
        f_new = np.full_like(f_old, np.nan)

        # 0: rest
        f_new[:, :, 0] = f_old[:, :, 0]

        # 1: east  (x + 1)
        f_new[:, 1:, 1] = f_old[:, :-1, 1]

        # 2: north (y + 1)
        f_new[1:, :, 2] = f_old[:-1, :, 2]

        # 3: west  (x - 1)
        f_new[:, :-1, 3] = f_old[:, 1:, 3]

        # 4: south (y - 1)
        f_new[:-1, :, 4] = f_old[1:, :, 4]

        # 5: north-east (x + 1, y + 1)
        f_new[1:, 1:, 5] = f_old[:-1, :-1, 5]

        # 6: north-west (x - 1, y + 1)
        f_new[1:, :-1, 6] = f_old[:-1, 1:, 6]

        # 7: south-west (x - 1, y - 1)
        f_new[:-1, :-1, 7] = f_old[1:, 1:, 7]

        # 8: south-east (x + 1, y - 1)
        f_new[:-1, 1:, 8] = f_old[1:, :-1, 8]

        self.f = f_new

    def bounce_back(self):
        mask = self.solid_mask
        f_old = self.f.copy()

        for i in range(9):
            self.f[:, :, self.opp[i]][mask] = f_old[:, :, i][mask]

    def inlet_zou_he_velocity_left(self, step=None, ramp_steps=None):
        mask = self.inlet_mask

        if step is None or ramp_steps is None:
            u_in = self.v_char
        else:
            u_in = self.v_char * min(1.0, step / ramp_steps)

        f0 = self.f[mask, 0]
        f2 = self.f[mask, 2]
        f3 = self.f[mask, 3]
        f4 = self.f[mask, 4]
        f6 = self.f[mask, 6]
        f7 = self.f[mask, 7]

        rho_in = (
                         f0 + f2 + f4
                         + 2.0 * (f3 + f6 + f7)
                 ) / (1.0 - u_in)

        self.f[mask, 1] = f3 + (2.0 / 3.0) * rho_in * u_in

        self.f[mask, 5] = (
                f7
                - 0.5 * (f2 - f4)
                + (1.0 / 6.0) * rho_in * u_in
        )

        self.f[mask, 8] = (
                f6
                + 0.5 * (f2 - f4)
                + (1.0 / 6.0) * rho_in * u_in
        )

    def outlet_zou_he_pressure_right(self, rho_out=1.0):
        mask = self.outlet_mask

        f0 = self.f[mask, 0]
        f1 = self.f[mask, 1]
        f2 = self.f[mask, 2]
        f4 = self.f[mask, 4]
        f5 = self.f[mask, 5]
        f8 = self.f[mask, 8]

        ux_out = -1 + (
                f0 + f2 + f4
                + 2 * (f1 + f5 + f8)
        ) / rho_out

        self.f[mask, 3] = f1 - (2 / 3) * rho_out * ux_out

        self.f[mask, 6] = (
                f8
                - 0.5 * (f2 - f4)
                - (1 / 6) * rho_out * ux_out
        )

        self.f[mask, 7] = (
                f5
                + 0.5 * (f2 - f4)
                - (1 / 6) * rho_out * ux_out
        )

    def outlet_copy_right(self):
        mask = self.outlet_mask

        for i in range(9):
            self.f[mask, i] = self.f[:, -2, i][mask]

    def outlet_anti_bounce_back_pressure_right(self, rho_out=1.0):
        mask = self.outlet_mask

        rho = np.sum(self.f, axis=2)

        ux = np.sum(self.f * self.e[:, 0], axis=2) / rho
        uy = np.sum(self.f * self.e[:, 1], axis=2) / rho

        cs2 = 1.0 / 3.0
        cs4 = cs2 ** 2

        ux_w = ux[:, -2][mask] + 0.5 * (ux[:, -2][mask] - ux[:, -3][mask])
        uy_w = uy[:, -2][mask] + 0.5 * (uy[:, -2][mask] - uy[:, -3][mask])

        u2_w = ux_w ** 2 + uy_w ** 2

        cu = self.e[3, 0] * ux_w + self.e[3, 1] * uy_w

        self.f[mask, 3] = (
                -self.f[mask, 1]
                + 2 * self.w[3] * rho_out * (
                        1
                        + (cu ** 2) / (2 * cs4)
                        - u2_w / (2 * cs2)
                )
        )

        cu = self.e[6, 0] * ux_w + self.e[6, 1] * uy_w

        self.f[mask, 6] = (
                -self.f[mask, 8]
                + 2 * self.w[6] * rho_out * (
                        1
                        + (cu ** 2) / (2 * cs4)
                        - u2_w / (2 * cs2)
                )
        )

        cu = self.e[7, 0] * ux_w + self.e[7, 1] * uy_w

        self.f[mask, 7] = (
                -self.f[mask, 5]
                + 2 * self.w[7] * rho_out * (
                        1
                        + (cu ** 2) / (2 * cs4)
                        - u2_w / (2 * cs2)
                )
        )

    # def equilibrium(self, rho, ux, uy):
    #     feq = np.zeros((rho.shape[0], rho.shape[1], 9))
    #
    #     u2 = ux ** 2 + uy ** 2
    #
    #     for i in range(9):
    #         eu = self.e[i, 0] * ux + self.e[i, 1] * uy
    #
    #         feq[:, :, i] = self.w[i] * rho * (
    #                 1
    #                 + 3 * eu
    #                 + 4.5 * eu ** 2
    #                 - 1.5 * u2
    #         )
    #
    #     return feq

    def outlet_cbc_right(self):
        """
        CBC / characteristic outlet на правій межі x = Nx - 1.

        Ідея:
        1) беремо збережені макрозмінні на outlet: rho, ux, uy = m(t)
        2) будуємо з них f_eq і записуємо в праву колонку
        3) по внутрішніх вузлах рахуємо просторові похідні
        4) через характеристики прогнозуємо m(t + dt)
        5) зберігаємо його для наступного кроку
        """

        outlet_mask = self.outlet_mask
        y_mask = self.outlet_mask[:, -1]

        cs = 1.0 / np.sqrt(3.0)
        dt = 1.0
        dx = 1.0

        # =====================================================
        # 1. Беремо m(t), яке CBC зберігає окремо
        # =====================================================

        rho_b = self.cbc_rho_right.copy()
        ux_b = self.cbc_ux_right.copy()
        uy_b = self.cbc_uy_right.copy()

        # safety clamp
        rho_b = np.maximum(rho_b, 1e-8)

        # =====================================================
        # 2. З m(t) будуємо f_eq і ставимо в outlet-вузли
        # =====================================================

        u2 = ux_b ** 2 + uy_b ** 2

        for i in range(9):
            eu = self.e[i, 0] * ux_b + self.e[i, 1] * uy_b

            self.f[outlet_mask, i] = self.w[i] * rho_b * (
                    1
                    + 3.0 * eu
                    + 4.5 * eu ** 2
                    - 1.5 * u2
            )

        # =====================================================
        # 3. Рахуємо ∂x rho, ∂x ux, ∂x uy на правій межі
        #    backward difference:
        #    df/dx = (3f_b - 4f_{-1} + f_{-2}) / (2dx)
        # =====================================================

        rho_1 = self.rho[y_mask, -2]
        rho_2 = self.rho[y_mask, -3]

        ux_1 = self.ux[y_mask, -2]
        ux_2 = self.ux[y_mask, -3]

        uy_1 = self.uy[y_mask, -2]
        uy_2 = self.uy[y_mask, -3]

        drho_dx = (3.0 * rho_b - 4.0 * rho_1 + rho_2) / (2.0 * dx)
        dux_dx = (3.0 * ux_b - 4.0 * ux_1 + ux_2) / (2.0 * dx)
        duy_dx = (3.0 * uy_b - 4.0 * uy_1 + uy_2) / (2.0 * dx)

        # =====================================================
        # 4. Характеристики для правого outlet
        #
        # lambda_minus = ux - cs  -> хвиля заходить ззовні в домен
        # lambda_plus  = ux + cs  -> хвиля виходить з домену
        # lambda_0     = ux       -> перенос uy
        #
        # Для non-reflecting outlet:
        # L_minus = 0
        # =====================================================

        lambda_plus = ux_b + cs
        lambda_0 = ux_b

        L_minus = 0.0

        L_plus = lambda_plus * (
                drho_dx + (rho_b / cs) * dux_dx
        )

        L_0 = lambda_0 * duy_dx

        # =====================================================
        # 5. Відновлюємо часові похідні макрозмінних
        # =====================================================

        drho_dt = -0.5 * (L_plus + L_minus)

        dux_dt = -(cs / (2.0 * rho_b)) * (L_plus - L_minus)

        duy_dt = -L_0

        # =====================================================
        # 6. Оновлюємо m(t + dt)
        # =====================================================

        rho_next = rho_b + dt * drho_dt
        ux_next = ux_b + dt * dux_dt
        uy_next = uy_b + dt * duy_dt

        # safety
        rho_next = np.maximum(rho_next, 1e-8)

        # можна трохи обмежити швидкість, щоб не вибухало
        ux_next = np.clip(ux_next, -0.2, 0.2)
        uy_next = np.clip(uy_next, -0.2, 0.2)

        # зберігаємо для наступного кроку
        self.cbc_rho_right = rho_next
        self.cbc_ux_right = ux_next
        self.cbc_uy_right = uy_next

    def visualize_velocity(self):
        speed = np.sqrt(self.ux ** 2 + self.uy ** 2)
        speed[self.solid_mask] = 0.0

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
        field[self.solid_mask] = 1.0

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

    # def print_stats(self, step):
    #     rho_now = np.sum(self.f, axis=2)
    #
    #     print(
    #         step,
    #         "rho mean/min/max",
    #         np.mean(rho_now), np.min(rho_now), np.max(rho_now),
    #         "ux mean/min/max",
    #         np.mean(self.ux), np.min(self.ux), np.max(self.ux),
    #         "rho right mean",
    #         np.mean(rho_now[1:-1, -2]),
    #         "ux right mean/max",
    #         np.mean(self.ux[1:-1, -2]), np.max(self.ux[1:-1, -2])
    #     )

    def print_stats(self, step):
        mask = self.fluid_mask
        outlet_mask = self.outlet_mask

        print(
            step,

            "rho mean/min/max",
            np.mean(self.rho[mask]),
            np.min(self.rho[mask]),
            np.max(self.rho[mask]),

            "ux mean/min/max",
            np.mean(self.ux[mask]),
            np.min(self.ux[mask]),
            np.max(self.ux[mask]),

            "rho outlet mean",
            np.mean(self.rho[outlet_mask]),

            "ux outlet mean/max",
            np.mean(self.ux[outlet_mask]),
            np.max(self.ux[outlet_mask]),
        )


if __name__ == "__main__":

    lbm = LBM(
        Nx=200,
        Ny=60,
        v_char=0.05,
        Re=70,
    )

    for step in range(3000):
        lbm.compute_macroscopic_cbc()
        lbm.outlet_cbc_right()
        lbm.collide_cbc()
        lbm.stream()

        lbm.inlet_zou_he_velocity_left(step, 200)
        # lbm.outlet_copy()
        # lbm.outlet_zou_he_pressure(rho_out=1.0)
        # lbm.outlet_anti_bounce_back_pressure(rho_out=1.0)

        lbm.bounce_back()

        # lbm.compute_macroscopic()

        lbm.visualize_velocity()
        # lbm.visualize_density()

        if step % 50 == 0:
            lbm.print_stats(step)

        if lbm.should_stop():
            break

    cv2.destroyAllWindows()
