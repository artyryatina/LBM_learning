import cv2
import numpy as np
import taichi as ti

@ti.data_oriented
class LBMTaichi:
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

        # print("viscosity =", self.viscosity)
        # print("tau =", self.tau)

        # =========================================================
        # D2Q9 LATTICE
        # =========================================================

        # allowed directions
        self.e = [
            (0, 0),
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1),
            (1, 1),
            (-1, 1),
            (-1, -1),
            (1, -1),
        ]

        self.w = [
            4 / 9,
            1 / 9, 1 / 9, 1 / 9, 1 / 9,
            1 / 36, 1 / 36, 1 / 36, 1 / 36,
        ]

        # opposite directions
        self.opp = [
            0, 3, 4, 1, 2, 7, 8, 5, 6
        ]

        # =====================================================
        # MASKS
        # =====================================================

        self.solid_mask = ti.field(dtype=ti.i32, shape=(self.Ny, self.Nx))

        self.fluid_mask = ti.field(dtype=ti.i32, shape=(self.Ny, self.Nx))
        self.cbc_fluid_mask = ti.field(dtype=ti.i32, shape=(self.Ny, self.Nx))

        self.inlet_mask = ti.field(dtype=ti.i32, shape=(self.Ny, self.Nx))
        self.outlet_mask = ti.field(dtype=ti.i32, shape=(self.Ny, self.Nx))

        self.initialize_masks()

        # =====================================================
        # MACROSCOPIC FIELDS
        # =====================================================

        self.ux = ti.field(dtype=ti.f64, shape=(self.Ny, self.Nx))
        self.uy = ti.field(dtype=ti.f64, shape=(self.Ny, self.Nx))
        self.rho = ti.field(dtype=ti.f64, shape=(self.Ny, self.Nx))

        self.initialize_macroscopic()

        # =====================================================
        # DISTRIBUTION FUNCTIONS
        # self.f[y, x, i]
        # =====================================================

        self.f = ti.field(dtype=ti.f64, shape=(self.Ny, self.Nx, 9))
        self.f_temp = ti.field(dtype=ti.f64, shape=(self.Ny, self.Nx, 9))

        self.initialize_distribution()

        # =====================================================
        # CBC OUTLET STATE
        # =====================================================

        self.cbc_rho_right = ti.field(dtype=ti.f64, shape=self.Ny)
        self.cbc_ux_right = ti.field(dtype=ti.f64, shape=self.Ny)
        self.cbc_uy_right = ti.field(dtype=ti.f64, shape=self.Ny)

        self.cbc_rho_right.fill(1.0)
        self.cbc_ux_right.fill(0.0)
        self.cbc_uy_right.fill(0.0)

        # =====================================================
        # VISUALIZATION
        # =====================================================

        cv2.namedWindow("LBMTaichi", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("LBMTaichi", 1200, 400)

        self.mouse_x = 0
        self.mouse_y = 0

        cv2.setMouseCallback("LBMTaichi", self.mouse_callback)

    @ti.kernel
    def initialize_masks(self):
        # =====================================================
        # SOLID
        # =====================================================

        self.solid_mask.fill(0)

        # walls
        for x in range(self.Nx):
            self.solid_mask[0, x] = 1
            self.solid_mask[self.Ny - 1, x] = 1

        # circle obstacle
        cx = self.Nx // 2
        cy = self.Ny // 2
        radius = 2

        for y, x in self.solid_mask:
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                self.solid_mask[y, x] = 1

        # =====================================================
        # FLUID
        # =====================================================

        for y, x in self.fluid_mask:
            self.fluid_mask[y, x] = 1 - self.solid_mask[y, x]

        # =====================================================
        # INLET
        # =====================================================

        self.inlet_mask.fill(0)

        for y in range(1, self.Ny - 1):
            self.inlet_mask[y, 0] = 1

        # =====================================================
        # OUTLET
        # =====================================================


        self.outlet_mask.fill(0)

        for y in range(1, self.Ny - 1):
            self.outlet_mask[y, self.Nx - 1] = 1

        # =====================================================
        # FLUID CBC
        # =====================================================

        for y, x in self.cbc_fluid_mask:
            self.cbc_fluid_mask[y, x] = (
                    self.fluid_mask[y, x] &
                    (1 - self.outlet_mask[y, x])
            )

    @ti.kernel
    def initialize_macroscopic(self):
        # =====================================================
        # INITIAL MACROSCOPIC FIELDS
        # =====================================================

        self.rho.fill(1.0)
        self.ux.fill(0.0)
        self.uy.fill(0.0)

        # for y, x in self.rho:
        #     self.rho[y, x] = 1.0
        #     self.ux[y, x] = 0.0
        #     self.uy[y, x] = 0.0
        #
        #     # self.ux[self.fluid_mask] = self.v_char
        #     if self.fluid_mask[y, x] == 1:
        #         # self.ux[y, x] = self.v_char
        #         pass
        #
        #     # self.ux[self.inlet_mask] = self.v_char
        #     if self.inlet_mask[y, x] == 1:
        #         # self.ux[y, x] = self.v_char
        #         pass

    @ti.kernel
    def initialize_distribution(self):
        # =====================================================
        # INITIAL DISTRIBUTION
        # self.f[y, x, i]
        # =====================================================

        for y, x in self.rho:

            u2 = self.ux[y, x] ** 2 + self.uy[y, x] ** 2

            for i in ti.static(range(9)): # ПРИБРАТИ ti.static?
                eu = self.e[i][0] * self.ux[y, x] + self.e[i][1] * self.uy[y, x]

                self.f[y, x, i] = self.w[i] * self.rho[y, x] * (
                        1.0
                        + 3.0 * eu
                        + 4.5 * eu ** 2
                        - 1.5 * u2
                )

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y

    @ti.kernel
    def compute_macroscopic_cbc(self):
        # =====================================================
        # COMPUTE MACROSCOPIC FIELDS ONLY IN CBC FLUID NODES
        # =====================================================

        for y, x in self.rho:
            if self.cbc_fluid_mask[y, x] == 1:
                rho_local = 0.0
                ux_local = 0.0
                uy_local = 0.0

                for i in ti.static(range(9)):
                    fi = self.f[y, x, i]

                    rho_local += fi
                    ux_local += fi * self.e[i][0]
                    uy_local += fi * self.e[i][1]

                self.rho[y, x] = rho_local
                self.ux[y, x] = ux_local / rho_local
                self.uy[y, x] = uy_local / rho_local

        # =====================================================
        # SET OUTLET MACROSCOPIC FIELDS FROM CBC STATE
        # =====================================================

        for y, x in self.outlet_mask:
            if self.outlet_mask[y, x] == 1:
                self.rho[y, x] = self.cbc_rho_right[y]
                self.ux[y, x] = self.cbc_ux_right[y]
                self.uy[y, x] = self.cbc_uy_right[y]

    @ti.kernel
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

        cs = 1.0 / ti.sqrt(3.0)
        dt = 1.0
        dx = 1.0

        x = self.Nx - 1 # частковий випадок стінка справа в 1 рядок

        for y in range(self.Ny):
            if self.outlet_mask[y, x] == 1:

                # =====================================================
                # 1. Беремо m(t), яке CBC зберігає окремо
                # =====================================================

                rho_b = self.cbc_rho_right[y]
                ux_b = self.cbc_ux_right[y]
                uy_b = self.cbc_uy_right[y]

                # safety clamp
                if rho_b < 1e-8:
                    rho_b = 1e-8

                # =====================================================
                # 2. З m(t) будуємо f_eq і ставимо в outlet-вузол
                # =====================================================

                u2 = ux_b * ux_b + uy_b * uy_b

                for i in ti.static(range(9)):
                    eu = (
                            self.e[i][0] * ux_b +
                            self.e[i][1] * uy_b
                    )

                    self.f[y, x, i] = self.w[i] * rho_b * (
                            1.0
                            + 3.0 * eu
                            + 4.5 * eu * eu
                            - 1.5 * u2
                    )

                # =====================================================
                # 3. Рахуємо ∂x rho, ∂x ux, ∂x uy на правій межі
                #    backward difference:
                #    df/dx = (3f_b - 4f_{-1} + f_{-2}) / (2dx)
                # =====================================================

                # rho_1 = self.rho[y, x - 1]
                # rho_2 = self.rho[y, x - 2]
                #
                # ux_1 = self.ux[y, x - 1]
                # ux_2 = self.ux[y, x - 2]
                #
                # uy_1 = self.uy[y, x - 1]
                # uy_2 = self.uy[y, x - 2]

                rho_1 = self.rho[y, self.Nx - 2]
                rho_2 = self.rho[y, self.Nx - 3]

                ux_1 = self.ux[y, self.Nx - 2]
                ux_2 = self.ux[y, self.Nx - 3]

                uy_1 = self.uy[y, self.Nx - 2]
                uy_2 = self.uy[y, self.Nx - 3]

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
                if rho_next < 1e-8:
                    rho_next = 1e-8

                # можна трохи обмежити швидкість, щоб не вибухало
                ux_next = ti.max(-0.2, ti.min(0.2, ux_next))
                uy_next = ti.max(-0.2, ti.min(0.2, uy_next))

                # зберігаємо для наступного кроку
                self.cbc_rho_right[y] = rho_next
                self.cbc_ux_right[y] = ux_next
                self.cbc_uy_right[y] = uy_next

    @ti.kernel
    def collide_cbc(self):
        """
        Collision для CBC outlet.
        """

        for y, x in self.rho:
            if self.cbc_fluid_mask[y, x] == 1:

                u2 = (
                        self.ux[y, x] * self.ux[y, x] +
                        self.uy[y, x] * self.uy[y, x]
                )

                for i in ti.static(range(9)):
                    eu = (
                            self.e[i][0] * self.ux[y, x] +
                            self.e[i][1] * self.uy[y, x]
                    )

                    feq = self.w[i] * self.rho[y, x] * (
                            1.0
                            + 3.0 * eu
                            + 4.5 * eu * eu
                            - 1.5 * u2
                    )

                    self.f[y, x, i] += -(self.f[y, x, i] - feq) / self.tau

    @ti.kernel
    def stream(self):
        self.f_temp.fill(float("nan"))

        # 0: rest
        for y, x in ti.ndrange(self.Ny, self.Nx):
            self.f_temp[y, x, 0] = self.f[y, x, 0]

        # 1: east (x + 1)
        for y, x in ti.ndrange(self.Ny, self.Nx - 1):
            self.f_temp[y, x + 1, 1] = self.f[y, x, 1]

        # 2: north (y + 1)
        for y, x in ti.ndrange(self.Ny - 1, self.Nx):
            self.f_temp[y + 1, x, 2] = self.f[y, x, 2]

        # 3: west (x - 1)
        for y, x in ti.ndrange(self.Ny, (1, self.Nx)):
            self.f_temp[y, x - 1, 3] = self.f[y, x, 3]

        # 4: south (y - 1)
        for y, x in ti.ndrange((1, self.Ny), self.Nx):
            self.f_temp[y - 1, x, 4] = self.f[y, x, 4]

        # 5: north-east (x + 1, y + 1)
        for y, x in ti.ndrange(self.Ny - 1, self.Nx - 1):
            self.f_temp[y + 1, x + 1, 5] = self.f[y, x, 5]

        # 6: north-west (x - 1, y + 1)
        for y, x in ti.ndrange(self.Ny - 1, (1, self.Nx)):
            self.f_temp[y + 1, x - 1, 6] = self.f[y, x, 6]

        # 7: south-west (x - 1, y - 1)
        for y, x in ti.ndrange((1, self.Ny), (1, self.Nx)):
            self.f_temp[y - 1, x - 1, 7] = self.f[y, x, 7]

        # 8: south-east (x + 1, y - 1)
        for y, x in ti.ndrange((1, self.Ny), self.Nx - 1):
            self.f_temp[y - 1, x + 1, 8] = self.f[y, x, 8]

        for y, x, i in self.f:
            self.f[y, x, i] = self.f_temp[y, x, i]

    @ti.kernel
    def inlet_zou_he_velocity_left(self, step: ti.i32, ramp_steps: ti.i32):

        u_in = self.v_char

        if step >= 0 and ramp_steps > 0:
            u_in = self.v_char * ti.min(1.0, step / ramp_steps)

        for y, x in self.inlet_mask:
            if self.inlet_mask[y, x] == 1:
                f0 = self.f[y, x, 0]
                f2 = self.f[y, x, 2]
                f3 = self.f[y, x, 3]
                f4 = self.f[y, x, 4]
                f6 = self.f[y, x, 6]
                f7 = self.f[y, x, 7]

                rho_in = (
                                 f0 + f2 + f4
                                 + 2.0 * (f3 + f6 + f7)
                         ) / (1.0 - u_in)

                self.f[y, x, 1] = (
                        f3
                        + (2.0 / 3.0) * rho_in * u_in
                )

                self.f[y, x, 5] = (
                        f7
                        - 0.5 * (f2 - f4)
                        + (1.0 / 6.0) * rho_in * u_in
                )

                self.f[y, x, 8] = (
                        f6
                        + 0.5 * (f2 - f4)
                        + (1.0 / 6.0) * rho_in * u_in
                )

    @ti.kernel
    def bounce_back(self):
        for y, x, i in self.f:
            self.f_temp[y, x, i] = self.f[y, x, i]
        for y, x in self.solid_mask:
            if self.solid_mask[y, x] == 1:
                for i in ti.static(range(9)):
                    self.f[y, x, self.opp[i]] = self.f_temp[y, x, i]

    def visualize_velocity(self):
        ux = self.ux.to_numpy()
        uy = self.uy.to_numpy()
        solid_mask = self.solid_mask.to_numpy().astype(bool)

        speed = np.sqrt(ux ** 2 + uy ** 2)
        speed[solid_mask] = 0.0

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

        cv2.imshow("LBMTaichi", img)
        cv2.waitKey(1)

    def visualize_density(self):
        field = self.rho.to_numpy()
        solid_mask = self.solid_mask.to_numpy().astype(bool)

        field[solid_mask] = 1.0

        img = (field - 0.95) / (1.10 - 0.95)
        img = np.clip(img, 0, 1)
        img = (255 * img).astype(np.uint8)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)

        grid_x = int(self.mouse_x - 1)
        grid_y = int(self.mouse_y - 1)

        grid_x = np.clip(grid_x, 0, self.Nx - 1)
        grid_y = np.clip(grid_y, 0, self.Ny - 1)

        rho_mouse = field[grid_y, grid_x]

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

        cv2.imshow("LBMTaichi", img)
        cv2.waitKey(1)

    def print_stats(self, step):
        rho = self.rho.to_numpy()
        ux = self.ux.to_numpy()

        fluid_mask = self.fluid_mask.to_numpy().astype(bool)
        outlet_mask = self.outlet_mask.to_numpy().astype(bool)

        print(
            step,

            "rho mean/min/max",
            np.mean(rho[fluid_mask]),
            np.min(rho[fluid_mask]),
            np.max(rho[fluid_mask]),

            "ux mean/min/max",
            np.mean(ux[fluid_mask]),
            np.min(ux[fluid_mask]),
            np.max(ux[fluid_mask]),

            "rho outlet mean",
            np.mean(rho[outlet_mask]),

            "ux outlet mean/max",
            np.mean(ux[outlet_mask]),
            np.max(ux[outlet_mask]),
        )

    def should_stop(self):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return True

        if cv2.getWindowProperty("LBMTaichi", cv2.WND_PROP_VISIBLE) < 1:
            return True

        return False


if __name__ == "__main__":
    import time

    steps = 500

    init_start = time.perf_counter()

    ti.init(
        arch=ti.gpu,
        default_fp=ti.f64,
    )

    lbm = LBMTaichi(
        Nx=1600,
        Ny=480,
        v_char=0.05,
        Re=70,
    )

    init_time = time.perf_counter() - init_start

    loop_start = time.perf_counter()

    for step in range(steps):
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

        # if step % 50 == 0:
        #     lbm.print_stats(step)
        #
        # if lbm.should_stop():
        #     break

    ti.sync()

    loop_time = time.perf_counter() - loop_start

    print(f"Initialization: {init_time:.6f} s")
    print(f"Simulation: {loop_time:.6f} s")
    print(f"Per step: {loop_time / steps:.6f} s")

    cv2.destroyAllWindows()