import cv2
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)

@ti.data_oriented
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

        self.ux = ti.field(dtype=ti.f32, shape=(self.Ny, self.Nx))
        self.uy = ti.field(dtype=ti.f32, shape=(self.Ny, self.Nx))
        self.rho = ti.field(dtype=ti.f32, shape=(self.Ny, self.Nx))

        self.initialize_macroscopic()

        # =====================================================
        # DISTRIBUTION FUNCTIONS
        # self.f[y, x, i]
        # =====================================================

        self.f = ti.field(dtype=ti.f32, shape=(self.Ny, self.Nx, 9))
        self.f_new = ti.field(dtype=ti.f32, shape=(self.Ny, self.Nx, 9))

        self.initialize_distribution()

        # =====================================================
        # CBC OUTLET STATE
        # =====================================================

        self.cbc_rho_right = ti.field(dtype=ti.f32, shape=self.Ny - 2)
        self.cbc_ux_right = ti.field(dtype=ti.f32, shape=self.Ny - 2)
        self.cbc_uy_right = ti.field(dtype=ti.f32, shape=self.Ny - 2)

        self.cbc_rho_right.fill(1.0)
        self.cbc_ux_right.fill(0.0)
        self.cbc_uy_right.fill(0.0)

        # =====================================================
        # VISUALIZATION
        # =====================================================

        cv2.namedWindow("LBM", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("LBM", 1200, 400)

        self.mouse_x = 0
        self.mouse_y = 0

        cv2.setMouseCallback("LBM", self.mouse_callback)

    @ti.kernel
    def initialize_masks(self):
        # =====================================================
        # SOLID
        # =====================================================

        self.solid_mask.fill(0.0)

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

        self.inlet_mask.fill(0.0)

        for y in range(1, self.Ny - 1):
            self.inlet_mask[y, 0] = 1

        # =====================================================
        # OUTLET
        # =====================================================


        self.outlet_mask.fill(0.0)

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