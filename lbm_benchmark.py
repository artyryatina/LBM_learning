import time

import cv2
import taichi as ti

from lbm_numpy import LBMNumPy
from lbm_taichi import LBMTaichi


# =====================================================
# PARAMETERS
# =====================================================

grid_sizes = [
    (200, 60),
    (400, 120),
    (800, 240),
    (1600, 480),
    (3200, 920),
]

v_char = 0.05
Re = 70

steps = 500
ramp_steps = 200

# =====================================================
# IMPLEMENTATIONS
# =====================================================

implementations = [
    LBMNumPy,
    LBMTaichi,
]

all_results = []

# =====================================================
# RUN
# =====================================================

for Nx, Ny in grid_sizes:

    results = {}

    for LBMClass in implementations:

        name = LBMClass.__name__

        print(f"GRID: {Nx} x {Ny} - {name}")

        init_start = time.perf_counter()

        if LBMClass is LBMTaichi:
            ti.init(
                arch=ti.gpu,
                default_fp=ti.f64,
            )

        lbm = LBMClass(
            Nx=Nx,
            Ny=Ny,
            v_char=v_char,
            Re=Re,
        )

        init_time = time.perf_counter() - init_start

        loop_start = time.perf_counter()

        for step in range(steps):
            lbm.compute_macroscopic_cbc()
            lbm.outlet_cbc_right()
            lbm.collide_cbc()
            lbm.stream()

            lbm.inlet_zou_he_velocity_left(step, ramp_steps)
            lbm.bounce_back()

            lbm.visualize_velocity()

        if LBMClass is LBMTaichi:
            ti.sync()

        loop_time = time.perf_counter() - loop_start

        print(f"Simulation: {loop_time:.6f} s\n")

        results[name] = {
            "init": init_time,
            "loop": loop_time,
        }

        cv2.destroyAllWindows()

        if LBMClass is LBMTaichi:
            ti.reset()

    numpy_time = results["LBMNumPy"]["loop"]
    taichi_time = results["LBMTaichi"]["loop"]
    speedup = numpy_time / taichi_time

    all_results.append(
        (Nx, Ny, numpy_time, taichi_time, speedup)
    )

# =====================================================
# RESULTS
# =====================================================

print()
print(
    f"{'Grid':<15}"
    f"{'NumPy, s':>12}"
    f"{'Taichi, s':>12}"
    f"{'Speedup':>12}"
)

print("-" * 51)

for Nx, Ny, numpy_time, taichi_time, speedup in all_results:
    print(
        f"{Nx} x {Ny:<8}"
        f"{numpy_time:>12.3f}"
        f"{taichi_time:>12.3f}"
        f"{speedup:>11.2f}x"
    )