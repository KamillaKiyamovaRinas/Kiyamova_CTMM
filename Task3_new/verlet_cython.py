import numpy as np
from multiprocessing import Pool, cpu_count
from Cython.nbody_cython import compute_accelerations_openmp

# Гравитационная постоянная в км^3 / (кг * месяц^2)
G_SI = 6.67430e-11
SECONDS_PER_MONTH = 2.592e6
G = G_SI * (1e-9) * (SECONDS_PER_MONTH ** 2)



def velocity_verlet_cython1(positions, velocities, masses, t_span, dt):
    t0, t_end = t_span
    steps = int((t_end - t0) / dt)
    trajectory = np.zeros((steps, len(masses), 2))
    accelerations = compute_accelerations_openmp(positions, masses)

    for t in range(steps):
        trajectory[t] = positions

        positions_new = positions + velocities * dt + 0.5 * accelerations * dt**2
        accelerations_new = compute_accelerations_openmp(positions_new, masses)
        velocities_new = velocities + 0.5 * (accelerations + accelerations_new) * dt

        positions = positions_new
        velocities = velocities_new
        accelerations = accelerations_new
    
    return trajectory