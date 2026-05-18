import numpy as np
from multiprocessing import Pool, cpu_count

# Гравитационная постоянная в км^3 / (кг * месяц^2)
G_SI = 6.67430e-11
SECONDS_PER_MONTH = 2.592e6
G = G_SI * (1e-9) * (SECONDS_PER_MONTH ** 2)


def compute_acc_for_body(i, positions, masses):
    N = len(masses)
    acc_i = np.zeros_like(positions[i])

    for j in range(N):
        if i == j:
            continue
        r_vec = positions[j] - positions[i]
        dist = np.linalg.norm(r_vec)
        if dist == 0:
            continue
        acc_i += G * masses[j] * r_vec / (dist**3)

    return acc_i

def compute_accelerations_parallel(positions, masses, n_proc=8):
    N = len(masses)

    args = [(i, positions, masses) for i in range(N)]

    with Pool(processes=n_proc) as pool:
        acc_list = pool.starmap(compute_acc_for_body, args)

    return np.array(acc_list)

def velocity_verlet_parallel(positions, velocities, masses, t_span, dt, n_proc=8):
    t0, t_end = t_span
    steps = int((t_end - t0) / dt)
    trajectory = np.zeros((steps, len(masses), 2))
    accelerations = compute_accelerations_parallel(positions, masses, n_proc)

    for t in range(steps):
        trajectory[t] = positions

        positions_new = positions + velocities * dt + 0.5 * accelerations * dt**2
        accelerations_new = compute_accelerations_parallel(positions_new, masses, n_proc)
        velocities_new = velocities + 0.5 * (accelerations + accelerations_new) * dt

        positions = positions_new
        velocities = velocities_new
        accelerations = accelerations_new
    
    return trajectory