import numpy as np

# Гравитационная постоянная в км^3 / (кг * месяц^2)
G_SI = 6.67430e-11
SECONDS_PER_MONTH = 2.592e6
G = G_SI * (1e-9) * (SECONDS_PER_MONTH ** 2)


def compute_accelerations(positions, masses):
    N = len(masses)
    acc = np.zeros_like(positions)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r_vec = positions[j] - positions[i]
            dist = np.linalg.norm(r_vec)
            if dist == 0:
                continue
            acc[i] += G * masses[j] * r_vec / (dist**3)

    return acc

def velocity_verlet(positions, velocities, masses, t_span, dt):
    t0, t_end = t_span
    steps = int((t_end - t0) / dt)
    trajectory = np.zeros((steps, len(masses), 2))
    accelerations = compute_accelerations(positions, masses)

    for t in range(steps):
        trajectory[t] = positions

        positions_new = positions + velocities * dt + 0.5 * accelerations * dt**2
        accelerations_new = compute_accelerations(positions_new, masses)
        velocities_new = velocities + 0.5 * (accelerations + accelerations_new) * dt

        positions = positions_new
        velocities = velocities_new
        accelerations = accelerations_new
    
    return trajectory