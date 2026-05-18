import numpy as np
import time
import matplotlib as plt

import sys
sys.path.insert(0, 'C:/Users/kkiya/Projects/ctmmTask3_new')

from verlet_computations import velocity_verlet
from verlet_parallel import velocity_verlet_parallel
from verlet_opencl2 import velocity_verlet_gpu
from verlet_cython import velocity_verlet_cython1
# from Cython.verlet_cython import velocity_verlet_cython

def compare_trajectories(traj1, traj2, n_bodies):
    """Корректное сравнение траекторий"""
    
    # Проверка размерностей
    if traj1.shape != traj2.shape:
        print(f"Разные размерности: {traj1.shape} vs {traj2.shape}")
        return None
    
    steps = traj1.shape[0]
    
    # 1. Относительная ошибка на каждом шаге (среднеквадратичная)
    step_errors = []
    for step in range(steps):
        # Пропускаем начальный шаг (он должен быть идентичным)
        if step == 0:
            continue
        diff_step = traj1[step] - traj2[step]
        norm_diff = np.linalg.norm(diff_step)
        norm_traj = np.linalg.norm(traj1[step])
        
        if norm_traj > 0:
            rel_error = norm_diff / norm_traj
        else:
            rel_error = norm_diff
        step_errors.append(rel_error)
    
    # 2. Максимальное отклонение для любого тела на любом шаге
    max_abs_diff = np.max(np.abs(traj1 - traj2))
    
    # 3. Средняя относительная ошибка (исключая начальный шаг)
    avg_rel_error = np.mean(step_errors) if step_errors else 0
    
    # 4. Соответствие конечного состояния
    final_pos1 = traj1[-1]
    final_pos2 = traj2[-1]
    final_diff = np.linalg.norm(final_pos1 - final_pos2)
    final_rel_error = final_diff / np.linalg.norm(final_pos1) if np.linalg.norm(final_pos1) > 0 else final_diff
    
    return {
        'avg_rel_error': avg_rel_error,
        'max_abs_diff': max_abs_diff,
        'final_abs_diff': final_diff,
        'final_rel_error': final_rel_error,
        'step_errors': step_errors
    }


if __name__ == "__main__":
    # ==== параметры ====
    Time = np.zeros((4, 5))
    SPACE_SCALE = 1e6
    VEL_SCALE = 1e-1
    MASS_SCALE = 1e22

    t_span = (0, 10)
    dt = 0.1
    i = 0
    for N in range(100, 600, 100):
        print(f"N = {N}")

        # ==== генерация данных ====
        np.random.seed(42)

        positions = np.random.uniform(-SPACE_SCALE, SPACE_SCALE, size=(N, 2))
        velocities = np.random.uniform(-VEL_SCALE, VEL_SCALE, size=(N, 2))
        masses = np.random.uniform(0.5, 1.5, size=N) * MASS_SCALE

        # ==== копии для честного сравнения ====
        positions_1 = positions.copy()
        velocities_1 = velocities.copy()

        positions_2 = positions.copy()
        velocities_2 = velocities.copy()

        positions_3 = positions.copy()
        velocities_3 = velocities.copy()

        positions_4 = positions.copy()
        velocities_4 = velocities.copy()

        # ==== обычная версия ====
        start = time.perf_counter()
        trajectory_1 = velocity_verlet(positions_1, velocities_1, masses, t_span, dt)
        time_serial = time.perf_counter() - start

        print(f"Без параллельности: {time_serial:.4f} сек")
        Time[0, i] = time_serial

        # ==== параллельная версия ====
        start = time.perf_counter()
        trajectory_2 = velocity_verlet_parallel(positions_2, velocities_2, masses, t_span, dt)
        time_parallel = time.perf_counter() - start

        print(f"С параллельностью: {time_parallel:.4f} sec")
        Time[1, i] = time_parallel

        # ==== cython-версия ====

        start = time.perf_counter()
        trajectory_3 = velocity_verlet_cython1(positions_3, velocities_3, masses, t_span, dt)
        time_cython = time.perf_counter() - start

        print(f"С Cython: {time_cython:.4f} sec")
        Time[2, i] = time_cython

        # ==== opencl-версия ====

        start = time.perf_counter()
        trajectory_4 = velocity_verlet_gpu(positions_4, velocities_4, masses, t_span, dt)
        time_opencl = time.perf_counter() - start

        print(f"С OpenCL и GPU: {time_opencl:.4f} sec")
        Time[3, i] = time_opencl

        # ==== ускорение ====
        speedup1 = time_serial / time_parallel
        print(f"Ускорение 1: {speedup1:.2f}x")

        speedup2 = time_serial / time_cython
        print(f"Ускорение 2: {speedup2:.2f}x")

        
        # === проверка ===
        comparison = compare_trajectories(trajectory_1, trajectory_3, N)
        
        if comparison:
            print("\n=== СРАВНЕНИЕ ТРАЕКТОРИЙ С CYTHON ===")
            print(f"Средняя относительная ошибка (без step 0): {comparison['avg_rel_error']:.6e}")
            print(f"Максимальная абсолютная разница: {comparison['max_abs_diff']:.6e}")
            print(f"Конечное состояние - абсолютная разница: {comparison['final_abs_diff']:.6e}")
            print(f"Конечное состояние - относительная ошибка: {comparison['final_rel_error']:.6e}")

        # === проверка ===
        comparison = compare_trajectories(trajectory_1, trajectory_4, N)
        
        if comparison:
            print("\n=== СРАВНЕНИЕ ТРАЕКТОРИЙ С OPENCL===")
            print(f"Средняя относительная ошибка (без step 0): {comparison['avg_rel_error']:.6e}")
            print(f"Максимальная абсолютная разница: {comparison['max_abs_diff']:.6e}")
            print(f"Конечное состояние - абсолютная разница: {comparison['final_abs_diff']:.6e}")
            print(f"Конечное состояние - относительная ошибка: {comparison['final_rel_error']:.6e}")

        i += 1
    

    plt.figure(figsize=(10, 12))
    for i in range(5):
        plt.plot(N, Time[i, :], marker='o')

    Legend = ["Стандартный алгоритм", "Параллельный алгоритм", "Cython", "OpenCL"]
    plt.grid()
    plt.title("Сравнение скорости работы разных реализаций метода Верле")
    plt.xlabel('Количество тел')
    plt.ylabel('Время выполнения, сек')
    plt.legend(Legend)
    plt.show()
