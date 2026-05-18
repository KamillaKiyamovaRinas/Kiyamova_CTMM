from dolfin import *
from mshr import Circle, generate_mesh

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import imageio
import os


def solve_heat_equation(
        R,
        a_coef,
        T,
        dt,
        test_case,
        mesh_density=40):

    # Сетка
    mesh = generate_mesh(Circle(Point(0, 0), R), mesh_density)

    V = FunctionSpace(mesh, "P", 1)

    # Границы
    class GammaD(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] <= 0

    class GammaN(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > 0

    boundaries = MeshFunction(
        "size_t",
        mesh,
        mesh.topology().dim() - 1,
        0
    )

    GammaD().mark(boundaries, 1)
    GammaN().mark(boundaries, 2)

    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)


    # Задача
    u_expr = test_case["u"]
    f_expr = test_case["f"]
    g_expr = test_case["g"]
    h_expr = test_case["h"]
    u0_expr = test_case["u0"]


    # Начальное условие
    u_n = interpolate(
        Expression(
            u0_expr,
            degree=5,
            t=0,
            a=a_coef
        ),
        V
    )

    # Вариационная постановка
    u = TrialFunction(V)
    v = TestFunction(V)

    times = []
    errors_L2 = []
    errors_max = []

    # GIF кадры
    frame_files = []

    t = 0.0
    step = 0

    while t < T + 1e-12:
        t += dt

        # Точное решение
        u_exact = Expression(
            u_expr,
            degree=5,
            t=t,
            a=a_coef
        )

        h = Expression(
            h_expr,
            degree=5,
            t=t,
            a=a_coef
        )

        # Правая часть
        f = Expression(
            f_expr,
            degree=5,
            t=t,
            a=a_coef
        )

        # Нейман
        g = Expression(
            g_expr,
            degree=5,
            t=t,
            a=a_coef,
            R=R
        )

        # Дирихле
        bc = DirichletBC(V, h, boundaries, 1)

        # Неявная схема Эйлера
        a_form = (
                u*v*dx
                + dt*a_coef*dot(grad(u), grad(v))*dx
        )

        L_form = (
                u_n*v*dx
                + dt*f*v*dx
                + dt*g*v*ds(2)
        )

        # Решение
        uh = Function(V)

        solve(a_form == L_form, uh, bc)

        # Ошибки
        u_ref = interpolate(u_exact, V)

        err_L2 = errornorm(u_ref, uh, "L2")

        err_max = np.max(np.abs(
            uh.vector().get_local()
            - u_ref.vector().get_local()
        ))

        times.append(t)
        errors_L2.append(err_L2)
        errors_max.append(err_max)

        print(
            f"t = {t:.3f} | "
            f"L2 = {err_L2:.3e} | "
            f"Max = {err_max:.3e}"
        )

        # Визуализация
        filename = f"frame_{step:04d}.png"

        plot_frame(
            mesh,
            uh,
            u_ref,
            t,
            filename
        )

        frame_files.append(filename)

        # Следующий шаг
        u_n.assign(uh)

        step += 1

    # GIF
    images = []

    for filename in frame_files:
        images.append(imageio.imread(filename))

    gif_name = f"heat_solution_case_{test_case['name']}.gif"

    imageio.mimsave(
        gif_name,
        images,
        fps=5
    )

    print(f"\nGIF saved: {gif_name}")

    # Удаление кадров
    for filename in frame_files:
        os.remove(filename)

    return (
        mesh,
        uh,
        u_ref,
        times,
        errors_L2,
        errors_max
    )



def plot_frame(
        mesh,
        u_num,
        u_ex,
        t,
        filename):

    coords = mesh.coordinates()
    cells = mesh.cells()

    triang = tri.Triangulation(
        coords[:, 0],
        coords[:, 1],
        cells
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Numerical
    t1 = ax[0].tripcolor(
        coords[:, 0],
        coords[:, 1],
        cells,
        u_num.compute_vertex_values(mesh),
        shading='flat',
        edgecolors='k',
        linewidth=0.4
    )

    ax[0].set_title(f"Numerical solution\nt={t:.2f}")
    ax[0].set_aspect("equal")

    plt.colorbar(t1, ax=ax[0])

    # Exact
    t2 = ax[1].tripcolor(
        coords[:, 0],
        coords[:, 1],
        cells,
        u_ex.compute_vertex_values(mesh),
        shading='flat',
        edgecolors='k',
        linewidth=0.4
    )

    ax[1].set_title(f"Exact solution\nt={t:.2f}")
    ax[1].set_aspect("equal")

    plt.colorbar(t2, ax=ax[1])

    plt.tight_layout()

    plt.savefig(filename)

    plt.close()

def plot_errors(times, errL2, errMax, case_name):

    plt.figure(figsize=(8, 5))

    plt.semilogy(
        times,
        errL2,
        label="L2 error",
        linewidth=2
    )

    plt.semilogy(
        times,
        errMax,
        label="Max error",
        linewidth=2
    )

    plt.xlabel("Time")
    plt.ylabel("Error")

    plt.title(f"Errors ({case_name})")

    plt.grid(True)

    plt.legend()

    plt.show()



CASES = [
    dict(
        name="trig",

        u="sin(8*x[0] + t)",

        f="cos(8*x[0] + t) + 64*a*sin(8*x[0] + t)",

        h="sin(8*x[0] + t)",

        g="8*x[0]/R * cos(8*x[0] + t)",

        u0="sin(8*x[0])"
    ),
]


# Параметры
R = 1.0
a_coef = 1.0

T = 2.0
dt = 0.05

# Запуск тестов

for case in CASES:

    print("\n================================================")
    print(f"CASE: {case['name']}")
    print("================================================\n")

    (
        mesh,
        uh,
        uex,
        times,
        errL2,
        errMax
    ) = solve_heat_equation(
        R=R,
        a_coef=a_coef,
        T=T,
        dt=dt,
        test_case=case,
        mesh_density=10
    )

    # Графики ошибок
    plot_errors(
        times,
        errL2,
        errMax,
        case["name"]
    )