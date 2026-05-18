from dolfin import *
from mshr import Circle, generate_mesh
import matplotlib.pyplot as plt
import numpy as np


def solve_bvp(R, alpha, u_expr, f_expr, g_expr, n=40):

    # Геометрия
    mesh = generate_mesh(Circle(Point(0, 0), R), n)

    V = FunctionSpace(mesh, "P", 1)

    # Границы
    class GammaD(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] <= 0

    class GammaN(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0] > 0

    boundaries = MeshFunction("size_t", mesh, 1, 0)
    GammaD().mark(boundaries, 1)
    GammaN().mark(boundaries, 2)

    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Данные
    u_exact = Expression(u_expr, degree=4, R=R, alpha=alpha)
    f = Expression(f_expr, degree=4, R=R, alpha=alpha)
    g = Expression(g_expr, degree=4, R=R)

    # BC
    bc = DirichletBC(V, u_exact, boundaries, 1)

    # Вариационная задача
    u = TrialFunction(V)
    v = TestFunction(V)

    a = (dot(grad(u), grad(v)) + alpha*u*v) * dx
    L = f*v*dx + g*v*ds(2)

    # Решение
    uh = Function(V)
    solve(a == L, uh, bc)

    # Ошибки
    u_ref = interpolate(u_exact, V)

    err_L2 = errornorm(u_ref, uh, "L2")
    err_max = np.max(np.abs(
        uh.vector().get_local() - u_ref.vector().get_local()
    ))

    return mesh, uh, u_ref, err_L2, err_max


def plot_solution(mesh, u_num, u_ex, case):

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    coords = mesh.coordinates()
    cells = mesh.cells()

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

    ax[0].set_title(f"Numerical solution ({case})")
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

    ax[1].set_title(f"Exact solution ({case})")
    ax[1].set_aspect("equal")
    plt.colorbar(t2, ax=ax[1])

    plt.tight_layout()
    plt.show()


CASES = {
    # 1: dict(
    #     u="1 + x[0]*x[0] + 2*x[1]*x[1]",
    #     f="-6 + alpha*(1 + x[0]*x[0] + 2*x[1]*x[1])",
    #     g="2*x[0]*(x[0]/R) + 4*x[1]*(x[1]/R)"
    # ),

    # 2: dict(
    #     u="sin(x[0])*cos(x[1])",
    #     f="2*sin(x[0])*cos(x[1]) + alpha*sin(x[0])*cos(x[1])",
    #     g="cos(x[0])*cos(x[1])*(x[0]/R) - sin(x[0])*sin(x[1])*(x[1]/R)"
    # ),

    2: dict(
        u="sin(8*x[0])",
        f="64*sin(8*x[0]) + alpha*sin(8*x[0])",
        g="8*x[0]/R * cos(8*x[0])"
    ),

    # 3: dict(
    #     u="exp(x[0] + x[1])",
    #     f="-2*exp(x[0] + x[1]) + alpha*exp(x[0] + x[1])",
    #     g="exp(x[0] + x[1])*(x[0]/R + x[1]/R)"
    # )
}


R = 1.0
alpha = 1.0

for k, data in CASES.items():
    print(f"Case {k}:")
    print(
        f" u = {data["u"]}\n f = {data["f"]}\n g = {data["g"]}\n"
    )

    mesh, uh, uex, eL2, emax = solve_bvp(
        R, alpha,
        data["u"],
        data["f"],
        data["g"],
        n=5
    )

    print(f"L2   = {eL2:.3e}")
    print(f"Max  = {emax:.3e}\n")

    plot_solution(mesh, uh, uex, k)