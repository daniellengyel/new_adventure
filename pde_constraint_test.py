
from dolfin import *

from dolfin_adjoint import *


import moola

n = 8
mesh = UnitSquareMesh(n, n)

# cf = MeshFunction("bool", mesh, mesh.geometric_dimension())
# subdomain = CompiledSubDomain('std::abs(x[0]-0.5) < 0.25 && std::abs(x[1]-0.5) < 0.25')
# subdomain.mark(cf, True)
# mesh = refine(mesh, cf)



V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)

f = interpolate(Expression("x[0]+x[1]", name='Control', degree=1), W)
u = Function(V, name='State')
v = TestFunction(V)

F = (inner(grad(u), grad(v)) - f * v) * dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, u, bc)


x = SpatialCoordinate(mesh)
w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
d = 1 / (2 * pi ** 2)
d = Expression("d*w", d=d, w=w, degree=3)

alpha = Constant(1e-6)
J = assemble((0.5 * inner(u - d, u - d)) * dx + alpha / 2 * f ** 2 * dx)
control = Control(f)


rf = ReducedFunctional(J, control)


problem = MoolaOptimizationProblem(rf)
f_moola = moola.DolfinPrimalVector(f)
solver = moola.NewtonCG(problem, f_moola, options={'gtol': 1e-9,
                                                   'maxiter': 20,
                                                   'display': 3,
                                                   'ncg_hesstol': 0})


sol = solver.solve()
f_opt = sol['control'].data

# plot(f_opt, title="f_opt")

# # Define the expressions of the analytical solution
# f_analytic = Expression("1/(1+alpha*4*pow(pi, 4))*w", w=w, alpha=alpha, degree=3)
# u_analytic = Expression("1/(2*pow(pi, 2))*f", f=f_analytic, degree=3)

# # We can then compute the errors between numerical and analytical
# # solutions.

# f.assign(f_opt)
# solve(F == 0, u, bc)
# control_error = errornorm(f_analytic, f_opt)
# state_error = errornorm(u_analytic, u)
# print("h(min):           %e." % mesh.hmin())
# print("Error in state:   %e." % state_error)
# print("Error in control: %e." % control_error)

# # The example code can be found in ``examples/poisson-mother`` in the
# # ``dolfin-adjoint`` source tree, and executed as follows:
# #
# # .. code-block:: bash
# #
# #   $ python poisson-mother.py
# #   ...
# #
# # Convergence order and mesh independence
# # ***************************************
# #
# # It is highly desirable that the optimisation algorithm achieve mesh
# # independence: i.e., that the required number of optimisation
# # iterations is independent of the mesh resolution.  Achieving mesh
# # independence requires paying careful attention to the inner product
# # structure of the function space in which the solution is sought.
# #
# # For our desired temperature, the analytical solutions of the optimisation
# # problem is:
# #
# # .. math::
# #     f_{\textrm{analytic}} &= \frac{1}{1+4\alpha \pi^4}\sin(\pi x) \sin(\pi y) \\
# #     u_{\textrm{analytic}} &= \frac{1}{2\pi^2}f_{\textrm{analytic}}
# #
# # The following numerical experiments solve the optimisation problem
# # for a sequence of meshes with increasing resolutions and record the
# # numerical error and the required number of optimisation iterations.
# # A regularisation coefficient of :math:`\alpha = 10^{-6}` was used, and
# # the optimisation was stopped when the :math:`L_2` norm of the
# # reduced functional gradient dropped below :math:`10^{-9}`.
# #
# #
# # Moola Newton-CG
# # ---------------
# #
# # The Moola Newton-CG algorithm implements an inexact Newton method.
# # Hence, even though the optimality system of our problem is linear,
# # we can not expect the algorithm to converge in a single iteration
# # (however, we could it enforce that by explicitly setting the
# # relative tolerance of the CG algorithm to zero).
# #
# # Running the Newton-CG algorithm for the different meshes yielded:
# #
# # ===================  =================  ============== ================
# #   Mesh element size  Newton iterations  CG iterations  Error in control
# # ===================  =================  ============== ================
# #   6.250e-02            3                 54            3.83e-02
# #   3.125e-02            3                 59            1.69e-02
# #   1.563e-02            3                 57            8.05e-03
# #   7.813e-03            3                 58            3.97e-03
# # ===================  =================  ============== ================
# #
# # Here CG iterations denotes the total number of CG iterations during
# # the optimisation. Mesh independent convergence can be observed, both
# # in the Newton and CG iterations.
# #
# # From our choice of discretisation (:math:`DG_0` for :math:`f`), we
# # expect a 1st order of convergence for the control variable.  Indeed,
# # the error column in the numerical experiments confirm that this rate
# # is obtained in practice.
# #
# # Moola L-BFGS
# # ------------
# #
# # The L-BFGS algorithm in Moola implements the limited memory quasi
# # Newton method with Broyden-Fletcher-Goldfarb-Shanno updates.  For
# # the numerical experiments, the set of the memory history was set to
# # 10.
# #
# # The numerical results yield:
# #
# # ===================  ==================  =================
# #   Mesh element size  L-BFGS iterations   Error in control
# # ===================  ==================  =================
# #   6.250e-02             53                3.83e-02
# #   3.125e-02             50                1.69e-02
# #   1.563e-02             57                8.05e-03
# #   7.813e-03             56                3.97e-03
# # ===================  ==================  =================
# #
# # Again a mesh-independent convergence and a 1st order convergence of
# # the control can be observed.
# #
# # .. bibliography:: /documentation/poisson-mother/poisson-mother.bib
# #    :cited:
# #    :labelprefix: 1E-
