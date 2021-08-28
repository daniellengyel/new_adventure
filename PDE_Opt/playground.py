from dolfin import *
from dolfin_adjoint import *

import moola 

n = 64 
mesh = UnitSquareMesh(n, n)


# cf = MeshFunction("bool", mesh, mesh.geometric_dimension())
# subdomain = CompiledSubDomain('std::abs(x[0]-0.5) < 0.25 && std::abs(x[1]-0.5) < 0.25')
# subdomain.mark(cf, True)
# mesh = refine(mesh, cf)

V = FunctionSpace(mesh, "CG", 1)
U = FunctionSpace(mesh, "DG", 0)

u = interpolate(Expression("x[0]+x[1]", name='Control', degree=1), U)
y = Function(V, name='State')
v = TestFunction(V)

F = (inner(grad(y), grad(v)) - u * v) * dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, y, bc)

x = SpatialCoordinate(mesh)
w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
d = 1 / (2 * pi ** 2)
d = Expression("d*w", d=d, w=w, degree=3)

alpha = Constant(1e-6)
J = assemble((0.5 * inner(y - d, y - d)) * dx + alpha / 2 * u ** 2 * dx)
control = Control(u)

rf = ReducedFunctional(J, control)

print(rf.derivative().vector().get_local().shape)
print(rf(u))
print(u.vector().get_local().shape)
# print(compute_gradient(J, u))
# print(compute_gradient(rf, control))

# problem = MoolaOptimizationProblem(rf)
# f_moola = moola.DolfinPrimalVector(f)
# solver = moola.NewtonCG(problem, f_moola, options={'gtol': 1e-9,
#                                                    'maxiter': 20,
#                                                    'display': 3,
#                                                    'ncg_hesstol': 0})

# sol = solver.solve()
# f_opt = sol['control'].data

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