from pyomo.environ import *
import matplotlib.pyplot as plt
import models as Models


# drawing 2 plots -
# 1. decition space: the possible values of the vars X1, X2
# 2. pareto optimal front
def show_plots(x1_l, x2_l, f1_l, f2_l):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    edge, = ax1.plot(x1_l, x2_l, 'o-', markersize=6,
                     markerfacecolor='black', c='black')
    # decision space
    shaded = ax1.fill_between(x1_l, x2_l, color='lightgray')
    ax1.legend([(edge, shaded)], ['decision space'])
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')
    ax1.set_xlim((min(x1_l)-1, max(x1_l)+1))
    ax1.set_ylim((min(x2_l)-1, max(x2_l)+1))
    ax1.grid(True)

    # optimal front
    shaded = ax2.fill_between(f1_l, f2_l, color='lightgray')
    edge, =  ax2.plot(f1_l, f2_l, 'o-', c='b', label='Pareto optimal front')
    ax2.legend([(edge, shaded)], ['pareto optimal front'])
    ax2.set_xlabel('Objective function F1')
    ax2.set_ylabel('Objective function F2')
    ax2.set_xlim((min(f1_l)-1, max(f1_l)+1))
    ax2.set_ylim((min(f2_l)-1, max(f2_l)+1))
    ax2.grid(True)
    fig.tight_layout()

    plt.show()


# problems from models.py
linear_problem_model = Models.Simple_Linear_Problem_Model().get_model()
bnh_problem_model = Models.BNH_Problem_Model().get_model()

# choosing a problem model
model = linear_problem_model
# solver for solving the problem - need to be installed!
SOLVER = 'ipopt'

### augmented epsilon-constraint algorithm: ###

# f1 without f2
model.O_f2.deactivate()
solver = SolverFactory(SOLVER)
solver.solve(model)

f2_min = value(model.f2)

# f2 without f1
model.O_f2.activate()
model.O_f1.deactivate()
solver = SolverFactory(SOLVER)
solver.solve(model)

f2_max = value(model.f2)


# max: f1 + delta * slack
# s.t: f2 - slack = epsilon

model.O_f1.activate()
model.O_f2.deactivate()

model.del_component(model.O_f1)
model.del_component(model.O_f2)

model.epsilon = Param(initialize=0, mutable=True)
model.delta = Param(initialize=0.00001)
model.slack = Var(within=NonNegativeReals)
model.O_f1 = Objective(expr=model.f1 + model.delta *
                       model.slack, sense=maximize)
model.C_epsilon = Constraint(expr=model.f2 - model.slack == model.epsilon)

# min f2 <= epsilon <= max f2
epsilon_possible_values = list(range(int(f2_min), int(f2_max), 1)) + [f2_max]

x1_l, x2_l, f1_l, f2_l = [], [], [], []
# going through all epsilon possible values
for val in epsilon_possible_values:
    model.epsilon = val
    solver.solve(model)
    # save values for drwaing the graphs
    x1_l.append(value(model.X1))
    x2_l.append(value(model.X2))
    f1_l.append(value(model.f1))
    f2_l.append(value(model.f2))

show_plots(x1_l, x2_l, f1_l, f2_l)
