import numpy as np
import autograd.numpy as anp
import matplotlib.pyplot as plt

from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation


class FirstProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=2,
                         n_constr=2,
                         xl=np.array([-2,-2]),
                         xu=np.array([2,2]))

    def _evaluate(self, x, out, *args, **kwargs):
        
        f1 = 100 * (x[0]**2 + x[1]**2)
        f2 = (x[0]-1)**2 + x[1]**2

        g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8

        out["F"] = [f1, f2]
        out["G"] = [g1, g2]


# Definition of the BNH problem as it is presented in the article - 2 vars, 2 objectives and 2 constraints.
class BNH(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, n_constr=2, type_var=anp.double)
        self.xl = anp.zeros(self.n_var)
        self.xu = anp.array([5.0, 3.0])

    # Calculate the objectives values (f1, f2) according to the solution x
    def _evaluate(self, x, out, *args, **kwargs):
        f1 = 4 * x[:, 0] ** 2 + 4 * x[:, 1] ** 2
        f2 = (x[:, 0] - 5) ** 2 + (x[:, 1] - 5) ** 2
        g1 = (1 / 25) * ((x[:, 0] - 5) ** 2 + x[:, 1] ** 2 - 25)
        g2 = -1 / 7.7 * ((x[:, 0] - 8) ** 2 + (x[:, 1] + 3) ** 2 - 7.7)

        out["F"] = anp.column_stack([f1, f2])
        out["G"] = anp.column_stack([g1, g2])

    def _calc_pareto_front(self, n_points=100):
        x1 = anp.linspace(0, 5, n_points)
        x2 = anp.linspace(0, 5, n_points)
        x2[x1 >= 3] = 3

        X = anp.column_stack([x1, x2])
        return self.evaluate(X, return_values_of=["F"])


if __name__ == "__main__":
    problem = BNH()
    
    # Definition of the different parameters of the algorithm
    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=10,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )
    # Number of iteration the algorithm will do
    termination = get_termination("n_gen", 40)

    # Solve the problem and return the result
    result = minimize(problem,
                      algorithm,
                      termination,
                      seed=1,
                      save_history=True,
                      verbose=True)

    X = result.X
    F = result.F

    xl, xu = problem.bounds()
    plt.figure(figsize=(7, 5))
    plt.scatter(X[:, 0], X[:, 1], s=30, facecolors='none', edgecolors='r')
    plt.xlim(xl[0], xu[0])
    plt.ylim(xl[1], xu[1])
    plt.title("Design Space")
    plt.show()


    plt.figure(figsize=(7, 5))
    plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
    plt.title("Objective Space")
    plt.show()