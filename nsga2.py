import argparse
import autograd.numpy as anp
import matplotlib.pyplot as plt

from pymoo.optimize import minimize
from pymoo.util.remote import Remote
from pymoo.core.problem import Problem
from pymoo.factory import get_termination
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation

  

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

# Definition of the Kursawe problem as it is presented in the article - 3 vars, 2 objectives.
class Kursawe(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_constr=0, xl=-5, xu=5, type_var=anp.double)

    def _evaluate(self, x, out, *args, **kwargs):
        l = []
        
        for i in range(2):
            l.append(-10 * anp.exp(-0.2 * anp.sqrt(anp.square(x[:, i]) + anp.square(x[:, i + 1]))))

        f1 = anp.sum(anp.column_stack(l), axis=1)
        f2 = anp.sum(anp.power(anp.abs(x), 0.8) + 5 * anp.sin(anp.power(x, 3)), axis=1)

        out["F"] = anp.column_stack([f1, f2])

    def _calc_pareto_front(self, *args, **kwargs):
        return Remote.get_instance().load("pf", "kursawe.pf")


        
if __name__ == "__main__":
    problem_dict = {
        "kursawe" : Kursawe,
        "bnh" : BNH,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", help="List of problems tou can choose", choices=["kursawe", "bnh"],)
    parser.add_argument("generations", help="Number of iterations for the algorithm", type=int,)
    args = parser.parse_args()

    problem = problem_dict[args.problem]()
    # Definition of the different parameters of the algorithm
    algorithm = NSGA2(
        pop_size=40,
        n_offsprings=10,
        sampling=get_sampling("real_random"),
        crossover=get_crossover("real_sbx", prob=0.9, eta=15),
        mutation=get_mutation("real_pm", eta=20),
        eliminate_duplicates=True
    )

    # Number of iteration the algorithm will do
    termination = get_termination("n_gen", args.generations)

    # Solve the problem and return the result
    result = minimize(problem,
                      algorithm,
                      termination,
                      seed=1,
                      save_history=True,
                      verbose=True)

    X = result.X
    print(X)
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