from pyomo.environ import *


class Simple_Linear_Problem_Model:
    def __init__(self):
        self.model = ConcreteModel()
        self.model.X1 = Var(within=NonNegativeReals)
        self.model.X2 = Var(within=NonNegativeReals)

        self.model.C1 = Constraint(expr=self.model.X1 <= 20)
        self.model.C2 = Constraint(expr=self.model.X2 <= 40)
        self.model.C3 = Constraint(
            expr=5 * self.model.X1 + 4 * self.model.X2 <= 200)

        self.model.f1 = Var()
        self.model.f2 = Var()
        self.model.C_f1 = Constraint(expr=self.model.f1 == self.model.X1)
        self.model.C_f2 = Constraint(
            expr=self.model.f2 == 3 * self.model.X1 + 4 * self.model.X2)
        self.model.O_f1 = Objective(expr=self.model.f1, sense=maximize)
        self.model.O_f2 = Objective(expr=self.model.f2, sense=maximize)

    def get_model(self):
        return self.model


class BNH_Problem_Model:
    def __init__(self):
        self.model = ConcreteModel()
        self.model.X1 = Var(within=NonNegativeReals)
        self.model.X2 = Var(within=NonNegativeReals)

        self.model.C1 = Constraint(expr=self.model.X1 <= 5)
        self.model.C2 = Constraint(expr=self.model.X2 <= 3)
        self.model.C3 = Constraint(
            expr=(self.model.X1-5)**2 + self.model.X2**2 <= 25)
        self.model.C4 = Constraint(
            expr=(self.model.X1-8)**2 + (self.model.X2+3)**2 >= 7.7)

        self.model.f1 = Var()
        self.model.f2 = Var()
        self.model.C_f1 = Constraint(
            expr=self.model.f1 == 4*(self.model.X1**2) + 4*(self.model.X2**2))
        self.model.C_f2 = Constraint(
            expr=self.model.f2 == ((self.model.X1-5)**2 + (self.model.X2-5)**2))
        self.model.O_f1 = Objective(expr=-self.model.f1, sense=minimize)
        self.model.O_f2 = Objective(expr=-self.model.f2, sense=minimize)

    def get_model(self):
        return self.model
