from types import SimpleNamespace
import numpy as np
from scipy import optimize

class ExchangeEconomyClass:

    def __init__(self):

        par = self.par = SimpleNamespace()

        # a. preferences
        par.alpha = 1/3
        par.beta = 2/3

        # b. endowments
        par.w1A = 0.8
        par.w2A = 0.3
        par.w1B = 1-par.w1A
        par.w2B = 1-par.w2A

    def utility_A(self,x1A,x2A):
        par = self.par
        return x1A**par.alpha*x2A**(1-par.alpha)

    def utility_B(self,x1B,x2B):
        par = self.par
        return x1B**par.beta*x2B**(1-par.beta)

    def demand_A(self,p1):
        par = self.par
        x1A = par.alpha*(p1*par.w1A+par.w2A)/p1
        x2A = (1-par.alpha)*(p1*par.w1A+par.w2A)
        x1A = max(0, min(1, x1A))
        x2A = max(0, min(1, x2A))
        return x1A, x2A

    def demand_B(self,p1):
        par = self.par
        x1B = par.beta*(p1*par.w1B+par.w2B)/p1
        x2B = (1-par.beta)*(p1*par.w1B+par.w2B)
        x1B = max(0, min(1, x1B))
        x2B = max(0, min(1, x2B))
        return x1B, x2B 
    
    def initial_endowment(self):
        par = self.par
        initial_utility_A = self.utility_A(par.w1A,par.w2A)
        initial_utility_B = self.utility_B(par.w1B,par.w2B)
        return initial_utility_A, initial_utility_B
    

    # For question 2
    def check_market_clearing(self,p1):

        par = self.par

        x1A,x2A = self.demand_A(p1)
        x1B,x2B = self.demand_B(p1)

        eps1 = x1A-par.w1A + x1B-(1-par.w1A)
        eps2 = x2A-par.w2A + x2B-(1-par.w2A)

        return eps1,eps2

    # For question 4a
    def optimal_allocation(self):

        # create empty 
        max_utility = float('-inf')
        optimal_price = None
        optimal_allocation = (None, None)

        # define the prices
        prices = np.linspace(0.5,2.5,75)

        # deceide B's consumption
        for p1 in prices:
            x1_B_star,x2_B_star = self.demand_B(p1)

            # allocate rest to A
            remaining_x1A = 1- x1_B_star
            remaining_x2A = 1- x2_B_star

            # calculate A's utility 
            if remaining_x1A >= 0 and remaining_x2A >= 0:
                utility_for_A = self.utility_A(remaining_x1A,remaining_x2A)
            else:
                utility_for_A = 0

            # check if this is greater
            if utility_for_A > max_utility:
                max_utility = utility_for_A
                optimal_price = p1
                optimal_allocation = (remaining_x1A, remaining_x2A)

        return max_utility, optimal_price, optimal_allocation
    






    



        

