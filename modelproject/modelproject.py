from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class modelclass():

    def __init__(self):

        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()
       
        self.setup()

        self.allocate()

    def setup(self):
        """Baseline parameters"""
        par = self.par

        # a. Household parameters
        par.rho = 0.05           # discount rate
        par.n = 0.04             # population growth

        # b. Firms parameters
        par.p_f = 'cobb-douglas' # production function
        par.alpha = 1/3          # percentage of capital used in production

        # c. Government parameters
        par.tau = 0.0            # wage tax
        par.Gt = 0.0             # government purchases (per worker)

        # d. Start values and length of simulation
        par.K_ini = 0.1          # initial capital stock
        par.L_ini = 1.0          # initial population
        par.simT = 20            # length of simulation

    def allocate(self):
        """Allocate arrays for simulation"""
        par = self.par
        sim = self.sim

        # a. List of variables
        household = ['C1', 'C2', 's']
        firm = ['K', 'Y', 'L', 'k']
        prices = ['w', 'r', 'tau']
        government = ['Gt']

        # b. Allocates
        allvarnames = household + firm + prices + government
        for varname in allvarnames:
            sim.__dict__[varname] = np.nan * np.ones(par.simT)

    def simulate(self, do_print=True):
        """Simulate model"""
        t0 = time.time()
        par = self.par
        sim = self.sim

        # Initial values for simulation
        sim.K[0] = par.K_ini
        sim.L[0] = par.L_ini

        # Simulate the model
        for t in range(par.simT):
            self.simulate_before_s(par, sim, t)
            if t == par.simT - 1: continue

            s_min, s_max = self.find_s_bracket(par, sim, t)

            obj = lambda s: self.calc_euler_error(s, par, sim, t=t)
            result = optimize.root_scalar(obj, bracket=(s_min, s_max), method='bisect')
            s = result.root

            sim.s[t] = s
            self.simulate_after_s(par, sim, t, s)

        if do_print: print(f'simulation done in {time.time() - t0:.3f} secs')

    def find_s_bracket(self, par, sim, t, maxiter=500, do_print=False):
        """Find bracket for s to search in"""
        s_min = 0.0 + 1e-8
        s_max = 1.0 - 1e-8

        value = self.calc_euler_error(s_max, par, sim, t)
        sign_max = np.sign(value)
        lower = s_min
        upper = s_max

        it = 0
        while it < maxiter:
            s = (lower + upper) / 2
            value = self.calc_euler_error(s, par, sim, t)

            valid = not np.isnan(value)
            correct_sign = np.sign(value) * sign_max < 0

            if valid and correct_sign:
                s_min = s
                s_max = upper
                return s_min, s_max
            elif not valid:
                lower = s
            else:
                upper = s

            it += 1

    def calc_euler_error(self, s, par, sim, t):
        """Target function for finding s with bisection"""
        self.simulate_after_s(par, sim, t, s)
        self.simulate_before_s(par, sim, t + 1)

        par.beta = 1 / (1 + par.rho)
        LHS = sim.C1[t] ** (-1)
        RHS = (1 + sim.r[t + 1]) * par.beta * sim.C2[t + 1] ** (-1)

        return LHS - RHS

    def simulate_before_s(self, par, sim, t):
        """Simulate forward"""
        if t == 0:
            sim.K[t] = par.K_ini
            sim.L[t] = par.L_ini
        if t > 0:
            sim.L[t] = sim.L[t - 1] * (1 + par.n)

        sim.Y[t] = sim.K[t] ** par.alpha * (sim.L[t]) ** (1 - par.alpha)
        sim.r[t] = par.alpha * sim.K[t] ** (par.alpha - 1) * (sim.L[t]) ** (1 - par.alpha)
        sim.w[t] = (1 - par.alpha) * sim.K[t] ** (par.alpha) * (sim.L[t]) ** (-par.alpha)

        # Update tax and government expenditure
        sim.tau[t] = par.tau
        sim.Gt[t] = par.Gt

        sim.C2[t] = (1 + sim.r[t]) * (sim.K[t])

    def simulate_after_s(self, par, sim, t, s):
        """Simulate forward"""
        sim.k[t] = sim.K[t] / sim.L[t]
        sim.C1[t] = ((1 - par.tau) * sim.w[t] * (1.0 - s) * sim.L[t])

        I = sim.Y[t] - sim.C1[t] - sim.C2[t] - sim.Gt[t]
        sim.K[t + 1] = sim.K[t] + I

    def run_with_shock(self, tau_shock, Gt_shock):
        """Run simulation with a tax and government expenditure shock"""
        self.setup()
        self.allocate()

        # Apply the shocks
        self.par.tau = tau_shock
        self.par.Gt = Gt_shock

        # Run the simulation
        self.simulate(do_print=False)

    def plot_convergence(self, k_no_shock, k_with_shock, ks_1):
        """Plot the results of the convergence simulation"""
        fig = plt.figure(figsize=(6, 6 / 1.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(k_no_shock, label=r'$k_{t}$, (No Shock)', color='blue')
        ax.plot(k_with_shock, label=r'$k_{t}$, (With Shock)', color='red')
        ax.axhline(ks_1, ls='--', color='black', label='Analytical Steady State')
        ax.legend(frameon=True, fontsize=12)
        ax.set_title('Convergence of Capital Accumulation')
        ax.set_xlabel('Numbers of Periods')
        ax.set_ylabel('Steady State Value')
        fig.tight_layout()
        plt.show()

    def plot_k_vs_k(self, k_no_shock, k_with_shock, ks_1):
        """Plot k_{t+1} vs k_t"""
        k_t_max = max(np.max(k_no_shock), np.max(k_with_shock))
        k_vals = np.linspace(0, k_t_max, 100)
        
        # Production function for k_{t+1} vs k_t without shock
        k_next_no_shock = (1 - self.par.alpha) * k_vals ** self.par.alpha / ((1 + self.par.n) * (2 + self.par.rho))

        # Production function for k_{t+1} vs k_t with shock
        k_next_with_shock = (1 - self.par.alpha) * k_vals ** self.par.alpha / ((1 + self.par.n) * (2 + self.par.rho)) - self.par.Gt

        # Plot the trajectories
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(k_vals, k_next_no_shock, label=r'Without tax', color='blue')
        ax.plot(k_vals, k_next_with_shock, label=r'With tax', color='red')
        ax.legend(frameon=True, fontsize=12)
        ax.set_xlabel(r'$k_t$')
        ax.set_ylabel(r'$k_{t+1}$')
        ax.set_title(r'Capital Accumulation with and without tax')
        ax.grid(True)

        plt.show()