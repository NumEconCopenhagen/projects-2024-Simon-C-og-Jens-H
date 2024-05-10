from types import SimpleNamespace
import time
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class modelclass():
    
    def __init__(self):
        
        self.par = SimpleNamespace()
        self.sim = SimpleNamespace()

    def params(self):
        par = self.par

        par.alpha = 1/3
        par.rho = 0.05
        par.n = 0.04
        par.A = 1
        par.tau = 0
        par.Gt = 0

        par.initial_K = 0.05
        par.initial_L = 1
        par.periods = 50

    def euler_error(self, s, par, sim, t):
        self.sim_until_ss(par, sim, t, s)
        self.sim_after_ss(par, sim, t+1)

        par.beta = 1 / (1 + par.rho)
        LHS = sim.C1[t] ** -1
        RHS = (1 + sim.r[t+1]) * par.beta * sim.C2[t] ** -1
        return RHS - LHS

    def sim_after_ss(self, par, sim, t, s):
        sim.k[t] = sim.K[t] / sim.L[t]
        sim.C1[t] = ((1 - par.tau) * sim.w[t] * (1 - s) * sim.L[t])

        I = sim.Y[t] - sim.C1[t] - sim.C2[t] - sim.Gt[t]
        sim.K[t+1] = sim.K[t] + I

    def sim_until_ss(self, par, sim, t):
        if t == 0:
            sim.K[0] = par.initial_K
            sim.L[0] = par.initial_L
        if t > 0:
            sim.L[t] = sim.L[t - 1] * (1 + par.n)

        sim.Y[t] = sim.K[t] ** par.alpha * sim.L[t] ** (1 - par.alpha)
        sim.r[t] = par.alpha * sim.K[t] ** (par.alpha - 1) * sim.L[t] ** (1 - par.alpha)
        sim.w[t] = (1 - par.alpha) * sim.K[t] ** par.alpha *sim.L[t] ** -par.alpha

        sim.tau[t] = par.tau
        sim.Gt[t] = par.Gt
        
        sim.C2[t] = (1 + sim.r[t]) * sim.K[t]

    def brackets(self, par, sim, t, maxiter=500, do_print=False):
        s_min = 0 + 1e-8
        s_max = 1 - 1e-8

        euler = self.euler_error(s, par, sim, t)
        sign_max = np.sign(euler)
        lower = s_min
        upper = s_max
        it = 0
        while it < maxiter:
            s = (lower + upper) / 2
            euler = self.euler_error(s, par, sim, t)

            valid = not np.isnan(euler)
            sign = np.sign(euler) * sign_max < 0

            if valid and sign:
                s_min = s
                s_max = upper
                return s_min, s_max
            elif not valid:
                lower = s
            else:
                upper = s

            it += 1


    def sim_model(self):
        
        par = self.par
        sim = self.sim

        t0 = time.time()

        sim.K[0] = par.initial_K
        sim.L[0] = par.initial_L

        for t in range(par.periods):
            self.sim_until_ss(par, sim, t)
            if t == par.periods - 1: continue

            s_min, s_max = self.brackets(par, sim, t)

            obj = lambda s: self.euler_error(s, par, sim, t=t)
            result = optimize.root_scalar(obj, brackets=(s_min,s_max), method='bisect')
            s = result.root

            sim.s[t] = s
            self.sim_after_ss(par, sim, t, s)
        
    def run_with_shock(self, tau_shock, Gt_shock):
        """Run simulation with a tax and government expenditure shock"""
        par = self.par

        # Apply the shocks
        par.tau = tau_shock
        par.Gt = Gt_shock

        # Run the simulation
        self.sim_model()

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
        ax.plot(k_vals, k_next_no_shock, label=r'$k_{t+1}$ vs. $k_t$ (No Shock)', color='blue')
        ax.plot(k_vals, k_next_with_shock, label=r'$k_{t+1}$ vs. $k_t$ (With Shock)', color='red')
        ax.plot(k_no_shock[:-1], k_no_shock[1:], 'o-', label=r'Trajectory (No Shock)', color='blue')
        ax.plot(k_with_shock[:-1], k_with_shock[1:], 'o-', label=r'Trajectory (With Shock)', color='red')
        ax.axvline(ks_1, color='black', ls='--', label='Analytical Steady State')
        ax.legend(frameon=True, fontsize=12)
        ax.set_xlabel(r'$k_t$')
        ax.set_ylabel(r'$k_{t+1}$')
        ax.set_title(r'Capital Accumulation Trajectories ($k_{t+1}$ vs. $k_t$)')
        ax.grid(True)

        # Adding arrows to represent movement direction
        for i in range(len(k_no_shock) - 1):
            ax.annotate('', xy=(k_no_shock[i+1], k_no_shock[i+1]), xytext=(k_no_shock[i], k_no_shock[i+1]),
                        arrowprops=dict(arrowstyle='->', color='blue'))

        for i in range(len(k_with_shock) - 1):
            ax.annotate('', xy=(k_with_shock[i+1], k_with_shock[i+1]), xytext=(k_with_shock[i], k_with_shock[i+1]),
                        arrowprops=dict(arrowstyle='->', color='red'))

        plt.show()

    

    




