from types import SimpleNamespace
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider
import time


class OLG_model:
    def __init__(self, rho=0.05, A=1.00, alpha=1/3, n=0.04, tau=0.0):
        self.par = SimpleNamespace()
        self.par.rho = rho
        self.par.A = A
        self.par.alpha = alpha
        self.par.n = n
        self.par.tau = tau
        self.k_star = None
        self.k_star_payg = None

    def objective_function(self, k):
        par = self.par
        return k - ((par.A * (1 - par.alpha) * k ** par.alpha) / ((1 + par.n) * (2 + par.rho)))

    def solve_steady_state(self):
        result = optimize.root_scalar(self.objective_function, bracket=[1e-10, 10], method='bisect')
        self.k_star = result.root
        return self.k_star

    def plot_steady_state(self):
        par = self.par
        if self.k_star is None:
            self.solve_steady_state()

        k_star = self.k_star
        k_t = np.linspace(0, 2 * k_star, 400)
        k_t_plus_1 = (par.A * (1 - par.alpha) * k_t ** par.alpha) / ((1 + par.n) * (2 + par.rho))

        plt.figure(figsize=(10, 6))
        plt.plot(k_t, k_t, label='45-degree line', linestyle='--')
        plt.plot(k_t, k_t_plus_1, label='$k_{t+1}$', color='blue')
        plt.scatter(k_star, k_star, color='red', zorder=5, label=f'Steady State = {k_star:.3f}')
        plt.axvline(x=k_star, ymax=0.125 / max(k_t_plus_1), color='red', linestyle=':', label=f'$k^*$ = {k_star:.3f}')

        plt.xlabel('$k_t$')
        plt.ylabel('$k_{t+1}$')
        plt.title('Capital Accumulation Growth Path')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 0.3)
        plt.xlim(0, 0.3)
        plt.show()

    def objective_function_payg(self, k):
        par = self.par
        return k - ((par.A * (1 - par.alpha) * k ** par.alpha) / ((1 + par.n) * (2 + par.rho)) *
                    (1 / (1 + ((1 + par.rho) / (2 + par.rho)) * ((1 - par.alpha) / par.alpha) * par.tau)))

    def solve_steady_state_payg(self):
        result = optimize.root_scalar(self.objective_function_payg, bracket=[1e-10, 10], method='bisect')
        self.k_star_payg = result.root
        return self.k_star_payg

    def plot_steady_state_payg(self):
        par = self.par
        if self.k_star_payg is None:
            self.solve_steady_state_payg()

        k_star = self.k_star
        k_t = np.linspace(0, 2 * k_star, 400)
        k_t_plus_1 = (par.A * (1 - par.alpha) * k_t ** par.alpha) / ((1 + par.n) * (2 + par.rho))

        k_star_payg = self.k_star_payg
        k_t_payg = np.linspace(0, 2 * k_star_payg, 400)
        k_t_plus_1_payg = ((par.A * (1 - par.alpha) * k_t_payg ** par.alpha) /
                           ((1 + par.n) * (2 + par.rho)) *
                           (1 / (1 + ((1 + par.rho) / (2 + par.rho)) * ((1 - par.alpha) / par.alpha) * par.tau)))

        plt.figure(figsize=(10, 6))
        plt.plot(k_t_payg, k_t_payg, label='45-degree line', linestyle='--')
        plt.plot(k_t, k_t_plus_1, label='$k_{t+1}$', color='blue')
        plt.scatter(k_star, k_star, color='red', zorder=5, label=f'Steady State = {k_star:.3f}')
        plt.axvline(x=k_star, ymax=k_star / max(k_t_plus_1), color='red', linestyle=':', label=f'$k^*$ = {k_star:.3f}')

        plt.plot(k_t_payg, k_t_plus_1_payg, label='$k_{t+1}$ with PAYG', color='green')
        plt.scatter(k_star_payg, k_star_payg, color='red', zorder=5, label=f'Steady State with PAYG = {k_star_payg:.3f}')
        plt.axvline(x=k_star_payg, ymax=k_star_payg / max(k_t_plus_1_payg), color='red', linestyle=':', label=f'$k^*_{PAYG}$ = {k_star_payg:.3f}')

        plt.xlabel('$k_t$')
        plt.ylabel('$k_{t+1}$')
        plt.title('Capital Accumulation Growth Path with PAYG')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 0.3)
        plt.xlim(0, 0.3)
        plt.show()

    def plot_combined(self):
        par = self.par
        if self.k_star is None:
            self.solve_steady_state()
        if self.k_star_payg is None:
            self.solve_steady_state_payg()

        k_star = self.k_star
        k_star_payg = self.k_star_payg
        k_t = np.linspace(0, 2 * k_star, 400)
        k_t_plus_1 = (par.A * (1 - par.alpha) * k_t ** par.alpha) / ((1 + par.n) * (2 + par.rho))
        k_t_payg = np.linspace(0, 2 * k_star_payg, 400)
        k_t_plus_1_payg = ((par.A * (1 - par.alpha) * k_t_payg ** par.alpha) /
                           ((1 + par.n) * (2 + par.rho)) *
                           (1 / (1 + ((1 + par.rho) / (2 + par.rho)) * ((1 - par.alpha) / par.alpha) * par.tau)))

        plt.figure(figsize=(10, 6))
        plt.plot(k_t, k_t, label='45-degree line', linestyle='--')
        plt.plot(k_t, k_t_plus_1, label='$k_{t+1}$ without PAYG', color='blue')
        plt.scatter(k_star, k_star, color='red', zorder=5, label=f'Steady State without PAYG = {k_star:.3f}')
        plt.axvline(x=k_star, ymax=0.205 / max(k_t), color='red', linestyle=':')

        plt.plot(k_t_payg, k_t_plus_1_payg, label='$k_{t+1}$ with PAYG', color='green')
        plt.scatter(k_star_payg, k_star_payg, color='red', zorder=5, label=f'Steady State with PAYG = {k_star_payg:.3f}')
        plt.axvline(x=k_star_payg, ymax=0.08 / max(k_t_payg), color='red', linestyle=':')

        plt.xlabel('$k_t$')
        plt.ylabel('$k_{t+1}$')
        plt.title('Capital Accumulation Growth Path with and without PAYG')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.ylim(0, 0.3)
        plt.xlim(0, 0.3)
        plt.show()

    def calculate_steady_state_payg(self):
        par = self.par
        tau = par.tau
        n = par.n
        obj_payg = lambda k: k - ((par.A * (1 - par.alpha) * k**par.alpha) / ((1 + n) * (2 + par.rho)) * (1 / (1 + ((1 + par.rho) / (2 + par.rho)) * ((1 - par.alpha) / par.alpha) * tau)))
        result_payg = optimize.root_scalar(obj_payg, bracket=[1e-10, 10], method='bisect')
        k_star_payg = result_payg.root
        
        k_t = np.linspace(0, 0.25, 400)
        k_t_plus_1 = (par.A * (1 - par.alpha) * k_t ** par.alpha) / ((1 + n) * (2 + par.rho))
        k_t_plus_1_payg = ((par.A * (1 - par.alpha) * k_t ** par.alpha) / ((1 + n) * (2 + par.rho)) * (1 / (1 + ((1 + par.rho) / (2 + par.rho)) * ((1 - par.alpha) / par.alpha) * tau)))
        
        return k_t, k_t_plus_1, k_t_plus_1_payg, k_star_payg

    def plot_steady_state_payg_interactive_n(self, tau, n):
        self.par.tau = tau
        self.par.n = n
        k_t, k_t_plus_1, k_t_plus_1_payg, k_star_payg = self.calculate_steady_state_payg()
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_t, k_t, label='45-degree line', linestyle='--')
        plt.plot(k_t, k_t_plus_1, label='$k_{t+1}$ without PAYG', color='blue')
        plt.plot(k_t, k_t_plus_1_payg, label='$k_{t+1}$ with PAYG', color='green')
        plt.scatter(k_star_payg, k_star_payg, color='red', zorder=5, label=f'Steady State with PAYG = {k_star_payg:.3f}')
        plt.xlabel('$k_t$')
        plt.ylabel('$k_{t+1}$')
        plt.title('Capital Accumulation Growth Path with PAYG')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 0.25)
        plt.xlim(0, 0.25)
        plt.show()
    
    def plot_interactive_tau_and_n(self):
        interact(self.plot_steady_state_payg_interactive_n, tau=FloatSlider(min=0.0, max=1, step=0.05, value=0.35), n=FloatSlider(min=-0.1, max=0.1, step=0.01, value=0.04))

    def simulate_capital_accumulation(self, k0, periods, n, delay=False):
        par = self.par
        k_t = np.zeros(periods)
        k_t[0] = k0
        for t in range(1, periods):
            k_t[t] = (par.A * (1 - par.alpha) * k_t[t-1] ** par.alpha) / ((1 + n) * (2 + par.rho))
            if delay:
                time.sleep(0.1)
        return k_t
    
    def plot_convergence(self, k0, periods, delay=False):
        if self.k_star is None:
            self.solve_steady_state()

        k_t_old = self.simulate_capital_accumulation(k0, periods, 0.05, delay)
        k_t_new = self.simulate_capital_accumulation(k0, periods, 0.10, delay)

        plt.figure(figsize=(10, 6))
        plt.plot(range(periods), k_t_old, label='$k_t$, old (n = 0.05)')
        plt.plot(range(periods), k_t_new, label='$k_t$, new (n = 0.10)')
        plt.axhline(y=k_t_old[-1], color='black', linestyle='--', label='analytical steady state (n = 0.05)')
        plt.axhline(y=k_t_new[-1], color='black', linestyle='--', label='analytical steady state (n = 0.10)')
        plt.xlabel('Periods')
        plt.ylabel('Steady state capital')
        plt.title('Convergence of capital accumulation')
        plt.legend()
        plt.grid(True)
        plt.show()


    
