
import numpy as np
import pandas as pd
from types import SimpleNamespace
from scipy.optimize import minimize_scalar, root, minimize

# Define the parameters
par = SimpleNamespace()
par.A = 1.0
par.gamma = 0.5
par.alpha = 0.3
par.nu = 1.0
par.epsilon = 2.0
par.tau = 0.0
par.T = 0.0
w = 1.0  # numeraire
par.kappa = 0.1

# Define functions for optimal firm behavior
def optimal_labor(p_j, w, A, gamma):
    return (p_j * A * gamma / w) ** (1 / (1 - gamma))

def optimal_output(l_j, A, gamma):
    return A * (l_j ** gamma)

def optimal_profit(w, p_j, A, gamma):
    return (1 - gamma) / gamma * w * (p_j * A * gamma / w) ** (1 / (1 - gamma))

# Define functions for consumer behavior
def c1_optimal(l, w, p1, p2, alpha, T, pi1, pi2):
    return alpha * (w * l + T + pi1 + pi2) / p1

def c2_optimal(l, w, p1, p2, alpha, T, pi1, pi2, tau):
    return (1 - alpha) * (w * l + T + pi1 + pi2) / (p2 + tau)

def utility_maximization(l, w, p1, p2, alpha, nu, epsilon, T, pi1, pi2, tau):
    c1 = c1_optimal(l, w, p1, p2, alpha, T, pi1, pi2)
    c2 = c2_optimal(l, w, p1, p2, alpha, T, pi1, pi2, tau)
    utility = np.log(c1 * alpha * c2 * (1 - alpha)) - nu * (l ** (1 + epsilon)) / (1 + epsilon)
    return -utility  # We minimize the negative utility to maximize utility

def check_market_clearing(par, w, p1_grid, p2_grid):
    results = []

    for p1 in p1_grid:
        for p2 in p2_grid:
            # Calculate firm behaviors
            l1_star = optimal_labor(p1, w, par.A, par.gamma)
            l2_star = optimal_labor(p2, w, par.A, par.gamma)
            y1_star = optimal_output(l1_star, par.A, par.gamma)
            y2_star = optimal_output(l2_star, par.A, par.gamma)
            pi1_star = optimal_profit(w, p1, par.A, par.gamma)
            pi2_star = optimal_profit(w, p2, par.A, par.gamma)
            
            # Consumer behavior
            res = minimize_scalar(
                utility_maximization, 
                bounds=(0, 100), 
                args=(w, p1, p2, par.alpha, par.nu, par.epsilon, par.T, pi1_star, pi2_star, par.tau), 
                method='bounded'
            )
            l_star = res.x
            c1_star = c1_optimal(l_star, w, p1, p2, par.alpha, par.T, pi1_star, pi2_star)
            c2_star = c2_optimal(l_star, w, p1, p2, par.alpha, par.T, pi1_star, pi2_star, par.tau)
            
            # Market clearing conditions
            labor_clearing = np.isclose(l_star, l1_star + l2_star)
            good1_clearing = np.isclose(c1_star, y1_star)
            good2_clearing = np.isclose(c2_star, y2_star)
            
            results.append({
                'p1': p1, 
                'p2': p2, 
                'labor_clearing': labor_clearing, 
                'good1_clearing': good1_clearing, 
                'good2_clearing': good2_clearing
            })

    results_df = pd.DataFrame(results)
    return results_df

def market_clearing_conditions(prices, par, w):
    p1, p2 = prices

    # Calculate firm behaviors
    l1_star = optimal_labor(p1, w, par.A, par.gamma)
    l2_star = optimal_labor(p2, w, par.A, par.gamma)
    y1_star = optimal_output(l1_star, par.A, par.gamma)
    y2_star = optimal_output(l2_star, par.A, par.gamma)
    pi1_star = optimal_profit(w, p1, par.A, par.gamma)
    pi2_star = optimal_profit(w, p2, par.A, par.gamma)
    
    # Consumer behavior
    res = minimize_scalar(
        utility_maximization, 
        bounds=(0, 100), 
        args=(w, p1, p2, par.alpha, par.nu, par.epsilon, par.T, pi1_star, pi2_star, par.tau), 
        method='bounded'
    )
    l_star = res.x
    c1_star = c1_optimal(l_star, w, p1, p2, par.alpha, par.T, pi1_star, pi2_star)
    c2_star = c2_optimal(l_star, w, p1, p2, par.alpha, par.T, pi1_star, pi2_star, par.tau)
    
    # Market clearing conditions
    labor_clearing = l_star - (l1_star + l2_star)
    good1_clearing = c1_star - y1_star
    
    return np.array([labor_clearing, good1_clearing])

def find_equilibrium_prices(par, w, initial_guess=[1.0, 1.0]):
    # Reset parameters to initial state before finding equilibrium prices
    par.T = 0.0
    
    # Find the equilibrium prices
    solution = root(market_clearing_conditions, initial_guess, args=(par, w))

    # Extract the equilibrium prices
    equilibrium_prices = solution.x
    p1_star, p2_star = equilibrium_prices

    equilibrium_prices_df = pd.DataFrame({
        'p1_star': [p1_star],
        'p2_star': [p2_star]
    })
    
    return equilibrium_prices_df