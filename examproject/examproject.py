import numpy as np
import pandas as pd
from types import SimpleNamespace
import matplotlib.pyplot as plt
from scipy.optimize import root
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

class problem1:
    def __init__(self, par):
        self.par = par

    def firm_behavior(self, p_j, w):
        # Initialization
        A, gamma = self.par.A, self.par.gamma
        # Functions
        ell_j_star = (p_j * A * gamma / w) ** (1 / (1 - gamma))
        y_j_star = A * (ell_j_star) ** gamma
        pi_j_star = (1 - gamma) / gamma * w * (p_j * A * gamma / w) ** (1 / (1 - gamma))
        return ell_j_star, y_j_star, pi_j_star
    
    def firm_output(self, p_j, w):
        # Initialization
        A, gamma = self.par.A, self.par.gamma
        ell_j_star = (p_j * A * gamma / w) ** (1 / (1 - gamma))
        # Function
        y_j_star = A * ell_j_star ** gamma
        return y_j_star

    def consumer_behavior(self, p1, p2, w, tau, T):
        # Initialization
        _, _, pi1_star = self.firm_behavior(p1, w)
        _, _, pi2_star = self.firm_behavior(p2, w)
        profits = pi1_star + pi2_star

        # Budget constraint
        budget = lambda ell: w * ell + T + profits

        # Consumption given labor supply
        c1 = lambda ell: self.par.alpha * budget(ell) / p1
        c2 = lambda ell: (1 - self.par.alpha) * budget(ell) / (p2 + tau)

        # Consumer's optimization problem
        utility = lambda ell: np.log(c1(ell) ** self.par.alpha * c2(ell) ** (1 - self.par.alpha)) - self.par.nu * ell ** (1 + self.par.epsilon) / (1 + self.par.epsilon)
        ell_star = self.optimize_labor(utility)
        return c1(ell_star), c2(ell_star), ell_star

    def optimize_labor(self, utility):
        # Optimizing the utility-function wrt ell
        res = minimize_scalar(lambda ell: -utility(ell), bounds=(0, 100), method='bounded')
        return res.x

    def check_market_clearing(self, p1_vals, p2_vals, w=1):
        count = 0
        for p1 in p1_vals:
            for p2 in p2_vals:
                # Consumer optimal behavior
                c1_star, c2_star, ell_star = self.consumer_behavior(p1, p2, w, self.par.tau, self.par.T)

                # Firms optimal behavior
                ell1_star, y1_star, _ = self.firm_behavior(p1, w)
                ell2_star, y2_star, _ = self.firm_behavior(p2, w)

                # Market clearing
                labor_market_clearing = np.isclose(ell_star, ell1_star + ell2_star)
                goods_market1_clearing = np.isclose(c1_star, y1_star)
                goods_market2_clearing = np.isclose(c2_star, y2_star)

                # We count number of combinations where markets clear
                if labor_market_clearing and goods_market1_clearing and goods_market2_clearing:
                    count += 1
                    print(f"Market clears for p1: {p1}, p2: {p2}")
        
        # We create the output that will be displayed in the notebook
        if count == 0:
            print("No pairs found where markets clear.")
        else:
            print(f"Number of pairs where markets clear: {count}")
    
    def market_clearing_conditions(self, prices, w=1):
        # Initialization
        p1, p2 = prices
        c1_star, c2_star, ell_star = self.consumer_behavior(p1, p2, w, self.par.tau, self.par.T)
        ell1_star, y1_star, _ = self.firm_behavior(p1, w)
        ell2_star, y2_star, _ = self.firm_behavior(p2, w)

        # Market clearing for goodmarket 1 and goodmarket 2 (Walras' Law)
        market1_clearing = c1_star - y1_star
        market2_clearing = c2_star - y2_star

        return [market1_clearing, market2_clearing]
    
    def find_equilibrium_prices(self, initial_guess=[1, 1], w=1):
        # Finding roots for market clearing function
        solution = root(self.market_clearing_conditions, initial_guess, args=(w,))
        if solution.success:
            return solution.x
        else:
            raise ValueError("Equilibrium prices not found.")
    
    def plot_market_clearing_p1(self, p1_vals, p2_eq):
        # Empty list
        market_clearing_results = []

        for p1 in p1_vals:
            market_clearing = self.market_clearing_conditions([p1, p2_eq])
            market_clearing_results.append(market_clearing)

        market_clearing_results = np.array(market_clearing_results)

        # We plot the figure
        plt.figure(figsize=(10, 6))
        plt.plot(p1_vals, market_clearing_results[:, 0], label='Good 1 Market Clearing')
        plt.plot(p1_vals, market_clearing_results[:, 1], label='Good 2 Market Clearing')
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel('$p_1$')
        plt.ylabel('Market Clearing Condition')
        plt.title('Market Clearing Conditions as a Function of $p_1$')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_market_clearing_p2(self, p1_eq, p2_vals): # Same approach as the function above
        market_clearing_results = []
        for p2 in p2_vals:
            market_clearing = self.market_clearing_conditions([p1_eq, p2])
            market_clearing_results.append(market_clearing)

        market_clearing_results = np.array(market_clearing_results)

        plt.figure(figsize=(10, 6))
        plt.plot(p2_vals, market_clearing_results[:, 0], label='Good 1 Market Clearing')
        plt.plot(p2_vals, market_clearing_results[:, 1], label='Good 2 Market Clearing')
        plt.axhline(0, color='black', linestyle='--')
        plt.xlabel('$p_2$')
        plt.ylabel('Market Clearing Condition')
        plt.title('Market Clearing Conditions as a Function of $p_2$')
        plt.legend()
        plt.grid(True)
        plt.show()

    def social_welfare_function(self, tau, T):
        # Initialization
        p1, p2 = self.find_equilibrium_prices()
        c1_star, c2_star, ell_star = self.consumer_behavior(p1, p2, 1, tau, T)
        y2_star = self.firm_output(p2, 1)
        # The Social Welfare Function
        SWF = np.log(c1_star ** self.par.alpha * c2_star ** (1 - self.par.alpha)) - self.par.nu * ell_star ** (1 + self.par.epsilon) / (1 + self.par.epsilon) - self.par.kappa * y2_star
        return -SWF  # Minimize -SWF to maximize SWF

    def find_optimal_tax_transfer(self, initial_guess=[1.0, 0.0]):
        # We set the bound for tau and T and applies the method 'SLSQP'
        bounds = [(0, None), (None, None)]  
        method = 'SLSQP'  
        # We minimize the SWF
        result = minimize(lambda x: self.social_welfare_function(x[0], x[1]), initial_guess, method=method, bounds=bounds)
        # And store the results if success 
        if result.success:
            return result.x[0], result.x[1]
        else:
            raise ValueError("Optimization failed to find optimal tau and T.")

class problem2:
    def __init__(self, J, N, K, sigma, v, c):
        # Initialization
        self.J = J
        self.N = N
        self.K = K
        self.sigma = sigma
        self.v = v
        self.c = c

        self.results_df = None
        self.results_with_switching_df = None

        self.share_graduates_choosing_career = None
        self.average_subjective_expected_utility = None
        self.average_ex_post_realized_utility = None

        self.share_graduates_choosing_career_switching = None
        self.average_subjective_expected_utility_switching = None
        self.average_ex_post_realized_utility_switching = None
    
    def simulate(self):
        # Initialization
        expected_utility = np.zeros(self.J)
        avg_realized_utility = np.zeros(self.J)

        # Simulation loop
        for j in range(self.J):
            sum_expected_utility = 0.0
            sum_realized_utility = 0.0
            
            for k in range(self.K):
                # Draw epsilon from normal distribution
                epsilon_ij = np.random.normal(loc=0, scale=self.sigma, size=self.N)
                
                # Utility for career j
                utility_ij = self.v[j] + epsilon_ij
                
                # Expected utility
                sum_expected_utility += np.mean(utility_ij)
                
                # Average realized utility
                sum_realized_utility += np.mean(self.v[j] + epsilon_ij)
            
            # Both are divided by K
            expected_utility[j] = sum_expected_utility / self.K
            avg_realized_utility[j] = sum_realized_utility / self.K
        
        return expected_utility, avg_realized_utility
    
    def simulate_2(self, seed=7):
        # Set the seed
        np.random.seed(seed)
        # Empty list
        results = []
        
        # Simulation loop
        for k in range(self.K):
            epsilon_friends = {i: np.random.normal(0, self.sigma, (i, self.J)) for i in range(1, self.N + 1)}
            epsilon_personal = {i: np.random.normal(0, self.sigma, self.J) for i in range(1, self.N + 1)}
            
            # Simulation loop
            for i in range(1, self.N + 1):
                # Calculate prior expected utilities
                prior_expected_utilities = self.v + np.mean(epsilon_friends[i], axis=0)
                
                # Career chosen with highest expected utility
                chosen_career = np.argmax(prior_expected_utilities) + 1
                
                # Realized utility
                realized_utility = self.v[chosen_career - 1] + epsilon_personal[i][chosen_career - 1]
                
                # Append the loop-simulations to the empty list
                results.append({
                    'Graduate': i,
                    'Chosen Career': chosen_career,
                    'Prior Expected Utility': prior_expected_utilities[chosen_career - 1],
                    'Realized Utility': realized_utility
                })
        # Put the results in a DataFrame
        self.results_df = pd.DataFrame(results)
    
    def calculate_statistics(self):
        # Make sure there are results from previous function
        if self.results_df is None or len(self.results_df) == 0:
            raise ValueError("Simulation results are empty. Please run simulate() first.")
        
        # We calculate statistics
        self.share_graduates_choosing_career = self.results_df.groupby('Graduate')['Chosen Career'].value_counts(normalize=True).unstack(fill_value=0)
        self.average_subjective_expected_utility = self.results_df.groupby('Graduate')['Prior Expected Utility'].mean()
        self.average_ex_post_realized_utility = self.results_df.groupby('Graduate')['Realized Utility'].mean()

        # Used in simulation_3 (later question)
        if self.results_with_switching_df is not None and len(self.results_with_switching_df) > 0:
            self.share_graduates_choosing_career_switching = self.results_with_switching_df.groupby('Graduate')['New Chosen Career'].value_counts(normalize=True).unstack(fill_value=0)
            self.average_subjective_expected_utility_switching = self.results_with_switching_df.groupby('Graduate')['Prior Expected Utility'].mean()
            self.average_ex_post_realized_utility_switching = self.results_with_switching_df.groupby('Graduate')['New Realized Utility'].mean()
    
    def visualize_results(self):
        # Again, make sure there are results stored
        if self.results_df is None or len(self.results_df) == 0:
            raise ValueError("Simulation results are empty. Please run simulate() first.")
        
        # We plot the share of graduates choosing each career
        plt.figure(figsize=(12, 12))
        plt.subplot(3, 1, 1)
        colors = ['red', 'green', 'blue']
        for idx, career in enumerate(range(1, self.J + 1)):
            plt.plot(self.share_graduates_choosing_career.index, self.share_graduates_choosing_career[career], label=f'Career {career}', color=colors[idx])
        plt.title('Share of Graduates Choosing Each Career')
        plt.xlabel('Graduate')
        plt.ylabel('Share')
        plt.xticks(np.arange(1, self.N + 1))
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # We plot average subjective expected utility
        plt.figure(figsize=(12, 12))
        plt.subplot(3, 1, 2)
        plt.plot(self.average_subjective_expected_utility.index, self.average_subjective_expected_utility, label='Average Subjective Expected Utility', color='blue')
        plt.title('Average Subjective Expected Utility')
        plt.xlabel('Graduate')
        plt.ylabel('Avg. Exp. Utility')
        plt.xticks(np.arange(1, self.N + 1)) 
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # we plot average realized utility
        plt.figure(figsize=(12, 12))
        plt.subplot(3, 1, 3)
        plt.plot(self.average_ex_post_realized_utility.index, self.average_ex_post_realized_utility, label='Average Ex Post Realized Utility', color='blue')
        plt.title('Average Realized Utility')
        plt.xlabel('Graduate')
        plt.ylabel('Avg. Realized Utility')
        plt.xticks(np.arange(1, self.N + 1))
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    def simulate_3(self, seed=7):
        # Set the seed     
        np.random.seed(seed)
        # Create empty list
        results_with_switching = []

        # We loop
        for k in range(self.K):
            epsilon_friends = {i: np.random.normal(0, self.sigma, (i, self.J)) for i in range(1, self.N + 1)}
            epsilon_personal = {i: np.random.normal(0, self.sigma, self.J) for i in range(1, self.N + 1)}

            # We loop
            for i in range(1, self.N + 1):
                # Prior expected utilities
                prior_expected_utilities = self.v + np.mean(epsilon_friends[i], axis=0)
                
                # Career choice with highest expected utility
                chosen_career = np.argmax(prior_expected_utilities) + 1
                
                # Realized utility
                realized_utility = self.v[chosen_career - 1] + epsilon_personal[i][chosen_career - 1]
                
                # Reconsidering career choice
                new_prior_expected_utilities = prior_expected_utilities - self.c
                new_prior_expected_utilities[chosen_career - 1] = realized_utility
                
                new_chosen_career = np.argmax(new_prior_expected_utilities) + 1
                new_realized_utility = self.v[new_chosen_career - 1] + epsilon_personal[i][new_chosen_career - 1]
                
                # Append the loop-simulations to the empty list
                results_with_switching.append({
                    'Graduate': i,
                    'Initial Chosen Career': chosen_career,
                    'Initial Realized Utility': realized_utility,
                    'New Chosen Career': new_chosen_career,
                    'New Realized Utility': new_realized_utility,
                    'Prior Expected Utility': prior_expected_utilities[chosen_career - 1],
                    'Switched': chosen_career != new_chosen_career  # True if career was switched
                })

        # Store the results as a DataFrame
        self.results_with_switching_df = pd.DataFrame(results_with_switching)
    
    def visualize_results_with_switching(self):
        # Make sure there are stores results
        if self.results_with_switching_df is None or len(self.results_with_switching_df) == 0:
            raise ValueError("Simulation results with switching are empty. Please run simulate_and_switch() first.")
            
        # We plot the share of graduates choosing each career after switching
        plt.figure(figsize=(12, 12))
        plt.subplot(3, 1, 1)
        colors = ['red', 'green', 'blue']
        for idx, career in enumerate(range(1, self.J + 1)):
            plt.plot(self.share_graduates_choosing_career_switching.index, self.share_graduates_choosing_career_switching[career], label=f'Career {career}', color=colors[idx])
        plt.title('Share of Graduates Choosing Each Career after Switching')
        plt.xlabel('Graduate')
        plt.ylabel('Share')
        plt.xticks(np.arange(1, self.N + 1)) 
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # We plot the average subjective expected utility after switching
        plt.figure(figsize=(12, 12))
        plt.subplot(3, 1, 2)
        plt.plot(self.average_subjective_expected_utility_switching.index, self.average_subjective_expected_utility_switching, label='Average Subjective Expected Utility', color='blue')
        plt.title('Average Subjective Expected Utility after Switching')
        plt.xlabel('Graduate')
        plt.ylabel('Avg. Exp. Utility')
        plt.xticks(np.arange(1, self.N + 1))  
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # We plot the average realized utility after switching
        plt.figure(figsize=(12, 12))
        plt.subplot(3, 1, 3)
        plt.plot(self.average_ex_post_realized_utility_switching.index, self.average_ex_post_realized_utility_switching, label='Average Ex Post Realized Utility', color='blue')
        plt.title('Average Realized Utility after Switching')
        plt.xlabel('Graduate')
        plt.ylabel('Avg. Realized Utility')
        plt.xticks(np.arange(1, self.N + 1))  
        plt.legend()
        plt.tight_layout()
        plt.show()

        # We plot the share of graduates choosing to switch careers
        plt.figure(figsize=(10, 6))
        switching_stats = self.results_with_switching_df.groupby('Initial Chosen Career')['Switched'].mean()
        plt.bar(switching_stats.index, switching_stats.values, color='skyblue')
        plt.title('Share of Graduates Choosing to Switch Careers by Initial Career Choice')
        plt.xlabel('Initial Chosen Career')
        plt.ylabel('Share of Graduates')
        plt.xticks(range(1, self.J + 1))
        plt.ylim(0, 1)  
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()        

class problem3:
    def __init__(self, X, y):
        # Initialization
        self.X = X
        self.y = y
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.find_points()

    def find_point(self, condition):
        # We calculate distances
        distances = np.linalg.norm(self.X - self.y, axis=1)
        filtered_indices = np.where(condition)[0]
        if len(filtered_indices) == 0:
            return None
        min_index = np.argmin(distances[filtered_indices])
        return self.X[filtered_indices[min_index]]

    def find_points(self):
        # We store A, B, C and D
        self.A = self.find_point((self.X[:, 0] > self.y[0]) & (self.X[:, 1] > self.y[1]))
        self.B = self.find_point((self.X[:, 0] > self.y[0]) & (self.X[:, 1] < self.y[1]))
        self.C = self.find_point((self.X[:, 0] < self.y[0]) & (self.X[:, 1] < self.y[1]))
        self.D = self.find_point((self.X[:, 0] < self.y[0]) & (self.X[:, 1] > self.y[1]))

    def plot(self):
        # Make sure there are stored results from previous function
        if self.A is None or self.B is None or self.C is None or self.D is None:
            print("Could not find all points A, B, C, D.")
            return

        # We plot the figure that is asked for
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X[:, 0], self.X[:, 1], label='Points in X')
        plt.scatter(self.y[0], self.y[1], color='red', label='Point y')
        plt.scatter([self.A[0], self.B[0], self.C[0], self.D[0]], [self.A[1], self.B[1], self.C[1], self.D[1]], color='green', label='A, B, C, D', marker='o')
        plt.text(self.A[0], self.A[1], 'A', fontsize=12, verticalalignment='bottom')
        plt.text(self.B[0], self.B[1], 'B', fontsize=12, verticalalignment='top')
        plt.text(self.C[0], self.C[1], 'C', fontsize=12, verticalalignment='top')
        plt.text(self.D[0], self.D[1], 'D', fontsize=12, verticalalignment='bottom')

        plt.plot([self.A[0], self.B[0], self.C[0], self.A[0]], [self.A[1], self.B[1], self.C[1], self.A[1]], 'b-', label='Triangle ABC')
        plt.plot([self.C[0], self.D[0], self.A[0], self.C[0]], [self.C[1], self.D[1], self.A[1], self.C[1]], 'g-', label='Triangle CDA')

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title('Plot')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def find_coordinates(self):
        # Initilization
        A, B, C, D, y = self.A, self.B, self.C, self.D, self.y
        

        if any(p is None for p in [A, B, C, D]):
            print("")
            return None, None

        # We calculate the barycentric coordinates for ABC
        denom_ABC = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r1_ABC = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denom_ABC
        r2_ABC = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denom_ABC
        r3_ABC = 1 - r1_ABC - r2_ABC

        # We calculate the barycentric coordinates for CDA
        denom_CDA = (D[1] - A[1]) * (C[0] - A[0]) + (A[0] - D[0]) * (C[1] - A[1])
        r1_CDA = ((D[1] - A[1]) * (y[0] - A[0]) + (A[0] - D[0]) * (y[1] - A[1])) / denom_CDA
        r2_CDA = ((A[1] - C[1]) * (y[0] - A[0]) + (C[0] - A[0]) * (y[1] - A[1])) / denom_CDA
        r3_CDA = 1 - r1_CDA - r2_CDA

        return (r1_ABC, r2_ABC, r3_ABC), (r1_CDA, r2_CDA, r3_CDA)
    
    def check_point_inside_triangle(self):
        # Initialization
        (r1_ABC, r2_ABC, r3_ABC), (r1_CDA, r2_CDA, r3_CDA) = self.find_coordinates()

        # We find out if y is inside either ABC or CDA
        inside_ABC = r1_ABC >= 0 and r2_ABC >= 0 and r3_ABC >= 0
        inside_CDA = r1_CDA >= 0 and r2_CDA >= 0 and r3_CDA >= 0

        # We create prints if y is inside either of the triangle, and if it isnt at all 
        if inside_ABC:
            print(f"Point y is inside triangle ABC with barycentric coordinates ({r1_ABC:.4f}, {r2_ABC:.4f}, {r3_ABC:.4f})")
            print(f"Point y is NOT inside triangle CDA with barycentric coordinates ({r1_CDA:.4f}, {r2_CDA:.4f}, {r3_CDA:.4f})")
        elif inside_CDA:
            print(f"Point y is NOT inside triangle ABC with barycentric coordinates ({r1_ABC:.4f}, {r2_ABC:.4f}, {r3_ABC:.4f})")
            print(f"Point y is inside triangle CDA with barycentric coordinates ({r1_CDA:.4f}, {r2_CDA:.4f}, {r3_CDA:.4f})")
        else:
            print("Point y is NOT inside triangle ABC or CDA")

    def compute_and_check(self, f):
        # Initialization
        r_ABC, r_CDA = self.compute_barycentric_coordinates()
        
        # Make sure there are results stored
        if r_ABC and r_CDA:
            # We calculate approximations for the triangles
            approximation_ABC = r_ABC[0] * f(*self.A) + r_ABC[1] * f(*self.B) + r_ABC[2] * f(*self.C)
            approximation_CDA = r_CDA[0] * f(*self.C) + r_CDA[1] * f(*self.D) + r_CDA[2] * f(*self.A)
            return approximation_ABC, approximation_CDA
        return None, None
    
    def approx_vs_true(self):
        # We create the function f(x)/f(y)
        f = lambda x1, x2: x1 * x2

        # We create A, B, C and D
        A = min((x for x in self.X if x[0] > self.y[0] and x[1] > self.y[1]), key=lambda x: np.linalg.norm(x - self.y), default=np.nan)
        B = min((x for x in self.X if x[0] > self.y[0] and x[1] < self.y[1]), key=lambda x: np.linalg.norm(x - self.y), default=np.nan)
        C = min((x for x in self.X if x[0] < self.y[0] and x[1] < self.y[1]), key=lambda x: np.linalg.norm(x - self.y), default=np.nan)
        D = min((x for x in self.X if x[0] < self.y[0] and x[1] > self.y[1]), key=lambda x: np.linalg.norm(x - self.y), default=np.nan)

        # If we fail to create A, B, C and D
        if any(np.isnan(point).any() for point in [A, B, C, D]):
            print("One or more points A, B, C, D could not be found.")
            return

        # We create the denominator and the barycentric coordiantes
        denom_ABC = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r1_ABC = ((B[1] - C[1]) * (self.y[0] - C[0]) + (C[0] - B[0]) * (self.y[1] - C[1])) / denom_ABC
        r2_ABC = ((C[1] - A[1]) * (self.y[0] - C[0]) + (A[0] - C[0]) * (self.y[1] - C[1])) / denom_ABC
        r3_ABC = 1 - r1_ABC - r2_ABC

        # ...which we do for either triangles
        denom_CDA = (D[1] - A[1]) * (C[0] - A[0]) + (A[0] - D[0]) * (C[1] - A[1])
        r1_CDA = ((D[1] - A[1]) * (self.y[0] - A[0]) + (A[0] - D[0]) * (self.y[1] - A[1])) / denom_CDA
        r2_CDA = ((A[1] - C[1]) * (self.y[0] - A[0]) + (C[0] - A[0]) * (self.y[1] - A[1])) / denom_CDA
        r3_CDA = 1 - r1_CDA - r2_CDA

        # We calculate the true value of y
        fy_true = f(self.y[0], self.y[1])

        # We print the results - the if-function is used with the objective 'to end' up in the right triangle
        if r1_ABC >= 0 and r2_ABC >= 0 and r3_ABC >= 0:
            approximation_ABC = r1_ABC * f(*A) + r2_ABC * f(*B) + r3_ABC * f(*C)
            print(f"Approximation of f(y): {approximation_ABC:.3f}")
            print(f"True value of f(y):    {fy_true:.3f}")
        elif r1_CDA >= 0 and r2_CDA >= 0 and r3_CDA >= 0:
            approximation_CDA = r1_CDA * f(*C) + r2_CDA * f(*D) + r3_CDA * f(*A)
            print(f"Approximation of f(y): {approximation_CDA:.3f}")
            print(f"True value of f(y):    {fy_true:.3f}")
        else:
            print("Point y is not inside triangle ABC or CDA")

        # We calculate the difference between the true value and the approximation. And print. 
        diff = fy_true - approximation_ABC
        print(f"Difference is {diff:.3f}")

    def last_one(self, Y):
        # Create an empty list
        results = []

        # We loop over the list given. We update y, and compute new A, B, C and D for these and in the end calculate barycentric coordinates
        for y in Y:
            self.y = y 
            self.find_points()  
            r_ABC, r_CDA = self.find_coordinates()  

            # We compute the true value of y
            fy_true = y[0] * y[1]

            # We compute approximations of y - dependant on which triangle the point is inside
            if r_ABC and (r_ABC[0] >= 0 and r_ABC[1] >= 0 and r_ABC[2] >= 0):
                f_y_approx = r_ABC[0] * (self.A[0] * self.A[1]) + r_ABC[1] * (self.B[0] * self.B[1]) + r_ABC[2] * (self.C[0] * self.C[1])
            elif r_CDA and (r_CDA[0] >= 0 and r_CDA[1] >= 0 and r_CDA[2] >= 0):
                f_y_approx = r_CDA[0] * (self.C[0] * self.C[1]) + r_CDA[1] * (self.D[0] * self.D[1]) + r_CDA[2] * (self.A[0] * self.A[1])
            else:
                f_y_approx = np.nan

            # Storing the true value and the approximization
            results.append((y, fy_true, f_y_approx))

        # We print the results that will be shown in the notebook
        for result in results:
            y, f_y_true, f_y_approx = result
            print(f"Point y = {y}:")
            print(f"True value of f(y): {f_y_true:.2f}")
            print(f"Approximated value of f(y): {f_y_approx:.5f}\n")