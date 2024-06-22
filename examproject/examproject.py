import numpy as np
import pandas as pd
from types import SimpleNamespace
import matplotlib.pyplot as plt


class problem1:
    def __init__(self):
        self.par = SimpleNamespace()
        self.par.A = 1.0
        self.par.gamma = 0.5

        # households
        self.par.alpha = 0.3
        self.par.nu = 1.0
        self.par.epsilon = 2.0

        # government
        self.par.tau = 0.0
        self.par.T = 0.0

        # Question 3
        self.par.kappa = 0.1

        self.p1_range = np.linspace(0.1, 2.0, 10)
        self.p2_range = np.linspace(0.1, 2.0, 10)
        
        self.results = []

    def optimal_labor(self, p):
        return (p * self.A * self.gamma / self.w) ** (1 / (1 - self.gamma))

    def optimal_output(self, labor):
        return self.A * (labor ** self.gamma)

    def implied_profits(self, p):
        labor = self.optimal_labor(p)
        return (1 - self.gamma) / self.gamma * self.w * labor

    def optimal_consumption(self, labor, p1, p2, pi1, pi2, tau):
        T = tau * ((self.w * labor + pi1 + pi2) / (p2 + tau))
        c1 = self.alpha * (self.w * labor + T + pi1 + pi2) / p1
        c2 = (1 - self.alpha) * (self.w * labor + T + pi1 + pi2) / (p2 + tau)
        return c1, c2

    def utility_function(self, labor, p1, p2, pi1, pi2, tau):
        c1, c2 = self.optimal_consumption(labor, p1, p2, pi1, pi2, tau)
        utility = np.log(c1 ** self.alpha * c2 ** (1 - self.alpha)) - self.nu * (labor ** (1 + self.epsilon)) / (1 + self.epsilon)
        return utility

    def solve_consumer_problem(self):
        max_utility = -np.inf
        optimal_labor = None
        optimal_c1 = None
        optimal_c2 = None
        
        for p1 in self.p1_range:
            for p2 in self.p2_range:
                for tau in [0.0]:  # Considering only tau = 0 initially
                    for T in [0.0]:  # Considering only T = 0 initially
                        labor_guess = 1.0  # Initial guess for labor
                        
                        # Define the utility function to maximize
                        objective = lambda labor: -self.utility_function(labor, p1, p2, self.implied_profits(p1), self.implied_profits(p2), tau)
                        
                        # Optimize labor
                        from scipy.optimize import minimize_scalar
                        res = minimize_scalar(objective)
                        if res.success:
                            current_utility = -res.fun
                            if current_utility > max_utility:
                                max_utility = current_utility
                                optimal_labor = res.x
                                optimal_c1, optimal_c2 = self.optimal_consumption(optimal_labor, p1, p2, self.implied_profits(p1), self.implied_profits(p2), tau)
        
        return {
            'optimal_labor': optimal_labor,
            'optimal_c1': optimal_c1,
            'optimal_c2': optimal_c2,
            'max_utility': max_utility
        }

    def check_market_clearing(self):
        consumer_solution = self.solve_consumer_problem()
        optimal_labor = consumer_solution['optimal_labor']
        optimal_c1 = consumer_solution['optimal_c1']
        optimal_c2 = consumer_solution['optimal_c2']
        
        for p1 in self.p1_range:
            for p2 in self.p2_range:
                labor1 = self.optimal_labor(p1)
                labor2 = self.optimal_labor(p2)
                output1 = self.optimal_output(labor1)
                output2 = self.optimal_output(labor2)
                pi1 = self.implied_profits(p1)
                pi2 = self.implied_profits(p2)
                
                total_labor = labor1 + labor2
                c1, c2 = self.optimal_consumption(total_labor, p1, p2, pi1, pi2, self.tau)
                
                # Market clearing conditions
                labor_clearing = np.isclose(total_labor, optimal_labor)
                good1_clearing = np.isclose(c1, optimal_c1)
                good2_clearing = np.isclose(c2, optimal_c2)
                
                self.results.append({
                    'p1': p1,
                    'p2': p2,
                    'Labor Clearing': labor_clearing,
                    'Good1 Clearing': good1_clearing,
                    'Good2 Clearing': good2_clearing
                })
    
    def get_results_df(self):
        return pd.DataFrame(self.results)


class problem3:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.find_points()

    def find_point(self, condition):
        distances = np.linalg.norm(self.X - self.y, axis=1)
        filtered_indices = np.where(condition)[0]
        if len(filtered_indices) == 0:
            return None
        min_index = np.argmin(distances[filtered_indices])
        return self.X[filtered_indices[min_index]]

    def find_points(self):
        self.A = self.find_point((self.X[:, 0] > self.y[0]) & (self.X[:, 1] > self.y[1]))
        self.B = self.find_point((self.X[:, 0] > self.y[0]) & (self.X[:, 1] < self.y[1]))
        self.C = self.find_point((self.X[:, 0] < self.y[0]) & (self.X[:, 1] < self.y[1]))
        self.D = self.find_point((self.X[:, 0] < self.y[0]) & (self.X[:, 1] > self.y[1]))

    def plot(self):
        if self.A is None or self.B is None or self.C is None or self.D is None:
            print("Could not find all points A, B, C, D.")
            return

        plt.figure(figsize=(8, 6))
        plt.scatter(self.X[:, 0], self.X[:, 1], label='Points in X')
        plt.scatter(self.y[0], self.y[1], color='red', marker='s', label='Point y')
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
        A, B, C, D, y = self.A, self.B, self.C, self.D, self.y
        
        if any(p is None for p in [A, B, C, D]):
            print("Could not find all points A, B, C, D.")
            return None, None

        # Compute the barycentric coordinates for triangle ABC
        denom_ABC = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r1_ABC = ((B[1] - C[1]) * (y[0] - C[0]) + (C[0] - B[0]) * (y[1] - C[1])) / denom_ABC
        r2_ABC = ((C[1] - A[1]) * (y[0] - C[0]) + (A[0] - C[0]) * (y[1] - C[1])) / denom_ABC
        r3_ABC = 1 - r1_ABC - r2_ABC

        # Compute the barycentric coordinates for triangle CDA
        denom_CDA = (D[1] - A[1]) * (C[0] - A[0]) + (A[0] - D[0]) * (C[1] - A[1])
        r1_CDA = ((D[1] - A[1]) * (y[0] - A[0]) + (A[0] - D[0]) * (y[1] - A[1])) / denom_CDA
        r2_CDA = ((A[1] - C[1]) * (y[0] - A[0]) + (C[0] - A[0]) * (y[1] - A[1])) / denom_CDA
        r3_CDA = 1 - r1_CDA - r2_CDA

        return (r1_ABC, r2_ABC, r3_ABC), (r1_CDA, r2_CDA, r3_CDA)
    
    def check_point_inside_triangle(self):
        (r1_ABC, r2_ABC, r3_ABC), (r1_CDA, r2_CDA, r3_CDA) = self.find_coordinates()

        inside_ABC = r1_ABC >= 0 and r2_ABC >= 0 and r3_ABC >= 0
        inside_CDA = r1_CDA >= 0 and r2_CDA >= 0 and r3_CDA >= 0

        if inside_ABC:
            print(f"Point y is inside triangle ABC with barycentric coordinates ({r1_ABC:.4f}, {r2_ABC:.4f}, {r3_ABC:.4f})")
            print(f"Point y is not inside triangle CDA with barycentric coordinates ({r1_CDA:.4f}, {r2_CDA:.4f}, {r3_CDA:.4f})")
        elif inside_CDA:
            print(f"Point y is not inside triangle ABC with barycentric coordinates ({r1_ABC:.4f}, {r2_ABC:.4f}, {r3_ABC:.4f})")
            print(f"Point y is inside triangle CDA with barycentric coordinates ({r1_CDA:.4f}, {r2_CDA:.4f}, {r3_CDA:.4f})")
        else:
            print("Point y is not inside triangle ABC or CDA")

    def compute_and_check(self, f):
        r_ABC, r_CDA = self.compute_barycentric_coordinates()
        if r_ABC and r_CDA:
            approximation_ABC = r_ABC[0] * f(*self.A) + r_ABC[1] * f(*self.B) + r_ABC[2] * f(*self.C)
            approximation_CDA = r_CDA[0] * f(*self.C) + r_CDA[1] * f(*self.D) + r_CDA[2] * f(*self.A)
            return approximation_ABC, approximation_CDA
        return None, None
    
    def approx_vs_true(self):
        f = lambda x1, x2: x1 * x2

        A = min((x for x in self.X if x[0] > self.y[0] and x[1] > self.y[1]), key=lambda x: np.linalg.norm(x - self.y), default=np.nan)
        B = min((x for x in self.X if x[0] > self.y[0] and x[1] < self.y[1]), key=lambda x: np.linalg.norm(x - self.y), default=np.nan)
        C = min((x for x in self.X if x[0] < self.y[0] and x[1] < self.y[1]), key=lambda x: np.linalg.norm(x - self.y), default=np.nan)
        D = min((x for x in self.X if x[0] < self.y[0] and x[1] > self.y[1]), key=lambda x: np.linalg.norm(x - self.y), default=np.nan)

        if any(np.isnan(point).any() for point in [A, B, C, D]):
            print("One or more points A, B, C, D could not be found.")
            return

        denom_ABC = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
        r1_ABC = ((B[1] - C[1]) * (self.y[0] - C[0]) + (C[0] - B[0]) * (self.y[1] - C[1])) / denom_ABC
        r2_ABC = ((C[1] - A[1]) * (self.y[0] - C[0]) + (A[0] - C[0]) * (self.y[1] - C[1])) / denom_ABC
        r3_ABC = 1 - r1_ABC - r2_ABC

        denom_CDA = (D[1] - A[1]) * (C[0] - A[0]) + (A[0] - D[0]) * (C[1] - A[1])
        r1_CDA = ((D[1] - A[1]) * (self.y[0] - A[0]) + (A[0] - D[0]) * (self.y[1] - A[1])) / denom_CDA
        r2_CDA = ((A[1] - C[1]) * (self.y[0] - A[0]) + (C[0] - A[0]) * (self.y[1] - A[1])) / denom_CDA
        r3_CDA = 1 - r1_CDA - r2_CDA

        fy_true = f(self.y[0], self.y[1])

        if r1_ABC >= 0 and r2_ABC >= 0 and r3_ABC >= 0:
            approximation_ABC = r1_ABC * f(*A) + r2_ABC * f(*B) + r3_ABC * f(*C)
            print(f"Point y is inside triangle ABC with barycentric coordinates ({r1_ABC:.3f}, {r2_ABC:.3f}, {r3_ABC:.3f})")
            print(f"Approximation of f(y): {approximation_ABC:.3f}")
            print(f"True value of f(y):    {fy_true:.3f}")
        elif r1_CDA >= 0 and r2_CDA >= 0 and r3_CDA >= 0:
            approximation_CDA = r1_CDA * f(*C) + r2_CDA * f(*D) + r3_CDA * f(*A)
            print(f"Point y is inside triangle CDA with barycentric coordinates ({r1_CDA:.3f}, {r2_CDA:.3f}, {r3_CDA:.3f})")
            print(f"Approximation of f(y): {approximation_CDA:.3f}")
            print(f"True value of f(y):    {fy_true:.3f}")
        else:
            print("Point y is not inside triangle ABC or CDA")

        diff = fy_true - approximation_ABC
        print(f"Difference is {diff:.3f}")

    def last_one(self, Y):
        results = []

        for y in Y:
            self.y = y  # Update self.y to current y
            self.find_points()  # Recompute A, B, C, D for the new y
            r_ABC, r_CDA = self.find_coordinates()  # Compute barycentric coordinates

            # Compute true value of f(y)
            fy_true = y[0] * y[1]

            # Compute approximate value of f(y)
            if r_ABC and (r_ABC[0] >= 0 and r_ABC[1] >= 0 and r_ABC[2] >= 0):
                f_y_approx = r_ABC[0] * (self.A[0] * self.A[1]) + r_ABC[1] * (self.B[0] * self.B[1]) + r_ABC[2] * (self.C[0] * self.C[1])
            elif r_CDA and (r_CDA[0] >= 0 and r_CDA[1] >= 0 and r_CDA[2] >= 0):
                f_y_approx = r_CDA[0] * (self.C[0] * self.C[1]) + r_CDA[1] * (self.D[0] * self.D[1]) + r_CDA[2] * (self.A[0] * self.A[1])
            else:
                f_y_approx = np.nan

            # Store the results
            results.append((y, fy_true, f_y_approx))

        # Print results
        for result in results:
            y, f_y_true, f_y_approx = result
            print(f"Point y = {y}:")
            print(f"  True value of f(y): {f_y_true:.2f}")
            print(f"  Approximated value of f(y): {f_y_approx:.5f}")
            print()