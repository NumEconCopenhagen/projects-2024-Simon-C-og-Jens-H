import numpy as np
import pandas as pd
from types import SimpleNamespace
import matplotlib.pyplot as plt


class problem_1:
    def __init__(self, A, gamma, alpha, nu, epsilon, tau, T):
        # Initialize parameters
        self.A = A
        self.gamma = gamma
        self.alpha = alpha
        self.nu = nu
        self.epsilon = epsilon
        self.tau = tau
        self.T = T
        self.w = 1.0  # Wage (numeraire), assumed to be 1 for simplicity
        
        # Price ranges
        self.p1_range = np.linspace(0.1, 2.0, 10)
        self.p2_range = np.linspace(0.1, 2.0, 10)
        
        # Results storage
        self.results_df = None
    
    def optimal_labor(self, p, A, w, gamma):
        return (p * A * gamma / w) ** (1 / (1 - gamma))
    
    def optimal_output(self, labor, A, gamma):
        return A * (labor ** gamma)
    
    def implied_profits(self, p, A, w, gamma):
        labor = self.optimal_labor(p, A, w, gamma)
        return (1 - gamma) / gamma * w * labor
    
    def optimal_consumption(self, w, labor, alpha, p1, p2, pi1, pi2, tau):
        T = tau * (pi1 + pi2) / (p2 + tau)
        c1 = alpha * (w * labor + T + pi1 + pi2) / p1
        c2 = (1 - alpha) * (w * labor + T + pi1 + pi2) / (p2 + tau)
        return c1, c2
    
    def utility_function(self, c1, c2, alpha, nu, labor, epsilon):
        utility = np.log(c1 ** alpha * c2 ** (1 - alpha)) - nu * (labor ** (1 + epsilon)) / (1 + epsilon)
        return utility
    
    def solve(self):
        results = []

        for p1 in self.p1_range:
            for p2 in self.p2_range:
                # Firm 1
                labor1 = self.optimal_labor(p1, self.A, self.w, self.gamma)
                output1 = self.optimal_output(labor1, self.A, self.gamma)
                pi1 = self.implied_profits(p1, self.A, self.w, self.gamma)

                # Firm 2
                labor2 = self.optimal_labor(p2, self.A, self.w, self.gamma)
                output2 = self.optimal_output(labor2, self.A, self.gamma)
                pi2 = self.implied_profits(p2, self.A, self.w, self.gamma)

                # Total labor and consumption
                total_labor = labor1 + labor2
                c1, c2 = self.optimal_consumption(self.w, total_labor, self.alpha, p1, p2, pi1, pi2, self.tau)

                # Calculate utility
                utility = self.utility_function(c1, c2, self.alpha, self.nu, total_labor, self.epsilon)

                # Market clearing conditions
                labor_clearing = np.isclose(total_labor, labor1 + labor2)
                good1_clearing = np.isclose(c1, output1)
                good2_clearing = np.isclose(c2, output2)

                results.append({
                    'p1': p1,
                    'p2': p2,
                    'Labor Clearing': labor_clearing,
                    'Good1 Clearing': good1_clearing,
                    'Good2 Clearing': good2_clearing,
                    'Utility': utility
                })

        # Convert results to DataFrame
        self.results_df = pd.DataFrame(results)
    
    def plot_results(self):
        if self.results_df is None or self.results_df.empty:
            print("No market clearing prices found.")
            return

        # Extract results where labor market clears
        labor_clear = self.results_df[self.results_df['Labor Clearing']]
        all_clear = self.results_df[self.results_df['Labor Clearing'] & self.results_df['Good1 Clearing'] & self.results_df['Good2 Clearing']]
        
        # Plotting
        plt.figure(figsize=(12, 8))

        # Plot for Labor Clearing
        plt.subplot(2, 2, 1)
        if not labor_clear.empty:
            plt.scatter(labor_clear['p1'], labor_clear['p2'], c='black', label='Labor Clearing', marker='o')
        plt.xlabel('p1')
        plt.ylabel('p2')
        plt.title('Labor Market Clearing')
        plt.legend()
        plt.grid(True)

        # Plot for Good1 Clearing
        plt.subplot(2, 2, 2)
        plt.scatter(all_clear['p1'], all_clear['p2'], c='black', label='Good1 Clearing', marker='o')
        plt.xlabel('p1')
        plt.ylabel('p2')
        plt.title('Good1 Market Clearing')
        plt.legend()
        plt.grid(True)

        # Plot for Good2 Clearing
        plt.subplot(2, 2, 3)
        plt.scatter(all_clear['p1'], all_clear['p2'], c='black', label='Good2 Clearing', marker='o')
        plt.xlabel('p1')
        plt.ylabel('p2')
        plt.title('Good2 Market Clearing')
        plt.legend()
        plt.grid(True)

        # Plot for All Clearing Conditions
        plt.subplot(2, 2, 4)
        plt.scatter(all_clear['p1'], all_clear['p2'], c='black', label='All Markets Clearing', marker='o')
        plt.xlabel('p1')
        plt.ylabel('p2')
        plt.title('All Markets Clearing')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


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