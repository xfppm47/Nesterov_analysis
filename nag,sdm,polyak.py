import numpy as np
import matplotlib.pyplot as plt


def sdm(f, grad, x_initial, alpha, iterations):
    """General Descent Method (Steepest Descent with fixed alpha)."""
    x = np.array(x_initial).astype(float).copy()
    value = [f(x)]
    for _ in range(iterations):
        x = x - alpha * grad(x)
        value.append(f(x))
    return value

def polyak(f, grad, x_initial, alpha, iterations, beta):
    """Polyak Heavy Ball (Momentum evaluated at current point)."""
    x = np.array(x_initial).astype(float).copy()
    x_prev = x.copy()
    value = [f(x)]
    for _ in range(iterations):
        g = grad(x)
        x_new = x - alpha * g + beta * (x - x_prev)
        x_prev, x = x.copy(), x_new
        value.append(f(x))
    return value

def nag_fixed(f, grad, x_initial, alpha, iterations, beta):
    """Nesterov Accelerated Gradient with Fixed Momentum."""
    x = np.array(x_initial).astype(float).copy()
    x_prev = x.copy()
    value = [f(x)]
    for _ in range(iterations):
        y = x + beta * (x - x_prev) 
        g = grad(y)                  
        x_prev, x = x.copy(), y - alpha * g
        value.append(f(x))
    return value

def nag_adaptive(f, grad, x_initial, alpha, iterations):
    """NAG with Nesterov's Dynamic Schedule (Adaptive Beta)."""
    x = np.array(x_initial).astype(float).copy()
    x_prev = x.copy()
    value = [f(x)]
    t = 0
    for _ in range(iterations):
        t_next = (1 + np.sqrt(1 + 4 * t**2)) / 2
        beta_k = (t - 1) / t_next
        t = t_next
        y = x + beta_k * (x - x_prev)
        g = grad(y)
        x_prev, x = x.copy(), y - alpha * g
        value.append(f(x))
    return value


def plot_comparison(results, title):
    plt.figure(figsize=(10, 6))
    for name, vals in results.items():
        plt.plot(vals, label=name, linewidth=2)
    plt.yscale('log')
    plt.title(title, fontsize=14)
    plt.xlabel("Iterations")
    plt.ylabel("f(x) value (Log Scale)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()


def run_combined_analysis():
    iters = 100
    beta = 0.9

    # --- CASE 1: Simple Functions (x^2 and x^4) ---
    f1 = lambda x: np.sum(x**4)
    g1 = lambda x: 4*x**3
    x0_1 = np.array([2.0])
    res1 = {
        "SDM": sdm(f1, g1, x0_1, 0.01, iters),
        "Polyak": polyak(f1, g1, x0_1, 0.01, iters, beta),
        "NAG Fixed": nag_fixed(f1, g1, x0_1, 0.01, iters, beta),
        "NAG Adaptive": nag_adaptive(f1, g1, x0_1, 0.01, iters)
    }
    plot_comparison(res1, "Case 1: Simple Function $x^4$")

    # --- CASE 2: Changing Q Matrix (Quadratic) ---
    Q2 = np.array([[20, 0], [0, 1]])
    f2 = lambda x: 0.5 * x.T @ Q2 @ x
    g2 = lambda x: Q2 @ x
    x0_2 = np.array([10., 10.])
    res2 = {
        "SDM": sdm(f2, g2, x0_2, 0.04, iters),
        "Polyak": polyak(f2, g2, x0_2, 0.04, iters, 0.8),
        "NAG Fixed": nag_fixed(f2, g2, x0_2, 0.04, iters, 0.8),
        "NAG Adaptive": nag_adaptive(f2, g2, x0_2, 0.04, iters)
    }
    plot_comparison(res2, "Case 2: Quadratic with Modified Q Matrix")

    # --- CASE 3: Larger Q Matrix (n=50) ---
    n3 = 50
    Q3 = np.diag(np.linspace(1, 10, n3))
    f3 = lambda x: 0.5 * x.T @ Q3 @ x
    g3 = lambda x: Q3 @ x
    x0_3 = np.ones(n3) * 5
    res3 = {
        "SDM": sdm(f3, g3, x0_3, 0.1, iters),
        "Polyak": polyak(f3, g3, x0_3, 0.1, iters, beta),
        "NAG Fixed": nag_fixed(f3, g3, x0_3, 0.1, iters, beta),
        "NAG Adaptive": nag_adaptive(f3, g3, x0_3, 0.1, iters)
    }
    plot_comparison(res3, f"Case 3: Larger Q Matrix ($n={n3}$)")

    # --- CASE 4: Large K(Q) (Condition Number = 1000) ---
    Q4 = np.diag([1000, 1])
    f4 = lambda x: 0.5 * x.T @ Q4 @ x
    g4 = lambda x: Q4 @ x
    x0_4 = np.array([10., 1.])
    res4 = {
        "SDM": sdm(f4, g4, x0_4, 0.001, iters),
        "Polyak": polyak(f4, g4, x0_4, 0.001, iters, beta),
        "NAG Fixed": nag_fixed(f4, g4, x0_4, 0.001, iters, beta),
        "NAG Adaptive": nag_adaptive(f4, g4, x0_4, 0.001, iters)
    }
    plot_comparison(res4, "Case 4: Ill-conditioned Quadratic ($K(Q)=1000$)")

    # --- CASE 5: Different Alpha (Sensitivity) ---
    res5 = {}
    for a in [0.00001, 0.0005, 0.0002, 0.0005, 0.001]:
        res5[f"NAG Adaptive (a={a})"] = nag_adaptive(f4, g4, x0_4, a, iters)
    plot_comparison(res5, "Case 5: NAG Adaptive with Varying Alpha")

    # --- CASE 6: General Benchmark Function (Rosenbrock) ---
    f6 = lambda x: (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
    g6 = lambda x: np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])
    x0_6 = np.array([-1.2, 1.0])
    res6 = {
        "SDM": sdm(f6, g6, x0_6, 0.001, 200),
        "Polyak": polyak(f6, g6, x0_6, 0.001, 200, beta),
        "NAG Fixed": nag_fixed(f6, g6, x0_6, 0.001, 200, beta),
        "NAG Adaptive": nag_adaptive(f6, g6, x0_6, 0.001, 200)
    }
    plot_comparison(res6, "Case 6: General Benchmark (Rosenbrock)")

if __name__ == "__main__":
    run_combined_analysis()

