import numpy as np
import matplotlib.pyplot as plt

# 1. Vanilla NAG
def vanilla_nag(grad, x0, lr, mu, iters):
    theta = np.array(x0, dtype=float)
    phi_prev = np.array(x0, dtype=float)
    trajectory = [theta.copy()]
    for _ in range(iters):
        g = grad(theta)
        phi = theta - lr * g
        theta = phi + mu * (phi - phi_prev)
        phi_prev = phi.copy()
        trajectory.append(theta.copy())
    return np.array(trajectory)

# 2. Sutskever Formulation
def sutskever_nag(grad, x0, lr, mu, iters):
    phi = np.array(x0, dtype=float)
    v = np.zeros_like(phi)
    theta = phi + mu * v
    trajectory = [theta.copy()]
    for _ in range(iters):
        theta_temp = phi + mu * v
        g = grad(theta_temp)
        v = mu * v - lr * g
        phi = phi + v
        theta_next = phi + mu * v
        trajectory.append(theta_next.copy())
    return np.array(trajectory)

# 3. Bengio Formulation
def bengio_nag(grad, x0, lr, mu, iters):
    theta = np.array(x0, dtype=float)
    v = np.zeros_like(theta)
    trajectory = [theta.copy()]
    for _ in range(iters):
        g = grad(theta)
        v = mu * v - lr * g
        theta = theta + mu * v - lr * g
        trajectory.append(theta.copy())
    return np.array(trajectory)

# Problem setup
Q = np.array([[20, 0], [0, 1]])
def f(x):
    return 0.5 * x.T @ Q @ x

def grad(x): 
    return Q @ x

x0 = np.array([10.0, 10.0])
lr = 0.04
mu = 0.9
iters = 50

# Run optimizers
traj_vanilla = vanilla_nag(grad, x0, lr, mu, iters)
traj_sutskever = sutskever_nag(grad, x0, lr, mu, iters)
traj_bengio = bengio_nag(grad, x0, lr, mu, iters)

# Calculate function values
vals_vanilla = [f(x) for x in traj_vanilla]
vals_sutskever = [f(x) for x in traj_sutskever]
vals_bengio = [f(x) for x in traj_bengio]

# Plotting function values
plt.clf()
plt.plot(range(len(vals_vanilla)), vals_vanilla, 'o-', label='Vanilla NAG', alpha=0.6, markersize=8, linewidth=3)
plt.plot(range(len(vals_sutskever)), vals_sutskever, 'x-', label='Sutskever NAG', alpha=0.8, markersize=6, linewidth=2)
plt.plot(range(len(vals_bengio)), vals_bengio, '.-', label='Bengio NAG', alpha=1.0, markersize=3, linewidth=1)

plt.yscale('log')
plt.title('Function Value vs. Iterations (Log Scale)')
plt.xlabel('Iteration')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.show()

print("Max difference (Vanilla vs Sutskever) in f(x):", np.max(np.abs(np.array(vals_vanilla) - np.array(vals_sutskever))))
print("Max difference (Vanilla vs Bengio) in f(x):", np.max(np.abs(np.array(vals_vanilla) - np.array(vals_bengio))))
