import matplotlib.pyplot as plt
import numpy as np

def in_unit_disk(point, region_radius):
    return np.linalg.norm(point) <= region_radius

def plot_initial_positions(agents, targets, region_radius):
    plt.figure(figsize=(8, 8))
    plt.gca().add_patch(plt.Circle((0, 0), region_radius, fill=False, color='green', linestyle='--', label='Region of Interest'))
    plt.scatter(agents[:, 0], agents[:, 1], c='b', label='Initial Agents')
    plt.scatter(targets[:, 0], targets[:, 1], c='r', label='Targets')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.title('Initial Positions of Agents and Targets')
    plt.xlim(-region_radius, region_radius)
    plt.ylim(-region_radius, region_radius)
    plt.show()

def plot_final_positions(agents, targets, region_radius):
    plt.figure(figsize=(8, 8))
    plt.gca().add_patch(plt.Circle((0, 0), region_radius, fill=False, color='green', linestyle='--', label='Region of Interest'))
    plt.scatter(agents[:, 0], agents[:, 1], c='b', label='Agents')
    plt.scatter(targets[:, 0], targets[:, 1], c='r', label='Targets')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.title('Final Positions of Agents and Targets')
    plt.xlim(-region_radius, region_radius)
    plt.ylim(-region_radius, region_radius)
    plt.show()

def plot_objective_and_penalty(F_values, P_values):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(F_values, label='F(x)')
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(P_values, label='P(x)')
    plt.xlabel('Iteration')
    plt.ylabel('Penalty Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_heat_map(agents, targets, region_radius, error_probability_fn, alpha):
    x_vals = np.linspace(-region_radius, region_radius, 100)
    y_vals = np.linspace(-region_radius, region_radius, 100)
    heat_map = np.zeros((100, 100))

    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            if in_unit_disk([x, y], region_radius):
                avg_error = 0
                for _ in range(50):  # Use a fixed number of samples for the heat map
                    xi_sample = np.random.uniform(0.5, 1.5)
                    avg_error += error_probability_fn(agents, [x, y], xi_sample, alpha)
                avg_error /= 50
                heat_map[j, i] = avg_error

    plt.figure(figsize=(8, 8))
    plt.imshow(heat_map, extent=(-region_radius, region_radius, -region_radius, region_radius), origin='lower', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Average Error Probability')
    plt.scatter(agents[:, 0], agents[:, 1], c='b', label='Agents')
    plt.scatter(targets[:, 0], targets[:, 1], c='r', label='Targets')
    plt.legend()
    plt.title('Heat Map of Average Error Probability with Agent and Target Positions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-region_radius, region_radius)
    plt.ylim(-region_radius, region_radius)
    plt.show()


