import matplotlib.pyplot as plt
import numpy as np

def plot_initial_positions(agent_positions, targets, region_radius):
    plt.figure(figsize=(8, 8))

    # Plot agents
    plt.scatter(agent_positions[:, 0], agent_positions[:, 1], c='blue', label='Agents', s=100)

    # Plot targets
    plt.scatter(targets[:, 0], targets[:, 1], c='red', marker='x', label='Targets', s=100)

    # Add region boundary
    unit_disk = plt.Circle((0, 0), region_radius, color='black', fill=False, linestyle='--', linewidth=2)
    plt.gca().add_artist(unit_disk)

    plt.xlim(-region_radius, region_radius)
    plt.ylim(-region_radius, region_radius)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Initial Positions of Agents and Targets')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_trajectories_with_delays(position_history, agents, targets, region_radius):
    plt.figure(figsize=(8, 8))

    unit_disk = plt.Circle((0, 0), region_radius, color='black', fill=False, linestyle='--', linewidth=2)
    plt.gca().add_artist(unit_disk)

    for i, agent in enumerate(agents):
        positions = position_history[:, i, :]

        plt.plot(positions[:, 0], positions[:, 1], label=f'Agent {i+1}')
        if agent.broadcasted_positions:
            broadcasted_positions = np.array(agent.broadcasted_positions)
            plt.scatter(broadcasted_positions[:, 0], broadcasted_positions[:, 1],
                        label=f'Broadcast {i+1}', edgecolor='black', color='blue', marker='o', s=50)

    plt.scatter([target[0] for target in targets], [target[1] for target in targets], 
                c='red', marker='x', label='Target', s=100)

    plt.xlim(-region_radius, region_radius)
    plt.ylim(-region_radius, region_radius)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Agent Trajectories with Delayed Broadcasts')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_detection_error_heatmap(agents, targets, region_radius, xi_samples):
    grid_size = 100  
    x = np.linspace(-region_radius, region_radius, grid_size)
    y = np.linspace(-region_radius, region_radius, grid_size)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)

    # Compute detection error probability for each grid point
    for i in range(grid_size):
        for j in range(grid_size):
            point = np.array([X[i, j], Y[i, j]])
            detection_error_prob = np.mean(
                [np.prod([1 - agent.detection_probability(point, xi_sample) for agent in agents])
                 for xi_sample in xi_samples]
            )
            Z[i, j] = detection_error_prob

    plt.figure(figsize=(8, 8))
    plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar(label='Detection Error Probability')

    plt.scatter([agent.position[0] for agent in agents], 
                [agent.position[1] for agent in agents], 
                c='blue', label='Agent', s=50, edgecolor='black')

    plt.scatter([target[0] for target in targets], 
                [target[1] for target in targets], 
                c='red', marker='o', label='Target', s=50)

    unit_disk = plt.Circle((0, 0), region_radius, color='black', fill=False, linestyle='--', linewidth=2)
    plt.gca().add_artist(unit_disk)

    plt.xlim(-region_radius, region_radius)
    plt.ylim(-region_radius, region_radius)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Heat Map of Detection Error Probability with Targets (Red Dots) and Unit Disk')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_mean_trajectory(mean_values, std_values, label):
    plt.figure()
    plt.plot(mean_values, label=f'Mean {label}')
    plt.fill_between(range(len(mean_values)), mean_values - std_values, mean_values + std_values, alpha=0.3)
    plt.xlabel('Iterations')
    plt.ylabel(label)
    plt.legend()
    plt.show()

def plot_gradient_norms(gradient_norms, window_size=10):
    moving_avg = np.convolve(gradient_norms, np.ones(window_size) / window_size, mode='valid')
    plt.figure()
    plt.plot(moving_avg, label="Gradient Norm (Moving Avg)")
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.show()

def plot_convergence(F_values, P_values):
    epochs = len(F_values)

    plt.figure(figsize=(10, 6))
    plt.plot(F_values, label='F(x)', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('F(x)')
    plt.title('Convergence of F(x)')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(P_values, label='P(x)', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('P(x)')
    plt.title('Convergence of P(x)')
    plt.legend()
    plt.grid(True)
    plt.show()
