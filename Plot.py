

import matplotlib.pyplot as plt
import numpy as np

def plot_initial_positions(agent_positions, targets, region_radius, filename=None):
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

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_trajectories_with_delays(position_history, agents, targets, region_radius, filename=None):
    plt.figure(figsize=(8, 8))

    # Draw the region boundary
    unit_disk = plt.Circle((0, 0), region_radius, color='black', fill=False, linestyle='--', linewidth=2)
    plt.gca().add_artist(unit_disk)

    # Lists to collect legend handles and labels
    handles = []
    labels = []

    # Plot agent trajectories and collect handles and labels
    for i, agent in enumerate(agents):
        positions = position_history[:, i, :]
        (line,) = plt.plot(positions[:, 0], positions[:, 1], label=f'Agent {i+1}')
        handles.append(line)
        labels.append(f'Agent {i+1}')
        if agent.broadcasted_positions:
            broadcasted_positions = np.array(agent.broadcasted_positions)
            scatter = plt.scatter(
                broadcasted_positions[:, 0], broadcasted_positions[:, 1],
                edgecolor='black', color='blue', marker='o', s=50
            )
            handles.append(scatter)
            labels.append(f'Broadcast {i+1}')

    # Plot targets and collect handle and label
    target_scatter = plt.scatter(
        [target[0] for target in targets], [target[1] for target in targets],
        c='red', marker='x', s=100
    )
    handles.append(target_scatter)
    labels.append('Target')

    # Set plot limits and labels
    plt.xlim(-region_radius, region_radius)
    plt.ylim(-region_radius, region_radius)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Agent Trajectories with Delayed Broadcasts')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    # Create a custom legend
    plt.legend(handles=handles, labels=labels, loc='best')

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_detection_error_heatmap(agents, targets, region_radius, xi_samples, filename=None):
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
    plt.title('Heat Map of Detection Error Probability')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_mean_trajectory(mean_values, std_values, label, title='', filename=None):
    plt.figure()
    plt.plot(mean_values, label=f'Mean {label}')
    plt.fill_between(range(len(mean_values)), mean_values - std_values, mean_values + std_values, alpha=0.3)
    plt.xlabel('Iterations')
    plt.ylabel(label)
    plt.title(title)
    plt.legend()

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_gradient_norms(gradient_norms, window_size=10, title='', filename=None):
    moving_avg = np.convolve(gradient_norms, np.ones(window_size) / window_size, mode='valid')
    plt.figure()
    plt.plot(moving_avg, label="Gradient Norm (Moving Avg)")
    plt.xlabel('Iterations')
    plt.ylabel('Gradient Norm')
    plt.title(title)
    plt.legend()

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_convergence(F_values, P_values, filename_prefix=None):
    epochs = len(F_values)

    plt.figure(figsize=(10, 6))
    plt.plot(F_values, label='F(x)', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('F(x)')
    plt.title('Convergence of F(x)')
    plt.legend()
    plt.grid(True)

    if filename_prefix:
        plt.savefig(f'{filename_prefix}_F_x_convergence.png')
        plt.close()
    else:
        plt.show()
        plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(P_values, label='P(x)', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('P(x)')
    plt.title('Convergence of P(x)')
    plt.legend()
    plt.grid(True)

    if filename_prefix:
        plt.savefig(f'{filename_prefix}_P_x_convergence.png')
        plt.close()
    else:
        plt.show()
        plt.close()
