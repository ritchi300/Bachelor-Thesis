import numpy as np
import matplotlib.pyplot as plt
import plot_module as pm

# Define constants for the problem
D = 10  # Number of agents
region_radius = 1
Y_num = 8  # Number of target points
alpha = 5  # Increased scaling factor for smaller detection radius
learning_rate = 0.02  # Increased learning rate
epochs = 5000
tolerance = 1e-3
delta = 0.001
num_samples = 50
detection_threshold = 0.1  # Threshold distance for detection
tau = 200  # Information delay in steps

def detection_probability(xi, y, xi_sample, alpha):
    return np.exp(-xi_sample * (alpha * np.linalg.norm(xi - y)) ** 2)

def sensor_detection(agent, targets, detection_threshold, pursued_targets):
    for target in targets:
        if np.linalg.norm(agent - target) < detection_threshold and not np.any(np.all(pursued_targets == target, axis=1)):
            return target
    return None

def move_towards(agent, target, learning_rate):
    direction = target - agent
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    return agent + learning_rate * direction

def roam(agent, region_radius, learning_rate):
    direction = np.random.uniform(-1, 1, 2)
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    new_position = agent + learning_rate * direction
    if np.linalg.norm(new_position) <= region_radius:
        return new_position
    else:
        return agent

def fit(agents, targets, alpha, tau):
    F_values = []
    P_values = []
    pursued_targets = np.array([]).reshape(0, 2)  # Initialize an empty array to store pursued targets
    position_history = [agents.copy()] * (tau + 1)  # Initialize history of positions with tau+1 entries

    for epoch in range(epochs):
        pursued_targets = np.array([]).reshape(0, 2)  # Reset pursued targets at the beginning of each epoch
        delayed_agents = position_history[0]  # Use the oldest positions in the history (tau steps ago)

        for i, agent in enumerate(agents):
            target = sensor_detection(delayed_agents[i], targets, detection_threshold, pursued_targets)
            if target is not None:
                agents[i] = move_towards(agent, target, learning_rate)
                pursued_targets = np.vstack([pursued_targets, target])  # Add target to pursued targets list
            else:
                agents[i] = roam(agent, region_radius, learning_rate)

        # Update the position history
        position_history.append(agents.copy())
        if len(position_history) > tau + 1:
            position_history.pop(0)

        # Compute objective and penalty values for logging
        obj_val = F(agents, alpha, targets)
        penalty_val = P(agents, alpha, targets)
        F_values.append(obj_val)
        P_values.append(penalty_val)
        print(f"Epoch {epoch}: Objective Value {obj_val}, Penalty Value {penalty_val}")

    return agents, F_values, P_values

def F(agents, alpha, targets):
    avg_error = 0
    xi_samples = np.random.uniform(0.5, 1.5, num_samples)
    for y in targets:
        avg_error += np.mean([error_probability(agents, y, xi_sample, alpha) for xi_sample in xi_samples])
    avg_error /= len(targets)
    return avg_error

def P(agents, alpha, targets):
    penalty = 0
    xi_samples = np.random.uniform(0.5, 1.5, num_samples)
    for y in targets:
        penalty += np.mean([max(0, error_probability(agents, y, xi_sample, alpha) - delta) ** 2 for xi_sample in xi_samples])
    penalty /= num_samples
    return penalty

def error_probability(agents, y, xi_sample, alpha):
    detection_probs = np.exp(-xi_sample * (alpha * np.linalg.norm(agents - y, axis=1)) ** 2)
    return np.prod(1 - detection_probs)

def in_unit_disk(point):
    return np.linalg.norm(point) <= 1

# Randomly initialize agent positions within the unit disk
np.random.seed(65)
agents = np.random.uniform(-region_radius, region_radius, (D, 2))
agents = agents[np.sqrt(np.sum(agents**2, axis=1)) <= region_radius]

# Ensure all agents are within the unit disk
agents = np.array([point if in_unit_disk(point) else point / np.linalg.norm(point) for point in agents])

# Define target points
Y = np.random.uniform(-region_radius, region_radius, (Y_num, 2))
Y = Y[np.sqrt(np.sum(Y**2, axis=1)) <= region_radius]

# Plot initial positions of agents and targets
pm.plot_initial_positions(agents, Y, region_radius)

# Fit the model
agents, F_values, P_values = fit(agents, Y, alpha, tau)

# Plot F(x) and P(x) over iterations
pm.plot_objective_and_penalty(F_values, P_values)

# Plot final positions of agents and targets
pm.plot_final_positions(agents, Y, region_radius)

# Plot heat map of average error probability
pm.plot_heat_map(agents, Y, region_radius, error_probability, alpha)