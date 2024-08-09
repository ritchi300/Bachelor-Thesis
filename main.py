import numpy as np
import matplotlib.pyplot as plt
import plot_module as pm

# Define constants 
D = 8  # Number of agents
region_radius = 4  
Y_num = 8  # Number of target points
alpha = 5  # Increased scaling factor for smaller detection radius
epochs = n = 6000  # number of iterations
delta = 0.001
num_samples = 80
detection_threshold = 0.1  # Threshold distance for detection
tau = 5  # Information delay in steps

# Define a Unit Disk
def in_unit_disk(point):
    return np.linalg.norm(point) <= region_radius

# Randomly initialize agent positions within the unit disk
np.random.seed(67)
agents = np.random.uniform(-region_radius, region_radius, (D, 2))
agents = agents[np.sqrt(np.sum(agents**2, axis=1)) <= region_radius]
agents = np.array([point if in_unit_disk(point) else point / np.linalg.norm(point) for point in agents])
original_agents = agents.copy()

# Define target points
Y = np.random.uniform(-region_radius, region_radius, (Y_num, 2))
Y = Y[np.sqrt(np.sum(Y**2, axis=1)) <= region_radius]

# Plot initial positions of agents and targets
pm.plot_initial_positions(agents, Y, region_radius)

# Function to calculate the detection probability
def detection_probability(agent, y, xi_sample, alpha):
    return np.exp(-xi_sample * (alpha * np.linalg.norm(agent - y)) ** 2)

# Function for sensor detection
def sensor_detection(agent, targets, detection_threshold, pursued_targets):
    for target in targets:
        if np.linalg.norm(agent - target) < detection_threshold and not np.any(np.all(pursued_targets == target, axis=1)):
            return target
    return None

# Function to calculate the average error
def F(agents, alpha, targets):
    avg_error = 0
    xi_samples = np.random.uniform(0.5, 1.5, num_samples)
    for y in targets:
        avg_error += np.mean([error_probability(agents, y, xi_sample, alpha) for xi_sample in xi_samples])
    avg_error /= len(targets)
    return avg_error

# Function to calculate the penalty
def P(agents, alpha, targets):
    penalty = 0
    xi_samples = np.random.uniform(0.5, 1.5, num_samples)
    for y in targets:
        penalty += np.mean([max(0, error_probability(agents, y, xi_sample, alpha) - delta) ** 2 for xi_sample in xi_samples])
    penalty /= len(targets)
    return penalty

# Function to calculate the error probability
def error_probability(agents, y, xi_sample, alpha):
    detection_probs = np.exp(-xi_sample * (alpha * np.linalg.norm(agents - y, axis=1)) ** 2)
    return np.prod(1 - detection_probs)

# Generate zeta-distributed random variables
def zeta_distribution(p, size):
    return np.random.zipf(p, size)

# Move an agent towards a target with a specified learning rate
def move_towards(agent, target, learning_rate):
    direction = target - agent
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    return agent + learning_rate * direction


visited_positions = [set() for _ in agents]


# Roam an agent within a specified region with a specified learning rate
def roam(agent, region_radius, learning_rate, visited_positions):
    while True:
        direction = np.random.uniform(-1, 1, 2)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        new_position = agent + learning_rate * direction
        if np.linalg.norm(new_position) <= region_radius and tuple(new_position) not in visited_positions:
            visited_positions.add(tuple(new_position))
            return new_position
    return agent

# Compute the gradients for the agents based on their positions, the targets, and a parameter alpha
def compute_gradients(agents, alpha, targets):
    gradients = np.zeros_like(agents)
    xi_samples = np.random.uniform(0.5, 1.5, num_samples)
    for i, agent in enumerate(agents):
        for target in targets:
            for xi_sample in xi_samples:
                detection_prob = detection_probability(agent, target, xi_sample, alpha)
                error_prob = 1 - detection_prob
                grad = 2 * (error_prob - delta) * detection_prob * (agent - target) * (alpha ** 2) * xi_sample
                gradients[i] += grad
    gradients /= num_samples * len(targets)
    return gradients

def share_positions(agents, epoch):
    if epoch % 100 == 0:  # Only print every 100 epochs to reduce output
        print(f"Epoch {epoch}:")
        for i, agent in enumerate(agents):
            print(f"Agent {i}: {agent}")

# Perform Stochastic Gradient Descent (SGD) to optimize the positions of the agents
def sgd(agents, alpha, targets, epochs, tau, step_size_func, p=None):
    F_values = []
    P_values = []
    pursued_targets = np.array([]).reshape(0, 2)  # Initialize an empty array to store pursued targets
    position_history = [agents.copy()] * (tau + 1)  # Initialize history of positions with tau+1 entries

    # Generate zeta-distributed waiting times
    waiting_times = zeta_distribution(p, epochs) if p else None
    for epoch in range(epochs):
    
        delayed_agents = position_history[0]  # Use the oldest positions in the history (tau steps ago)

        for i, agent in enumerate(agents):
            target = sensor_detection(delayed_agents[i], targets, detection_threshold, pursued_targets)
            if target is not None:
                agents[i] = move_towards(agent, target, step_size_func(epoch))
                pursued_targets = np.vstack([pursued_targets, target])  # Add target to pursued targets list
            else:
                agents[i] = roam(agent, region_radius, step_size_func(epoch), visited_positions[i])

        # Compute gradients using delayed positions
        gradients = compute_gradients(delayed_agents, alpha, targets)

        # Update agents' positions using SGD with delays
        agents -= step_size_func(epoch) * gradients

        # Update the position history
        position_history.append(agents.copy())
        if len(position_history) > tau + 1:
            position_history.pop(0)

        # Share positions after updating
        share_positions(agents, epoch)

        # Compute objective and penalty values for logging
        obj_val = F(agents, alpha, targets)
        penalty_val = P(agents, alpha, targets)
        F_values.append(obj_val)
        P_values.append(penalty_val)
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Objective Value {obj_val}, Penalty Value {penalty_val}")

    return agents, F_values, P_values

# Step size functions for SGD
def step_size_rule_1(n):
    return 1 / ((n / 100) + 10)

def step_size_rule_2(n, p):
    return 1 / ((n ** (1 / (p - 1)) / 100) + 10)

def reset_agents_and_plot(agents, Y, region_radius, error_probability, alpha, F_values, P_values):
    # Plot F(x) and P(x) over iterations
    pm.plot_objective_and_penalty(F_values, P_values)
    # Plot final positions of agents and targets
    pm.plot_final_positions(agents, Y, region_radius)
    # Plot heat map of average error probability
    pm.plot_heat_map(agents, Y, region_radius, error_probability, alpha)
    # Reset the agents' positions to their original positions
    return original_agents.copy()

# Running the SGD with different step size rules and plotting results
print("Using Step Size Rule 1")
agents, F_values, P_values = sgd(agents, alpha, Y, epochs, tau, step_size_rule_1)
agents = reset_agents_and_plot(agents, Y, region_radius, error_probability, alpha, F_values, P_values)

print("Using Step Size Rule 2")
for p_value in {2, 3, 4, 5 }:
    print(f"Current p_value: {p_value}")
    agents, F_values, P_values = sgd(agents, alpha, Y, epochs, tau, lambda epoch: step_size_rule_2(epoch, p_value), p=p_value)
    agents = reset_agents_and_plot(agents, Y, region_radius, error_probability, alpha, F_values, P_values)
