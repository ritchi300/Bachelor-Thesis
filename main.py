import numpy as np
import matplotlib.pyplot as plt
import plot_module as pm
from scipy.integrate import dblquad

# Define constants
D = 14  # Number of agents
region_radius = 4  # Region radius
Y_num = 8  # Number of target points
alpha = 5  # Detection radius
epochs = 10000  # Number of iterations
delta = 0.001
num_samples = 10
detection_threshold = 0.1  # Threshold distance for detection
tau = 5  # Information delay in steps
num_trials = 4  # Number of trials to run

# Randomly initialize agent positions
def initialize_positions(radius, num_points):
    points = np.random.uniform(-radius, radius, (num_points, 2))
    points = points[np.sqrt(np.sum(points**2, axis=1)) <= radius]
    points = np.array([point if np.linalg.norm(point) <= radius else point / np.linalg.norm(point) for point in points])
    return points

agents = initialize_positions(region_radius, D)
original_agents = agents.copy()
Y = initialize_positions(region_radius, Y_num)

# Function to calculate the detection probability
def detection_probability(agent, target, xi_sample, alpha):
    return np.exp(-xi_sample * (alpha * np.linalg.norm(agent - target)) ** 2)

# Function for sensor detection
def sensor_detection(agent, targets, detection_threshold, pursued_targets):
    for target in targets:
        if np.linalg.norm(agent - target) < detection_threshold and tuple(target) not in pursued_targets:
            return target
    return None

# Function to calculate the error probability
def error_probability(agents, target, xi_sample, alpha):
    detection_probs = np.exp(-xi_sample * (alpha * np.linalg.norm(agents - target, axis=1)) ** 2)
    return np.prod(1 - detection_probs)

# Integration over the unit disk
def integrand(theta, r, agents, xi_samples, alpha):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    point = np.array([x, y])
    return np.mean([error_probability(agents, point, xi_sample, alpha) for xi_sample in xi_samples])

def F(agents, alpha):
    xi_samples = np.random.uniform(0.5, 1.5, num_samples)
    result, _ = dblquad(integrand, 0, 1, lambda r: 0, lambda r: 2 * np.pi, args=(agents, xi_samples, alpha))
    return result / np.pi  

# Function to calculate the penalty
def P(agents, alpha, targets):
    penalty = 0
    xi_samples = np.random.uniform(0.5, 1.5, num_samples)
    for target in targets:
        penalty += np.mean([max(0, error_probability(agents, target, xi_sample, alpha) - delta) ** 2 for xi_sample in xi_samples])
    return penalty / len(targets)

# Move an agent towards a target
def move_towards(agent, target, learning_rate):
    direction = target - agent
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    return agent + learning_rate * direction
visited_positions = [set() for _ in agents]

# Roam an agent randomly within the region
def roam(agent, region_radius, learning_rate, visited_positions):
    while True:
        direction = np.random.uniform(-1, 1, 2)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction /= norm
        new_position = agent + learning_rate * direction
        if np.linalg.norm(new_position) <= region_radius and tuple(new_position) not in visited_positions:
            visited_positions.add(tuple(new_position))
            return new_position
    return agent

# Compute the gradients for agents
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
    return gradients / (num_samples * len(targets))



# Generate zeta-distributed random variables
def zeta_distribution(p, size):
    return np.random.zipf(p, size)


def share_positions(agents, waiting_times, wait_time, epoch):
    for i, agent in enumerate(agents):
        if wait_time[i] <= 0:  # If the waiting time for the agent has expired
            if epoch % 100 == 0: # Share position with other agents
                print(f"Agent {i} shares position at epoch {epoch}: {agent}")
            
            # Reset wait_time for the agent based on zeta distribution
            wait_time[i] = waiting_times[i, epoch] if waiting_times is not None else 0
        else:
            
            wait_time[i] -= 1

#  Stochastic Gradient Descent to optimize agent positions
def sgd(agents, alpha, targets, epochs, tau, step_size_func, p=None):
    F_values = []
    P_values = []
    pursued_targets = set()
    position_history = [agents.copy()] * (tau + 1)

    wait_time = [0] * len(agents)
    waiting_times = zeta_distribution(p, epochs * len(agents)).reshape((len(agents), epochs)) if p else None

    for epoch in range(epochs):
        step_size = step_size_func(epoch)
        delayed_agents = position_history[0]

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Step Size: {step_size}")

        for i, agent in enumerate(agents):
            target = sensor_detection(delayed_agents[i], targets, detection_threshold, pursued_targets)
            if target is not None:
                agents[i] = move_towards(agent, target, step_size)
                pursued_targets.add(tuple(target))
                if np.linalg.norm(agent - target) < detection_threshold:
                    print(f"Agent {i} captured target {target} at epoch {epoch}")
                    pursued_targets.remove(tuple(target))
            else:
                agents[i] = roam(agent, region_radius, step_size, visited_positions[i])

            share_positions(agents, waiting_times, wait_time, epoch)

        gradients = compute_gradients(delayed_agents, alpha, targets)
        agents -= step_size * gradients

        position_history.append(agents.copy())
        if len(position_history) > tau + 1:
            position_history.pop(0)

        obj_val = F(agents, alpha)
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

# Plotting and reset functions
def reset_agents_and_plot(agents, Y, region_radius, error_probability, alpha, F_values, P_values):
    pm.plot_objective_and_penalty(F_values, P_values)
    plt.figure()
    plt.scatter(agents[:, 0], agents[:, 1], c='blue', label='Agents')
    plt.scatter(Y[:, 0], Y[:, 1], c='red', marker='x', label='Targets')
    plt.xlim([-region_radius, region_radius])
    plt.ylim([-region_radius, region_radius])
    plt.title('Final Positions of Agents and Targets')
    plt.legend()
    plt.grid(True)
    plt.show()
    pm.plot_heat_map(agents, Y, region_radius, error_probability, alpha)
    return original_agents.copy()

def run_multiple_trials(num_trials, agents, alpha, Y, epochs, tau, step_size_func, p=None):
    all_F_values = []
    all_P_values = []

    for _ in range(num_trials):
        agents = original_agents.copy()
        pm.plot_initial_positions(agents, Y, region_radius)
        agents, F_values, P_values = sgd(agents, alpha, Y, epochs, tau, step_size_func, p)
        all_F_values.append(F_values)
        all_P_values.append(P_values)
        pm.plot_heat_map(agents, Y, region_radius, error_probability, alpha)

    # Convert to numpy arrays 
    all_F_values = np.array(all_F_values)
    all_P_values = np.array(all_P_values)

    # Calculate mean and standard deviation 
    mean_F_values = np.mean(all_F_values, axis=0)  
    std_F_values = np.std(all_F_values, axis=0)

    mean_P_values = np.mean(all_P_values, axis=0)
    std_P_values = np.std(all_P_values, axis=0)

    # Plot the results
    plot_mean_trajectory(mean_F_values, std_F_values, 'F(x)')
    plot_mean_trajectory(mean_P_values, std_P_values, 'P(x)')

    

def plot_mean_trajectory(mean_values, std_values, label):

    mean_values = np.asarray(mean_values, dtype=float)
    std_values = np.asarray(std_values, dtype=float)

    plt.figure()
    plt.plot(mean_values, label=f'Mean {label}')
    plt.fill_between(range(len(mean_values)), mean_values - std_values, mean_values + std_values, alpha=0.3)
    plt.xlabel('Iterations')
    plt.ylabel(label)
    plt.legend()
    plt.show()


 
print("Running multiple trials with Step Size Rule 1")
run_multiple_trials(num_trials, agents, alpha, Y, epochs, tau, step_size_rule_1)

print("Running multiple trials with Step Size Rule 2")
for p_value in {2, 3, 4, 5}:
    print(f"Current p_value: {p_value}")
    run_multiple_trials(num_trials, agents, alpha, Y, epochs, tau, lambda epoch: step_size_rule_2(epoch, p_value), p=p_value)
