from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import plot_module as pm
from scipy.integrate import dblquad
from scipy.stats import zipf

# Define constants for the simulation
D = 15  # Number of agents
region_radius = 1  # Radius of the circular region where agents operate
Y_num = 8  # Number of target points
epochs = 100  # Number of iterations for the simulation
delta = 0.0001  # Detection error threshold for penalties
num_samples = 5  # Number of stochastic samples (for randomness in simulations)
tau = 1  # Communication delay (in steps)
num_trials = 10  # Number of simulation trials to run
kappa = 2  # Penalty scaling factor
xi_samples = np.random.uniform(30, 31, num_samples)  # Random xi samples for detection probabilities
z = 1.5  # Zipf distribution parameter (for delays)
max_broadcast_time = 50  # Maximum allowed time for broadcasting positions

# Step size rules for gradient descent
def step_size_rule_1(n):
    return 1 / ((n / 100) + 10)

def step_size_rule_2(n, p):
    # Calculate q based on p
    q = min(0.5 * (1 / p - 1), 1)
    if n <= 0:
        n = 1  # Ensure n is positive to avoid division by zero or negative powers
    return 1 / ((n ** q / 100) + 10)

# Zipf distribution to model delays in broadcasting positions
def zipf_delay(z, size=1):
    zeta = np.random.zipf(z, size=size)
    return zeta


class PositionHistory:
    def __init__(self, tau):
        self.tau = tau  # Communication delay in steps
        self.history = []  # List to store (epoch, position) 

    def store_position(self, position, epoch):
        
        self.history.append((epoch, position.copy()))  # Save a copy of the position
        if len(self.history) > self.tau + 1:  # Limit history size to tau + 1 entries
            self.history.pop(0)  # Remove the oldest entry

    def get_delayed_position(self, current_epoch):
        
        if not self.history:
            return None  # Return None if history is empty
        delayed_epoch = current_epoch - self.tau  # Calculate delayed epoch
        # Find the closest position for the delayed epoch
        for epoch, position in reversed(self.history):
            if epoch <= delayed_epoch:
                return position
        return self.history[-1][1]  # Return the most recent position as fallback


class Agent:
    def __init__(self, initial_position, tau=tau):
        self.position = np.array(initial_position, dtype=float)  # Current position of the agent
        self.tau = tau  # Communication delay
        self.history_manager = PositionHistory(tau)  # Manage position history with communication delay
        self.trajectory = [self.position.copy()]  # List to store the trajectory of the agent
        self.broadcasted_positions = []  # List to store broadcasted positions
        self.history_manager.store_position(self.position, 0)  # Store the initial position
        self.z = z  # Zipf parameter for broadcast delay
        self.max_broadcast_time = max_broadcast_time  # Maximum allowed broadcast time
        self.next_broadcast_epoch = self.get_next_broadcast_time()  # Schedule next broadcast

    def get_next_broadcast_time(self):
        raw_delay = zipf_delay(self.z, size=1)[0]  # Get Zipf-distributed delay
        delay = min(raw_delay, self.max_broadcast_time)  # Limit the delay to a maximum
        return delay

    def get_delayed_position(self, current_epoch):
        if not self.history_manager.history:
            return None  # Return None if there is no history
        zipf_delay_value = self.get_next_broadcast_time()  # Get Zipf delay for broadcasting
        delayed_epoch = current_epoch - zipf_delay_value  # Apply the delay
        # Search for the position at or before the delayed epoch
        for epoch, position in reversed(self.history_manager.history):
            if epoch <= delayed_epoch:
                return position
        return self.history_manager.history[-1][1]  # Fallback to the most recent position

    def detection_probability(self, point, xi_sample):
        distance = np.linalg.norm(self.position - point)
        detection_prob = np.exp(-xi_sample * distance ** 2)  # Exponential decay with distance
        detection_prob = np.clip(detection_prob, 0, 1)  # Ensure probability is between 0 and 1
        return detection_prob

    def update_position(self, gradient, step_size):
        new_position = self.position - step_size * gradient  # Gradient descent step
        # Ensure the new position is within the region radius
        if np.linalg.norm(new_position) > region_radius:
            new_position = new_position / np.linalg.norm(new_position) * region_radius
        self.position = new_position
        self.trajectory.append(self.position.copy())  # Store the updated position in the trajectory

    def share_position(self, epoch):
        if epoch >= self.next_broadcast_epoch:
            self.history_manager.store_position(self.position, epoch)  # Store broadcasted position
            self.broadcasted_positions.append(self.position.copy())  # Save the broadcasted position
            self.next_broadcast_epoch = epoch + self.get_next_broadcast_time()  # Schedule the next broadcast
            return True  # Return True if a broadcast occurred
        return False  # Return False if no broadcast occurred

# SGD class for managing the simulation of agents using stochastic gradient descent
class SGD:
    def __init__(self, agents, targets, epochs, tau, step_size_func, delta, kappa):
        self.agents = agents  # List of agents
        self.targets = targets  # List of target points
        self.epochs = epochs  # Number of iterations
        self.tau = tau  # Communication delay
        self.step_size_func = step_size_func  # Step size function for gradient descent
        self.delta = delta  # Threshold for penalty
        self.kappa = kappa  # Penalty scaling factor
        self.xi_samples = xi_samples  # Stochastic samples

    def compute_gradients(self, current_epoch):
        gradients = np.zeros((len(self.agents), 2))  # Gradients for each agent's position (x, y)

        # Loop over all target points and stochastic samples
        for target in self.targets:
            for xi_sample in self.xi_samples:
                detection_errors = np.ones(len(self.agents))  # Detection error for each agent
                # Calculate detection errors for all agents
                for i, agent in enumerate(self.agents):
                    delayed_position = agent.get_delayed_position(current_epoch)  # Get delayed position
                    detection_prob = agent.detection_probability(target, xi_sample)  # Calculate detection probability
                    detection_errors[i] *= (1 - detection_prob)  # Update detection error
                
                # Calculate the gradient for each agent
                for i, agent in enumerate(self.agents):
                    delayed_position = agent.get_delayed_position(current_epoch)  # Get delayed position
                    detection_prob_self = agent.detection_probability(target, xi_sample)
                    distance_vector = delayed_position - target  # Vector from agent to target
                    detection_grad = 2 * xi_sample * detection_prob_self * distance_vector  # Gradient based on detection

                    # Error gradient
                    error_grad = -detection_errors[i] * detection_grad / (1 - detection_prob_self)
                    gradients[i] += error_grad
                    
                    # Penalty gradient if detection error exceeds the threshold delta
                    if detection_errors[i] > self.delta:
                        penalty_grad = 2 * (detection_errors[i] - self.delta) * detection_grad
                        gradients[i] += penalty_grad

        # Average the gradients over all targets and samples
        return gradients / (len(self.targets) * len(self.xi_samples))

    def run(self):
        position_history = np.zeros((self.epochs, len(self.agents), 2))  # Store position history for all agents
        gradient_norms = []  # Store gradient norms for each epoch

        for epoch in range(self.epochs):
            step_size = self.step_size_func(epoch)  # Get step size for current epoch

            # Compute gradients for all agents using delayed positions
            gradients = self.compute_gradients(epoch)

            # Compute and store the gradient norm (for convergence tracking)
            grad_norm = np.linalg.norm(gradients, axis=1).mean()
            gradient_norms.append(grad_norm)

            # Print progress every 10 epochs
            if epoch % 10 == 0:
                f_value = self.f()  # Compute objective function value
                p_value = self.calculate_P()  # Compute penalty term
                f_F_value = self.calculate_F()  # Compute detection error term
                print(f"Epoch {epoch}: f(x) = {f_value}, P(x) = {p_value}, F(x) = {f_F_value}")
                print(f"Avg Gradient Norm at Epoch {epoch}: {grad_norm:.6f}")

            # Update positions and broadcast positions for each agent
            for i, agent in enumerate(self.agents):
                agent.update_position(gradients[i], step_size)  # Update position based on gradient
                agent.share_position(epoch)  # Broadcast position if needed
                
            # Store current positions in position history
            position_history[epoch] = np.array([agent.position for agent in self.agents])

        return position_history, gradient_norms

    def calculate_F(self):
        global xi_samples

        def integrand(theta, r):
            # Convert polar coordinates (theta, r) to Cartesian (x, y)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            point = np.array([x, y])

            # Calculate average detection error probability
            detection_error_prob = np.mean(
                [np.sum([agent.detection_probability(point, xi_sample) for agent in self.agents])
                 for xi_sample in xi_samples]
            )
            return detection_error_prob

        # Integrate the error term over the unit disk (circular region)
        result, _ = dblquad(integrand, 0, 2 * np.pi, lambda r: 0, lambda r: region_radius)
        return result

    def calculate_P(self):
        penalties = []
        for agent in self.agents:
            # Calculate the distance of the agent to each target
            distances = np.linalg.norm(agent.position - self.targets, axis=1)
            detection_probs = np.exp(-xi_samples[:, np.newaxis] * distances ** 2)  # Detection probabilities
            error_probs = 1 - detection_probs  # Detection errors
            error_means = np.mean(error_probs, axis=0)  # Average detection error
            penalties.append(np.maximum(0, error_means - self.delta) ** 2)  # Apply penalty for errors > delta
        return np.mean(penalties) / len(self.targets)  # Average penalty

    def calculate_F_P_values(self, position_history):
        F_values = []
        P_values = []

        for epoch_positions in position_history:
            for i, pos in enumerate(epoch_positions):
                self.agents[i].position = pos  # Update agent positions
            F_values.append(self.calculate_F())  # Calculate F(x)
            P_values.append(self.calculate_P())  # Calculate P(x)

        return F_values, P_values

    def f(self):
        F_x = self.calculate_F()  # Compute error term F(x)
        P_x = self.calculate_P()  # Compute penalty term P(x)
        f_x = F_x + self.kappa * P_x  # Combine terms into the full objective
        return f_x

# Function to initialize random positions within the region
def initialize_positions(radius, num_points):
    points = np.random.uniform(-radius, radius, (num_points, 2))
    points = points[np.sqrt(np.sum(points**2, axis=1)) <= radius]  # Ensure points are inside the circle
    return points

# Run multiple simulation trials in parallel
def run_multiple_trials(num_trials, Y, initial_positions, epochs, tau, step_size_func, delta, kappa):
    results = Parallel(n_jobs=-1)(
        delayed(run_single_trial)(Y,initial_positions,epochs,tau,step_size_func,delta,kappa,i) for i in range(num_trials)
    )

    all_F_values, all_P_values, all_gradient_norms = zip(*results)

    np.save('F_values.npy', all_F_values)  # Save F(x) values to a file
    np.save('P_values.npy', all_P_values)  # Save P(x) values to a file

    # Plot mean trajectories and gradients
    pm.plot_mean_trajectory(np.mean(all_F_values, axis=0), np.std(all_F_values, axis=0), 'F(x)')
    pm.plot_mean_trajectory(np.mean(all_P_values, axis=0), np.std(all_P_values, axis=0), 'P(x)')
    pm.plot_gradient_norms(np.mean(all_gradient_norms, axis=0))



# Function to run a single trial
def run_single_trial(Y, initial_positions, epochs, tau, step_size_func, delta, kappa, trial_idx):
    agents = [Agent(position, tau=tau) for position in initial_positions]  
    
    pm.plot_initial_positions(np.array([agent.position for agent in agents]), Y, region_radius)  # Plot initial positions

    # Run SGD optimization
    sgd_instance = SGD(agents, Y, epochs, tau, step_size_func, delta, kappa)
    position_history, gradient_norms = sgd_instance.run()

    # Plot results
    pm.plot_trajectories_with_delays(position_history, agents, Y, region_radius)
    pm.plot_detection_error_heatmap(agents, Y, region_radius, xi_samples)  
    F_values, P_values = sgd_instance.calculate_F_P_values(position_history)

    return F_values, P_values, gradient_norms


 # Main execution
Y = initialize_positions(region_radius, Y_num)  # Initialize target positions once
initial_positions = initialize_positions(region_radius, D)  # Initialize agent positions once

# Run multiple trials with step_size_rule_1
run_multiple_trials(num_trials, Y, initial_positions, epochs, tau, step_size_rule_1, delta, kappa)

# Run with different p-values for step_size_rule_2
for p_value in {2, 3, 4, 5}:
    run_multiple_trials(num_trials,Y,initial_positions,epochs,tau,lambda epoch: step_size_rule_2(epoch, p_value),delta,kappa,)
