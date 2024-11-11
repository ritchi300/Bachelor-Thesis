from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
import plot_module as pm
from scipy.integrate import dblquad
from scipy.stats import zipf as zipf
import os 
# Enable energy constraints
ENABLE_ENERGY_CONSTRAINTS = True  # Set to True to enable energy constraints
MAX_ENERGY = 100  # Maximum energy agents can store
MOVEMENT_ENERGY_COST = 5   # Energy cost for moving
BROADCAST_ENERGY_COST = 3  # Energy cost for broadcasting
ENERGY_HARVEST_RATE = 0  # Energy harvested per epoch (stochastic)

# Define constants
D = 10  # Number of agents
region_radius = 1  # Region radius
Y_num = 8  # Number of target points
epochs = 10  # Number of iterations
delta = 0.0001  # Global delta variable
num_samples = 5  # Number of stochastic samples
tau = 1  # Communication delay in steps
num_trials = 2  # Number of trials to run
xi_samples = np.random.uniform(30, 31, num_samples)  # Random xi samples
z = 1.5  # Zipf parameter
max_broadcast_time = 100  # Maximum broadcast time

# Step size functions
def step_size_rule_1(n):
    return 1 / ((n / 100) + 10)

def step_size_rule_2(n, p):
    # calculate q
    q = min(0.5 * (1 / p - 1), 1)

    # Avoid division by zero or negative powers
    if n <= 0:
        n = 1  

    # Return step size
    return 1 / ((n ** q / 100) + 10)

# Zipf distribution for delays
def zipf_delay(z, size=1, random_state=None):
    if random_state is None:
        random_state = np.random
    zeta = random_state.zipf(z, size=size)
    return zeta

# Markov Energy Harvester
class MarkovEnergyHarvester:
    def __init__(self, states, transition_matrix_stationary, random_state=None):
        self.states = states
        self.transition_matrix_stationary = transition_matrix_stationary
        self.random_state = random_state if random_state is not None else np.random
        self.current_state_index = self.random_state.choice(len(states))

    def harvest_energy_stationary(self):
        transition_probs = self.transition_matrix_stationary[self.current_state_index]
        self.current_state_index = self.random_state.choice(len(self.states), p=transition_probs)
        return self.states[self.current_state_index]

# Non-Stationary Markov Energy Harvester
class NonStationaryMarkovEnergyHarvester:
    def __init__(self, states, transition_matrices_non_stationary, random_state=None):
        self.states = states
        self.transition_matrices_non_stationary = transition_matrices_non_stationary
        self.random_state = random_state if random_state is not None else np.random
        self.current_state_index = self.random_state.choice(len(states))

    def harvest_energy_non_stationary(self, time_step):
        transition_matrix = self.transition_matrices_non_stationary[time_step % len(self.transition_matrices_non_stationary)]
        transition_probs = transition_matrix[self.current_state_index]
        self.current_state_index = self.random_state.choice(len(self.states), p=transition_probs)
        return self.states[self.current_state_index]

class PositionHistory:
    def __init__(self, tau):
        self.tau = tau  # Communication delay
        self.history = []  # Stores a list of (epoch, position) tuples

    def store_position(self, position, epoch):
        """
        Store the current position of the agent at a given epoch.
        """
        self.history.append((epoch, position.copy()))  # Store epoch and position as a tuple
        if len(self.history) > self.tau + 1:  # Keep history limited to tau + 1 entries
            self.history.pop(0)  # Remove the oldest entry to maintain size

    def get_delayed_position(self, current_epoch):
        """
        Return the position corresponding to a delayed epoch based on the communication delay tau.
        """
        if not self.history:
            return None
        delayed_epoch = current_epoch - self.tau  # Calculate the delayed epoch
        for epoch, position in reversed(self.history):
            if epoch <= delayed_epoch:  # Find the closest position for the delayed epoch
                return position
        return self.history[-1][1]  # Return the most recent position if none is found

# Agent Class
class Agent:
    def __init__(self, initial_position, tau, energy_harvester, random_state=None):
        self.position = np.array(initial_position, dtype=float)
        self.tau = tau
        self.history_manager = PositionHistory(tau)
        self.trajectory = [self.position.copy()]
        self.broadcasted_positions = []
        self.history_manager.store_position(self.position, 0)
        self.z = z
        self.max_broadcast_time = max_broadcast_time
        self.energy = MAX_ENERGY
        self.energy_harvester = energy_harvester  # Assign a Markov energy harvester instance
        self.random_state = random_state if random_state is not None else np.random  # Assign random_state here

        self.next_broadcast_epoch = self.get_next_broadcast_time()  # Now this is called after self.random_state is set

    def get_next_broadcast_time(self):
        raw_delay = zipf_delay(self.z, size=1, random_state=self.random_state)[0]
        delay = min(raw_delay, self.max_broadcast_time)
        return delay

    def harvest_energy(self, time_step):
        # Check if the harvester is stationary or non-stationary and harvest accordingly
        if isinstance(self.energy_harvester, MarkovEnergyHarvester):
            harvested = self.energy_harvester.harvest_energy_stationary()
        else:
            harvested = self.energy_harvester.harvest_energy_non_stationary(time_step)
        
        self.energy = min(MAX_ENERGY, self.energy + harvested)
        # Uncomment the following line if you want to print energy harvesting details
        # print(f"Epoch {time_step}: Agent harvested {harvested:.2f} energy. Current energy: {self.energy:.2f}")

    def can_broadcast(self):
        return self.energy >= BROADCAST_ENERGY_COST

    def can_move(self):
        return self.energy >= MOVEMENT_ENERGY_COST

    def get_delayed_position(self, current_epoch):
        zipf_delay_value = self.get_next_broadcast_time()
        delayed_epoch = current_epoch - zipf_delay_value
        for epoch, position in reversed(self.history_manager.history):
            if epoch <= delayed_epoch:
                return position
        return self.history_manager.history[-1][1]

    def detection_probability(self, point, xi_sample):
        distance = np.linalg.norm(self.position - point)
        detection_prob = np.exp(-xi_sample * distance ** 2)
        return np.clip(detection_prob, 0, 1)

    def update_position(self, gradient, step_size):
        if ENABLE_ENERGY_CONSTRAINTS and not self.can_move():
            # Uncomment the following line if you want to print energy constraints details
            # print(f"Agent cannot move due to insufficient energy. Current energy: {self.energy:.2f}")
            return
        
        new_position = self.position - step_size * gradient
        if np.linalg.norm(new_position) > region_radius:
            new_position = new_position / np.linalg.norm(new_position) * region_radius
        self.position = new_position
        self.trajectory.append(self.position.copy())

        if ENABLE_ENERGY_CONSTRAINTS:
            self.energy -= MOVEMENT_ENERGY_COST
            # Uncomment the following line if you want to print energy usage details
            # print(f"Agent used {MOVEMENT_ENERGY_COST:.2f} energy for movement. Remaining energy: {self.energy:.2f}")

    def share_position(self, epoch):
        if epoch >= self.next_broadcast_epoch and (not ENABLE_ENERGY_CONSTRAINTS or self.can_broadcast()):
            self.history_manager.store_position(self.position, epoch)
            self.broadcasted_positions.append(self.position.copy())
            self.next_broadcast_epoch = epoch + self.get_next_broadcast_time()

            if ENABLE_ENERGY_CONSTRAINTS:
                self.energy -= BROADCAST_ENERGY_COST
                # Uncomment the following line if you want to print energy usage details
                # print(f"Agent used {BROADCAST_ENERGY_COST:.2f} energy for broadcasting. Remaining energy: {self.energy:.2f}")

            return True
        return False

class SGD:
    def __init__(self, agents, targets, epochs, tau, step_size_func, delta, kappa, xi_samples):
        self.agents = agents
        self.targets = targets
        self.epochs = epochs
        self.tau = tau
        self.step_size_func = step_size_func
        self.delta = delta
        self.kappa = kappa
        self.xi_samples = xi_samples

    def compute_gradients(self, current_epoch):
        """
        Compute the combined gradient for all agents based on F(x) and P(x) using delayed positions.
        """
        gradients = np.zeros((len(self.agents), 2))  # Gradient for each agent's position (x, y)

        for target in self.targets:
            for xi_sample in self.xi_samples:
                detection_errors = np.ones(len(self.agents))  # Initialize detection error for each agent
                
                # Calculate detection errors across all agents
                for i, agent in enumerate(self.agents):
                    delayed_position = agent.get_delayed_position(current_epoch)  # Get delayed position
                    detection_prob = agent.detection_probability(target, xi_sample)
                    detection_errors[i] *= (1 - detection_prob)
                
                for i, agent in enumerate(self.agents):
                    delayed_position = agent.get_delayed_position(current_epoch)  # Get delayed position
                    detection_prob_self = agent.detection_probability(target, xi_sample)
                    distance_vector = delayed_position - target
                    detection_grad = 2 * xi_sample * detection_prob_self * distance_vector

                    # Accumulate error gradient
                    error_grad = -detection_errors[i] * detection_grad / (1 - detection_prob_self)
                    gradients[i] += error_grad
                    
                    # Calculate penalty gradient
                    if detection_errors[i] > self.delta:
                        penalty_grad = 2 * (detection_errors[i] - self.delta) * detection_grad
                        gradients[i] += penalty_grad

        return gradients / (len(self.targets) * len(self.xi_samples))

    def run(self):
        position_history = np.zeros((self.epochs, len(self.agents), 2))
        gradient_norms = []

        for epoch in range(self.epochs):
            step_size = self.step_size_func(epoch)

            # Compute gradients for all agents using delayed positions
            gradients = self.compute_gradients(epoch)

            # Compute gradient norms for convergence tracking
            grad_norm = np.linalg.norm(gradients, axis=1).mean()
            gradient_norms.append(grad_norm)

            # Print progress every 10 epochs
            if epoch % 10 == 0:
                f_value = self.f()
                p_value = self.calculate_P()
                f_F_value = self.calculate_F()
                
                print(f"Epoch {epoch}: f(x) = {f_value}, P(x) = {p_value}, F(x) = {f_F_value}")
                print(f"Avg Gradient Norm at Epoch {epoch}: {grad_norm:.6f}")
                
            # Update each agent's position
            for i, agent in enumerate(self.agents):
                agent.update_position(gradients[i], step_size)
                agent.share_position(epoch)
                agent.harvest_energy(epoch)  # Harvest energy
                # Uncomment the following lines if you want to print broadcast details
                # if agent.share_position(epoch):
                #     print(f"Agent {i + 1} broadcasted position at epoch {epoch}")
                
            # Store the current positions for history
            position_history[epoch] = np.array([agent.position for agent in self.agents])

        return position_history, gradient_norms

    def calculate_F(self):
        """Calculate the error term F(x), which contains the average detection error probability for all target points."""
        def integrand(theta, r):
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            point = np.array([x, y])

            detection_error_prob = np.mean(
                [np.sum([agent.detection_probability(point, xi_sample) for agent in self.agents])
                 for xi_sample in self.xi_samples]
            )
            return detection_error_prob

        result, _ = dblquad(integrand, 0, 2 * np.pi, lambda r: 0, lambda r: region_radius)
        
        return result

    def calculate_P(self):
        """Calculate the penalty term P(x), which computes penalties for detection errors exceeding the threshold delta."""
        penalties = []
        for agent in self.agents:
            distances = np.linalg.norm(agent.position - self.targets, axis=1)
            detection_probs = np.exp(-self.xi_samples[:, np.newaxis] * distances ** 2)
            error_probs = 1 - detection_probs
            error_means = np.mean(error_probs, axis=0)
            penalties.append(np.maximum(0, error_means - self.delta) ** 2)
        return np.mean(penalties) / len(self.targets)

    def calculate_F_P_values(self, position_history):
        F_values = []
        P_values = []

        for epoch_positions in position_history:
            for i, pos in enumerate(epoch_positions):
                self.agents[i].position = pos
            F_values.append(self.calculate_F())
            P_values.append(self.calculate_P())

        return F_values, P_values

    def f(self):
        """Calculate the full objective function f(x) = F(x) + κ * P(x)."""
        F_x = self.calculate_F()  # Calculate the error term F(x)
        P_x = self.calculate_P()  # Calculate the penalty term P(x)
        f_x = F_x + self.kappa * P_x  # Combine both terms
        return f_x

def initialize_positions(radius, num_points, random_state=None):
    if random_state is None:
        random_state = np.random
    points = random_state.uniform(-radius, radius, (num_points, 2))
    points = points[np.sqrt(np.sum(points**2, axis=1)) <= radius]
    while len(points) < num_points:
        extra_points = random_state.uniform(-radius, radius, (num_points - len(points), 2))
        extra_points = extra_points[np.sqrt(np.sum(extra_points**2, axis=1)) <= radius]
        points = np.concatenate((points, extra_points), axis=0)
    return points[:num_points]

def run_multiple_trials(num_trials, Y, initial_positions, epochs, tau, step_size_func, delta, kappa, base_seed=42, step_size_name=''):
    results = Parallel(n_jobs=-1)(
        delayed(run_single_trial)(Y, initial_positions, epochs, tau, step_size_func, delta, kappa, i, base_seed, step_size_name)
        for i in range(num_trials)
    )

    all_F_values, all_P_values, all_gradient_norms = zip(*results)

    # Erstellen Sie ein Verzeichnis, um die Ergebnisse zu speichern
    os.makedirs("results", exist_ok=True)

    # Speichern der Ergebnisse als numpy-Dateien mit eindeutigen Dateinamen für kappa, delta und step_size
    np.save(f'results/F_values_kappa{kappa}_delta{delta}_{step_size_name}.npy', all_F_values)
    np.save(f'results/P_values_kappa{kappa}_delta{delta}_{step_size_name}.npy', all_P_values)

    # Plots als Dateien speichern
    fig1 = pm.plot_mean_trajectory(
        np.mean(all_F_values, axis=0),
        np.std(all_F_values, axis=0),
        'F(x)',
        title=f'F(x) for kappa={kappa}, delta={delta}, step_size={step_size_name}'
    )
    fig1.savefig(f'results/Fx_kappa{kappa}_delta{delta}_{step_size_name}.png')
    plt.close(fig1)  # Schließen der Grafik

    fig2 = pm.plot_mean_trajectory(
        np.mean(all_P_values, axis=0),
        np.std(all_P_values, axis=0),
        'P(x)',
        title=f'P(x) for kappa={kappa}, delta={delta}, step_size={step_size_name}'
    )
    fig2.savefig(f'results/Px_kappa{kappa}_delta{delta}_{step_size_name}.png')
    plt.close(fig2)

    fig3 = pm.plot_gradient_norms(
        np.mean(all_gradient_norms, axis=0),
        title=f'Gradient Norms for kappa={kappa}, delta={delta}, step_size={step_size_name}'
    )
    fig3.savefig(f'results/gradient_norms_kappa{kappa}_delta{delta}_{step_size_name}.png')
    plt.close(fig3)

def run_single_trial(Y, initial_positions, epochs, tau, step_size_func, delta, kappa, trial_idx, base_seed, step_size_name):
    # Set random seed for reproducibility
    seed = base_seed + trial_idx
    random_state = np.random.RandomState(seed)
    xi_samples = random_state.uniform(30, 31, num_samples)  # Random xi samples

    states = [0, 5, 10]
    stationary_transition_matrix = np.array([
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7]
    ])
    energy_harvesters = [MarkovEnergyHarvester(states, stationary_transition_matrix, random_state=random_state) for _ in range(D)]
    agents = [Agent(position, tau=tau, energy_harvester=energy_harvester, random_state=random_state)
              for position, energy_harvester in zip(initial_positions, energy_harvesters)]
    
        # Open file to write trial results, including step_size_name, kappa, and delta
    filename = f"trial_{trial_idx}_step_{step_size_name}_kappa_{kappa}_delta_{delta}_results.txt"
    with open(filename, "w") as file:
        file.write(f"Trial {trial_idx} Results (Step Size Rule: {step_size_name}, Kappa: {kappa}, Delta: {delta}):\n")
        
        # Write header information for targets and initial positions of agents
        file.write("Target Positions (Y):\n")
        for i, target in enumerate(Y):
            file.write(f"Target {i}: {target}\n")
        file.write("\nInitial Agent Positions:\n")
        for i, agent in enumerate(agents):
            file.write(f"Agent {i} Initial Position: {agent.position}\n")
        
        # Write main header for epoch data
        file.write(f"\n{'Epoch':<10} {'F(x)':<15} {'P(x)':<15} {'f(x)':<15} {'Avg Gradient Norm':<20}\n")
        
        # Run SGD optimization
        sgd_instance = SGD(agents, Y, epochs, tau, step_size_func, delta, kappa, xi_samples)
        
        # Store energy and gradient details across epochs
        position_history, gradient_norms = sgd_instance.run()
        for epoch in range(epochs):
            f_value = sgd_instance.f()
            p_value = sgd_instance.calculate_P()
            f_F_value = sgd_instance.calculate_F()
            avg_gradient_norm = gradient_norms[epoch]
            
            # Write epoch summary data
            file.write(f"{epoch:<10} {f_F_value:<15.6f} {p_value:<15.6f} {f_value:<15.6f} {avg_gradient_norm:<20.6f}\n")
            
            # Write positions, energy level, and energy usage for each agent
            file.write(f"\nEpoch {epoch} Agent Details:\n")
            for i, agent in enumerate(agents):
                # Calculate used energy
                initial_energy = agent.energy
                agent.harvest_energy(epoch)  # Harvest energy for this epoch
                energy_used = initial_energy - agent.energy + agent.energy_harvester.harvest_energy_stationary()
                
                # Write agent's position, energy level, and energy used
                file.write(f"Agent {i} Position: {agent.position}, Energy Level: {agent.energy:.2f}, Energy Used: {energy_used:.2f}\n")
        
        # Close file after writing all epochs for this trial
    
   
    pm.plot_initial_positions(np.array([agent.position for agent in agents]), Y, region_radius, 
                          filename=f'results/initial_positions_kappa{kappa}_delta{delta}_{step_size_name}.png')

    

    # Run SGD optimization
    sgd_instance = SGD(agents, Y, epochs, tau, step_size_func, delta, kappa, xi_samples)
    position_history, gradient_norms = sgd_instance.run()

   
    pm.plot_trajectories_with_delays(
    position_history, agents, Y, region_radius, 
    filename=f'results/trajectories_with_delays_kappa{kappa}_delta{delta}_{step_size_name}.png')
    pm.plot_detection_error_heatmap(agents, Y, region_radius, xi_samples, 
                                filename=f'results/detection_error_heatmap_kappa{kappa}_delta{delta}_{step_size_name}.png')
    F_values, P_values = sgd_instance.calculate_F_P_values(position_history)

    return F_values, P_values, gradient_norms

def main():
    # Set the base random seed
    base_seed = 42
    np.random.seed(base_seed)

    # Initialize target and agent positions once
    Y = initialize_positions(region_radius, Y_num, random_state=np.random)
    initial_positions = initialize_positions(region_radius, D, random_state=np.random)

    # Define parameter grids
    kappa_values = [1, 2, 4, 6]
    delta_values = [0.0001, 0.001]
    step_size_functions = {
        'step_size_rule_1': step_size_rule_1,
        'step_size_rule_2_p2': lambda n: step_size_rule_2(n, 2),
        'step_size_rule_2_p3': lambda n: step_size_rule_2(n, 3),
        'step_size_rule_2_p4': lambda n: step_size_rule_2(n, 4),
        # Add more if needed
    }

    # Loop over kappa, delta, and step size functions
    for kappa in kappa_values:
        for delta in delta_values:
            for step_size_name, step_size_func in step_size_functions.items():
                print(f"Running simulations with kappa={kappa}, delta={delta}, step_size={step_size_name}")
                # Run multiple trials
                run_multiple_trials(
                    num_trials,
                    Y,
                    initial_positions,
                    epochs,
                    tau,
                    step_size_func,
                    delta,
                    kappa,
                    base_seed=base_seed,
                    step_size_name=step_size_name
                )

if __name__ == "__main__":
    main()
