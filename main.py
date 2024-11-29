import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import Plot as pm  
from scipy.integrate import dblquad
from scipy.stats import zipf
import os  

# Enable energy constraints
ENABLE_ENERGY_CONSTRAINTS = True    
MAX_ENERGY = 100  # Maximum energy agents can store
MOVEMENT_ENERGY_COST = 10   # Energy cost for moving
BROADCAST_ENERGY_COST = 5  # Energy cost for broadcasting
HIGH_ENERGY_THRESHOLD = 0.7 * MAX_ENERGY  # Above 70% of MAX_ENERGY
LOW_ENERGY_THRESHOLD = 0.3 * MAX_ENERGY   # Below 30% of MAX_ENERGY

# Define constants for the simulation
D = 16  # Number of agents
region_radius = 1  # Radius of the  region where agents operate
Y_num = 8  # Number of target points
epochs = 1000  # Number of iterations for the simulation
num_samples = 7  # Number of stochastic samples 
tau = 10  # Communication delay 
num_trials = 2  # Number of simulation trials to run


# Step size rules for gradient descent
def step_size_rule_1(n):
    return 1 / ((n / 100) + 10)

def step_size_rule_2(n, p):
    # Calculate q based on p
    q = min(0.5 * (1 + 1 / p - 1), 1)
    if n <= 0:
        n = 1  # avoid division by zero 
    return 1 / ((n ** q / 100) + 10)

# Zipf distribution to model delays in broadcasting positions
def zipf_delay(z, size=1, random_state=None):
    if random_state is None:
        random_state = np.random
    max_delay = 0.5 * epochs  # Set a maximum delay equal to the total number of epochs
    delays = []
    while len(delays) < size:
        sample = random_state.zipf(z)
        if sample <= max_delay:
            delays.append(sample)
    return np.array(delays)

# Energy Harvester Base Class
class EnergyHarvester:
    def harvest_energy(self, time_step):
        # Subclasses should implement this method
        raise NotImplementedError("Subclasses must implement harvest_energy method")

# Markov Energy Harvester
class MarkovEnergyHarvester(EnergyHarvester):
    def __init__(self, states, transition_matrix_stationary, random_state=None):
        self.states = states
        self.transition_matrix_stationary = transition_matrix_stationary
        self.random_state = random_state if random_state is not None else np.random
        self.current_state_index = self.random_state.choice(len(states))

    def harvest_energy(self, time_step):
        transition_probs = self.transition_matrix_stationary[self.current_state_index]
        self.current_state_index = self.random_state.choice(len(self.states), p=transition_probs)
        return self.states[self.current_state_index]

# Non-Stationary Markov Energy Harvester
class NonStationaryMarkovEnergyHarvester(EnergyHarvester):
    def __init__(self, states, transition_matrices_non_stationary, random_state=None):
        self.states = states
        self.transition_matrices_non_stationary = transition_matrices_non_stationary
        self.random_state = random_state if random_state is not None else np.random
        self.current_state_index = self.random_state.choice(len(states))

    def harvest_energy(self, time_step):
        transition_matrix = self.transition_matrices_non_stationary[time_step % len(self.transition_matrices_non_stationary)]
        transition_probs = transition_matrix[self.current_state_index]
        self.current_state_index = self.random_state.choice(len(self.states), p=transition_probs)
        return self.states[self.current_state_index]
    
class PositionHistory:
    def __init__(self, tau):
        self.tau = tau  # Communication delay in steps
        self.history = []  # List to store epoch, position 

    def store_position(self, position, epoch):
        self.history.append((epoch, position.copy()))  
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
    def __init__(self, initial_position, tau, energy_harvester, random_state=None, p_value=2):
        self.position = np.array(initial_position, dtype=float)
        self.tau = tau
        self.history_manager = PositionHistory(tau)
        self.trajectory = [self.position.copy()]
        self.broadcasted_positions = []
        self.history_manager.store_position(self.position, 0)
        self.z = p_value  # Set the z parameter to p
        self.max_broadcast_time = None  # Remove the maximum broadcast time constraint
        self.next_broadcast_epoch = 0  # Initialize to 0
        self.energy_harvester = energy_harvester
        self.energy = MAX_ENERGY
        self.HIGH_ENERGY_THRESHOLD = HIGH_ENERGY_THRESHOLD
        self.LOW_ENERGY_THRESHOLD = LOW_ENERGY_THRESHOLD
        self.random_state = random_state if random_state is not None else np.random

        # Data recording attributes
        self.energy_levels = [self.energy]
        self.harvested_energy_history = []
        self.energy_consumption_history = []
        self.broadcast_times = []

    def start_new_epoch(self):
        # Start tracking energy consumption for a new epoch
        self.energy_consumption_history.append(0)

    def get_next_broadcast_time(self):
        raw_delay = zipf_delay(self.z, size=1, random_state=self.random_state)[0]
        return raw_delay
    
    def harvest_energy(self, time_step):
        harvested = self.energy_harvester.harvest_energy(time_step)
        self.energy = min(MAX_ENERGY, self.energy + harvested)
        self.harvested_energy_history.append(harvested)
        self.energy_levels.append(self.energy)
        # print(f"Epoch {time_step}: Agent harvested {harvested:.2f} energy. Current energy: {self.energy:.2f}")

    def can_broadcast(self):
        return self.energy >= BROADCAST_ENERGY_COST

    def can_move(self):
        return self.energy >= MOVEMENT_ENERGY_COST

    def get_energy_state(self):
        """
        Determine the energy state of the agent: 'high', 'medium', or 'low'.
        """
        if self.energy >= self.HIGH_ENERGY_THRESHOLD:
            return 'high'
        elif self.energy <= self.LOW_ENERGY_THRESHOLD:
            return 'low'
        else:
            return 'medium'
        
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

    def detection_probability(self, point, xi_sample, position=None):
        if position is None:
            position = self.position
        distance = np.linalg.norm(position - point)
        detection_prob = np.exp(-xi_sample * distance ** 2)
        detection_prob = np.clip(detection_prob, 0, 1)
        return detection_prob

    def update_position(self, gradient, step_size):
        energy_state = self.get_energy_state()

        if ENABLE_ENERGY_CONSTRAINTS and not self.can_move():
            self.energy_consumption_history[-1] += 0  # No energy consumed
            return

        # Movement probability based on energy state
        move_probability = {
            'high': 1.0,   # Always move
            'medium': 0.5, # 50% chance to move
            'low': 0.3     # 30% chance to move
        }

        if self.random_state.rand() > move_probability[energy_state]:
            # Skip movement to conserve energy
            self.energy_consumption_history[-1] += 0  # No energy consumed
            return

        # Adjust step size accordingly
        adjusted_step_size = step_size * move_probability[energy_state]

        new_position = self.position - adjusted_step_size * gradient
        if np.linalg.norm(new_position) > region_radius:
            new_position = new_position / np.linalg.norm(new_position) * region_radius
        self.position = new_position
        self.trajectory.append(self.position.copy())

        if ENABLE_ENERGY_CONSTRAINTS:
            self.energy -= MOVEMENT_ENERGY_COST
            self.energy_consumption_history[-1] += MOVEMENT_ENERGY_COST
            # print(f"Agent used {MOVEMENT_ENERGY_COST:.2f} energy for movement. Remaining energy: {self.energy:.2f}")

    def share_position(self, epoch):
        energy_state = self.get_energy_state()

        if ENABLE_ENERGY_CONSTRAINTS and not self.can_broadcast():
            return False

        # Increase broadcasting delay based on energy state
        if energy_state == 'low':
            additional_delay = self.get_next_broadcast_time() * 2
        elif energy_state == 'medium':
            additional_delay = self.get_next_broadcast_time()
        else:
            additional_delay = 0

        if epoch >= self.next_broadcast_epoch + additional_delay:
            self.history_manager.store_position(self.position, epoch)
            self.broadcasted_positions.append(self.position.copy())
            self.next_broadcast_epoch = epoch + self.get_next_broadcast_time()
            self.broadcast_times.append(epoch)

            if ENABLE_ENERGY_CONSTRAINTS:
                self.energy -= BROADCAST_ENERGY_COST
                self.energy_consumption_history[-1] += BROADCAST_ENERGY_COST
                # print(f"Agent used {BROADCAST_ENERGY_COST:.2f} energy for broadcasting. Remaining energy: {self.energy:.2f}")

            return True
        return False

# SGD class for managing the simulation of agents using stochastic gradient descent
class SGD:
    def __init__(self, agents, targets, epochs, tau, step_size_func, delta, kappa, xi_samples):
        self.agents = agents  # List of agents
        self.targets = targets  # List of target points
        self.epochs = epochs  # Number of iterations
        self.tau = tau  # Communication delay
        self.step_size_func = step_size_func  # Step size function for gradient descent
        self.delta = delta  # Threshold for penalty
        self.kappa = kappa  # Penalty scaling factor
        self.xi_samples = xi_samples  # Stochastic samples

        # Data recording attributes
        self.F_values = []
        self.P_values = []
        self.f_values = []
        self.gradient_norms = []

    def compute_gradients(self, current_epoch):
        gradients = np.zeros((len(self.agents), 2))  # Gradients for each agent's position (x, y)

        # Loop over all target points and stochastic samples
        for target in self.targets:
            for xi_sample in self.xi_samples:
                detection_error = 1.0  # Initialize total detection error

                detection_probs = []   # List to store detection probabilities
                delayed_positions = [] # Delayed positions for each agent

                # Calculate detection probabilities and total detection error
                for agent in self.agents:
                    delayed_position = agent.get_delayed_position(current_epoch)
                    delayed_positions.append(delayed_position)
                    detection_prob = agent.detection_probability(target, xi_sample, position=delayed_position)
                    detection_probs.append(detection_prob)
                    detection_error *= (1 - detection_prob)  # Update detection error

                # Calculate the gradient for each agent
                for i, agent in enumerate(self.agents):
                    detection_prob_self = detection_probs[i]
                    delayed_position = delayed_positions[i]
                    distance_vector = delayed_position - target
                    detection_grad = -2 * xi_sample * detection_prob_self * distance_vector  # Corrected gradient

                    # Check if detection_prob_self is 1 to avoid division by zero
                    if detection_prob_self == 1.0:
                        error_grad = np.zeros_like(detection_grad)
                    else:
                        error_grad = -detection_error * detection_grad / (1 - detection_prob_self)
                    gradients[i] += error_grad

                    # Penalty gradient if detection error exceeds the threshold delta
                    if detection_error > self.delta:
                        penalty_grad = 2 * (detection_error - self.delta) * error_grad
                        gradients[i] += self.kappa * penalty_grad

        # Average the gradients over all targets and samples
        return gradients / (len(self.targets) * len(self.xi_samples))

    def run(self):
        position_history = np.zeros((self.epochs, len(self.agents), 2))  # Store position history for all agents

        for epoch in range(self.epochs):
            # Start a new epoch for agents
            for agent in self.agents:
                agent.start_new_epoch()

            step_size = self.step_size_func(epoch)  # Get step size for current epoch

            # Compute gradients for all agents using delayed positions
            gradients = self.compute_gradients(epoch)

            grad_norm = np.linalg.norm(gradients, axis=1).mean()
            self.gradient_norms.append(grad_norm)

            # Compute and store F(x), P(x), f(x)
            F_x = self.calculate_F()
            P_x = self.calculate_P()
            f_x = F_x + self.kappa * P_x
            self.F_values.append(F_x)
            self.P_values.append(P_x)
            self.f_values.append(f_x)

            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: f(x) = {f_x}, P(x) = {P_x}, F(x) = {F_x}")
                print(f"Avg Gradient Norm at Epoch {epoch}: {grad_norm:.6f}")

            # Update positions and broadcast positions for each agent
            for i, agent in enumerate(self.agents):
                agent.update_position(gradients[i], step_size)  # Update position based on gradient
                agent.share_position(epoch)  # Broadcast position if needed
                agent.harvest_energy(epoch)  # Harvest energy at each epoch
                
            # Store current positions in position history
            position_history[epoch] = np.array([agent.position for agent in self.agents])

        return position_history, self.gradient_norms

    def calculate_F(self, num_points=50000):
        def calculate_F(self):
        agent_positions = np.array([agent.position for agent in self.agents])
        xi_samples = self.xi_samples
        region_radius = 1  # Radius of the circular region
        F_x = 0.0

        for xi_sample in xi_samples:
            # Define the integrand function
            def integrand(x, y):
                # Compute detection error at (x, y) for given ξ
                detection_error = np.prod([
                    1 - np.exp(-xi_sample * ((x - x_i[0])**2 + (y - x_i[1])**2))
                    for x_i in agent_positions
                ])
                return detection_error

            # Define the bounds for x and y
            def x_bounds():
                return -region_radius, region_radius

            def y_lower(x):
                return -np.sqrt(region_radius**2 - x**2)

            def y_upper(x):
                return np.sqrt(region_radius**2 - x**2)

            # Perform the double integral over x and y
            integral_result, error = dblquad(
                integrand,
                x_bounds()[0],  # Lower bound for x
                x_bounds()[1],  # Upper bound for x
                y_lower,        # Lower bound function for y
                y_upper         # Upper bound function for y
            )

            # Normalize the result by the area of the region
            F_x += integral_result / (np.pi * region_radius**2)

        # Average over the ξ samples
        F_x = F_x / len(xi_samples)

        return F_x

    def calculate_P(self):
        penalties = []
        for xi in self.xi_samples:
            penalties_per_target = []
            for target in self.targets:
                # Compute p_e(x, y, xi)
                detection_probs = np.prod([
                    1 - np.exp(-xi * np.linalg.norm(agent.position - target) ** 2)
                    for agent in self.agents
                ])
                penalty = max(0, detection_probs - self.delta) ** 2
                penalties_per_target.append(penalty)
            # Average over targets for this xi_sample
            average_penalty = np.mean(penalties_per_target)
            penalties.append(average_penalty)
        # Now average over xi_samples
        P_x = np.mean(penalties)
        return P_x

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
def initialize_positions(radius, num_points, random_state=None):
    if random_state is None:
        random_state = np.random
    points = random_state.uniform(-radius, radius, (num_points, 2))
    points = points[np.sqrt(np.sum(points**2, axis=1)) <= radius]  # Ensure points are inside the circle
    while len(points) < num_points:
        extra_points = random_state.uniform(-radius, radius, (num_points - len(points), 2))
        extra_points = extra_points[np.sqrt(np.sum(extra_points**2, axis=1)) <= radius]
        points = np.concatenate((points, extra_points), axis=0)
    return points[:num_points]

# Function to write data to a text file
def write_data_to_file(data):
    kappa = data['kappa']
    delta = data['delta']
    step_size_name = data['step_size_name']
    trial_idx = data['trial_idx']
    
    filename = f'{step_size_name}_k{kappa}_d{delta}_nr{trial_idx}.txt'
    with open(filename, 'w') as f:
        f.write("Simulation Parameters:\n")
        f.write(f"kappa: {kappa}\n")
        f.write(f"delta: {delta}\n")
        f.write(f"Step Size Function: {step_size_name}\n")
        f.write(f"Trial Index: {trial_idx}\n")
        f.write("================\n\n")

        f.write("Targets Positions:\n")
        for pos in data['targets_positions']:
            f.write(f"{pos}\n")
        f.write("\n")

        num_agents = len(data['agents_positions'])
        for i in range(num_agents):
            f.write(f"Agent {i} Data:\n")
            f.write("----------------\n")
            f.write("Positions:\n")
            for pos in data['agents_positions'][i]:
                f.write(f"{pos.tolist()}\n")
            f.write("\nEnergy Levels:\n")
            for energy in data['agents_energy_levels'][i]:
                f.write(f"{energy}\n")
            f.write("\nHarvested Energy:\n")
            for harvested in data['agents_harvested_energy'][i]:
                f.write(f"{harvested}\n")
            f.write("\nEnergy Consumption:\n")
            for consumed in data['agents_energy_consumption'][i]:
                f.write(f"{consumed}\n")
            f.write("\nBroadcast Times:\n")
            for time in data['agents_broadcast_times'][i]:
                f.write(f"{time}\n")
            f.write("\n")

        f.write("F(x) Values:\n")
        for val in data['F_values']:
            f.write(f"{val}\n")
        f.write("\nP(x) Values:\n")
        for val in data['P_values']:
            f.write(f"{val}\n")
        f.write("\nf(x) Values:\n")
        for val in data['f_values']:
            f.write(f"{val}\n")
        f.write("\nGradient Norms:\n")
        for val in data['gradient_norms']:
            f.write(f"{val}\n") 

# Run multiple simulation trials in parallel
def run_multiple_trials(num_trials, Y, initial_positions, epochs, tau, step_size_func, delta, kappa, base_seed=31, step_size_name='', energy_harvester_type='stationary', p_value=2):
    results = Parallel(n_jobs=-1)(
        delayed(run_single_trial)(
            Y,
            initial_positions,
            epochs,
            tau,
            step_size_func,
            delta,
            kappa,
            i,
            base_seed,
            step_size_name,
            energy_harvester_type,
            p_value
        )
        for i in range(num_trials)
    )
    # Rest of the function remains the same

    all_F_values, all_P_values, all_gradient_norms = zip(*results)

    # Create directory for saving plots
    plot_dir = f'plots/step_{step_size_name}_delta_{delta}_kappa_{kappa}'
    os.makedirs(plot_dir, exist_ok=True)

    # Filenames for plots
    F_values_filename = os.path.join(plot_dir, 'F_values.png')
    P_values_filename = os.path.join(plot_dir, 'P_values.png')
    gradient_norms_filename = os.path.join(plot_dir, 'gradient_norms.png')

    # Plot mean trajectories and gradients and save
    pm.plot_mean_trajectory(
        np.mean(all_F_values, axis=0),
        np.std(all_F_values, axis=0),
        label='F(x)',
        title=f'F(x) for kappa={kappa}, delta={delta}, step_size={step_size_name}',
        filename=F_values_filename
    )
    pm.plot_mean_trajectory(
        np.mean(all_P_values, axis=0),
        np.std(all_P_values, axis=0),
        label='P(x)',
        title=f'P(x) for kappa={kappa}, delta={delta}, step_size={step_size_name}',
        filename=P_values_filename
    )
    pm.plot_gradient_norms(
        np.mean(all_gradient_norms, axis=0),
        title=f'Gradient Norms for kappa={kappa}, delta={delta}, step_size={step_size_name}',
        filename=gradient_norms_filename
    )

# Function to run a single trial
def run_single_trial(Y, initial_positions, epochs, tau, step_size_func, delta, kappa, trial_idx, base_seed, step_size_name, energy_harvester_type='stationary', p_value=2):
    # Set random seed for reproducibility
    seed = base_seed + trial_idx
    random_state = np.random.RandomState(seed)
    xi_samples = random_state.uniform(28, 29, num_samples)

    # Define states
    states = [0, 5, 10]

    if energy_harvester_type == 'stationary':
        # Define stationary transition matrix
        stationary_transition_matrix = np.array([
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7]
        ])

        # Create stationary energy harvesters
        energy_harvesters = [
            MarkovEnergyHarvester(
                states,
                stationary_transition_matrix,
                random_state=random_state
            ) for _ in range(D)
        ]

    elif energy_harvester_type == 'non_stationary':
        # Define non-stationary transition matrices
        transition_matrices_non_stationary = [
            np.array([
                [0.9, 0.05, 0.05],
                [0.1, 0.85, 0.05],
                [0.05, 0.1, 0.85]
            ]),
            np.array([
                [0.85, 0.1, 0.05],
                [0.05, 0.9, 0.05],
                [0.05, 0.1, 0.85]
            ]),
            np.array([
                [0.8, 0.1, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.2, 0.7]
            ]),
            # Add more matrices if needed
        ]

        # Create non-stationary energy harvesters
        energy_harvesters = [
            NonStationaryMarkovEnergyHarvester(
                states,
                transition_matrices_non_stationary,
                random_state=random_state
            ) for _ in range(D)
        ]
    else:
        raise ValueError("Invalid energy harvester type. Choose 'stationary' or 'non_stationary'.")

    # Create agents with the selected energy harvesters
    agents = [
        Agent(
            position,
            tau=tau,
            energy_harvester=energy_harvester,
            random_state=random_state,
            p_value=p_value  
        ) for position, energy_harvester in zip(initial_positions, energy_harvesters)
    ]

    # Create directory for saving plots
    plot_dir = f'plots/step_{step_size_name}_delta_{delta}_kappa_{kappa}'
    os.makedirs(plot_dir, exist_ok=True)

    # Filenames for plots
    initial_positions_filename = os.path.join(plot_dir, f'initial_positions_trial_{trial_idx}.png')
    trajectories_filename = os.path.join(plot_dir, f'trajectories_trial_{trial_idx}.png')
    heatmap_filename = os.path.join(plot_dir, f'detection_error_heatmap_trial_{trial_idx}.png')

    # Plot initial positions and save
    pm.plot_initial_positions(
        np.array([agent.position for agent in agents]),
        Y,
        region_radius,
        filename=initial_positions_filename
    )

    # Run SGD optimization
    sgd_instance = SGD(agents, Y, epochs, tau, step_size_func, delta, kappa, xi_samples)
    position_history, gradient_norms = sgd_instance.run()

    # Plot results and save
    pm.plot_trajectories_with_delays(
        position_history,
        agents,
        Y,
        region_radius,
        filename=trajectories_filename
    )
    pm.plot_detection_error_heatmap(
        agents,
        Y,
        region_radius,
        xi_samples,
        filename=heatmap_filename
    )

    # Collect data
    data = {
        'agents_positions': [agent.trajectory for agent in agents],
        'agents_energy_levels': [agent.energy_levels for agent in agents],
        'agents_harvested_energy': [agent.harvested_energy_history for agent in agents],
        'agents_energy_consumption': [agent.energy_consumption_history for agent in agents],
        'agents_broadcast_times': [agent.broadcast_times for agent in agents],
        'targets_positions': Y.tolist(),
        'F_values': sgd_instance.F_values,
        'P_values': sgd_instance.P_values,
        'f_values': sgd_instance.f_values,
        'gradient_norms': sgd_instance.gradient_norms,
        'kappa': kappa,
        'delta': delta,
        'step_size_name': step_size_name,
        'trial_idx': trial_idx
    }

    # Write data to a text file
    write_data_to_file(data)
    plt.close('all')
    # Return collected data
    return data['F_values'], data['P_values'], data['gradient_norms']

# Main execution
def main():
    # Set the base random seed
    base_seed = 213
    np.random.seed(base_seed)

    # Initialize target and agent positions once
    Y = initialize_positions(region_radius, Y_num, random_state=np.random)
    initial_positions = initialize_positions(region_radius, D, random_state=np.random)

    # Define parameter grids
    kappa_values = [1,2,4]
    delta_values = [0.01,0.1,0.0001]
    
    # Define p values with higher granularity in (2, 3)
    p_values = [2, 2.2 , 2.4, 2.6, 2.8, 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10]   

    # Create step size functions for each p value
    step_size_functions = {
        f'step_size_rule_2_p{p_value}': (lambda p_value: lambda n: step_size_rule_2(n, p_value))(p_value)
        for p_value in p_values
    }

    # Include the baseline step size rule for comparison
    #step_size_functions['step_size_rule_1'] = step_size_rule_1

    energy_harvester_type = 'stationary'  # or 'non_stationary'

    # Define the number of trials for different ranges of p
    num_trials_p_in_2_3 = 2  # Increase the number of trials for p in (2, 3)
    num_trials_other_p = 2   # Fewer trials for other p values

    # Loop over kappa, delta, and step size functions
    for kappa in kappa_values:
        for delta in delta_values:
            for step_size_name, step_size_func in step_size_functions.items():
                # Debugging: Überprüfen, welche step_size_name verarbeitet werden
                print(f"Processing step size: {step_size_name}")

                # Prüfen, ob '_p' im Namen enthalten ist
                if '_p' in step_size_name:
                    p_value = float(step_size_name.split('_p')[1])
                else:
                    p_value = None  # Keine p-Werte für step_size_rule_1

                # Bestimme die Anzahl der Trials basierend auf dem p-Wert
                if p_value is not None and 2 < p_value < 3:
                    num_trials = num_trials_p_in_2_3
                else:
                    num_trials = num_trials_other_p

                print(f"Running simulations with kappa={kappa}, delta={delta}, step_size={step_size_name}, p={p_value}")
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
                    step_size_name=step_size_name,
                    energy_harvester_type=energy_harvester_type
                )

if __name__ == "__main__":
    main()
