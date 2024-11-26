Agent-Based Stochastic Gradient Descent Simulation with Energy Constraints and Communication Delays

This repository contains a Python implementation of a simulation framework for agent-based stochastic gradient descent (SGD) with energy constraints and communication delays. The simulation models agents operating in a circular region, each equipped with energy harvesters and subject to movement and broadcasting energy costs. Agents aim to minimize a global objective function while dealing with stochastic delays and energy limitations.

Table of Contents

	•	Features
	•	Installation
	•	Usage
	•	Running the Simulation
	•	Simulation Parameters
	•	Output
	•	Dependencies
	•	License

Features

	•	Agent-Based Modeling: Simulate multiple agents moving within a circular region, each with individual energy levels and harvesting capabilities.
	•	Energy Constraints: Agents consume energy when moving or broadcasting and harvest energy using Markov or Non-Stationary Markov models.
	•	Communication Delays: Agents experience stochastic delays in broadcasting their positions, modeled using a Zipf distribution.
	•	Stochastic Gradient Descent: Agents collaboratively perform SGD to minimize a global objective function, considering communication delays and energy constraints.
	•	Parallel Simulations: Run multiple simulation trials in parallel using joblib for efficient computation.
	•	Data Recording: Track various metrics such as agent positions, energy levels, harvested energy, energy consumption, and broadcasting times.
	•	Visualization: Generate plots for agent trajectories, detection error heatmaps, and convergence metrics.

Installation

	1.	Clone the repository:

git clone https://github.com/yourusername/yourrepository.git
cd yourrepository


	2.	Create a virtual environment (optional but recommended):

python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


	3.	Install the required packages:

pip install -r requirements.txt

Note: The requirements.txt file should contain the following packages:

numpy
scipy
matplotlib
joblib

Also, ensure that you have a module named Plot.py containing the plotting functions used in the code. Alternatively, you can replace calls to Plot with your own plotting functions.

Usage

Running the Simulation

The main simulation script is contained in simulation.py (assuming you name the file accordingly). You can run the simulation using:

python simulation.py

Simulation Parameters

The simulation can be customized by adjusting the parameters at the beginning of the script:
	•	Energy Constraints:
	•	ENABLE_ENERGY_CONSTRAINTS: Enable or disable energy constraints (True or False).
	•	MAX_ENERGY: Maximum energy agents can store.
	•	MOVEMENT_ENERGY_COST: Energy cost for moving.
	•	BROADCAST_ENERGY_COST: Energy cost for broadcasting.
	•	HIGH_ENERGY_THRESHOLD: Threshold for high energy state (default is 70% of MAX_ENERGY).
	•	LOW_ENERGY_THRESHOLD: Threshold for low energy state (default is 30% of MAX_ENERGY).
	•	Simulation Parameters:
	•	D: Number of agents.
	•	region_radius: Radius of the circular region where agents operate.
	•	Y_num: Number of target points in the region.
	•	epochs: Number of iterations for the simulation.
	•	num_samples: Number of stochastic samples for gradient estimation.
	•	tau: Communication delay parameter.
	•	num_trials: Number of simulation trials to run.
	•	Step Size Functions:
	•	step_size_rule_1(n): Basic step size rule.
	•	step_size_rule_2(n, p): Advanced step size rule with parameter p.
	•	Energy Harvesting Models:
	•	Stationary Markov Model: Agents harvest energy based on a stationary Markov process.
	•	Non-Stationary Markov Model: Agents harvest energy based on a non-stationary Markov process.
	•	Other Parameters:
	•	kappa: Penalty scaling factor in the objective function.
	•	delta: Threshold for penalty in the objective function.
	•	p_values: List of p values to use in the step size rule.

Customizing Parameters

You can modify the simulation parameters by editing the main() function in the script. For example:

# Define parameter grids
kappa_values = [1, 2, 4]
delta_values = [0.01, 0.1, 0.0001]

# Define p values for the step size function
p_values = [2, 2.2, 2.4, 2.6, 2.8, 3, 4, 5, 6, 7, 8, 9, 10]

Output

The simulation generates the following outputs:
	•	Data Files: For each trial, a text file containing detailed simulation data is saved. The filename format is:

[step_size_name]_k[kappa]_d[delta]_nr[trial_idx].txt


	•	Plots: The simulation generates several plots, saved in the plots directory:
	•	Initial Positions: Shows the initial positions of agents and targets.
	•	Agent Trajectories: Visualizes the movement of agents over time.
	•	Detection Error Heatmap: Displays the detection error over the region.
	•	Convergence Metrics: Plots of F(x), P(x), and gradient norms over epochs.

Dependencies

	•	Python 3.6 or higher
	•	NumPy
	•	SciPy
	•	Matplotlib
	•	Joblib

Make sure to have all dependencies installed. If using Plot.py for plotting functions, ensure it is available in your working directory or adjust the import statements accordingly.

License

This project is licensed under the MIT License - see the LICENSE file for details.
