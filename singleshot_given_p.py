import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from scipy import stats

# Define the utility function
def utility(n_going):
    is_crowded = n_going >= threshold_crowded
    if not is_crowded:
        if n_going == 0:
            return 0  # If no one is going, utility is 0
        elif n_going == threshold_crowded:
            return 1
        else:
            return (1 / threshold_crowded) * n_going
    else:
        if n_going == n_agents:
            return -1  # If all agents are going, utility is -1
        else:
            return -((1 / (n_agents - threshold_crowded)) * (n_going - threshold_crowded))

# Define the payoff function
def payoff(n_going, decision):
    if decision == 1:
        return utility(n_going)
    else:
        return -utility(n_going)  # Assuming staying has the opposite utility effect

# Simulation parameters
# days = 200  # Number of times the simulation is run for testing
days = 1  # Number of times the simulation is run for testing
n_agents = 101  # Total number of agents
threshold_crowded = 50  # Threshold for the bar to be considered crowded
n_going = 0  # Number of agents going to the bar
history = []

# Strategy probabilities
# strategy_p = [0.45]  # -8500
# strategy_p = [0.5]  # -4000
# strategy_p = [0.495]  # -4000
# strategy_p = [0.53]  # -2700
strategy_p = [0.548]  # -2650
# strategy_p = [0.55]  # -2650
# strategy_p = [0.56]  # -3000
# strategy_p = [0.561]  # -3000
# strategy_p = [0.25, 0.5, 0.75]  # Adjusted to contain different strategies
# strategy_p = [0.96, 0.35, 0.75]  # Adjusted to contain different strategies

# Initialize agent profiles with unique IDs and strategies
agent_profiles = [{
    'id': i,
    'p': strategy_p[i // (n_agents // len(strategy_p))] if i < (n_agents // len(strategy_p)) * len(strategy_p) else strategy_p[-1],  # Assign strategy based on position
    'history': [],
    'cumulative_utility': 0,
    'utility_going': 0,
    'utility_staying': 0
} for i in range(n_agents)]

# Create a list of agent IDs
agent_ids = [agent['id'] for agent in agent_profiles]

# Simulation
for day in range(days):
    decisions = []
    # Randomize the order of agent IDs
    np.random.shuffle(agent_ids)
    
    for agent_id in agent_ids:
        agent = next(filter(lambda x: x['id'] == agent_id, agent_profiles))
        # Determine decision based on strategy
        decision = np.random.binomial(1, agent['p'])
        decisions.append(decision)

    n_going = np.sum(decisions)
    history.append(n_going)

    for agent_id, decision in zip(agent_ids, decisions):
        agent = next(filter(lambda x: x['id'] == agent_id, agent_profiles))
        agent_utility = payoff(n_going, decision)  # Use the payoff function here
        agent['cumulative_utility'] += agent_utility  # Update cumulative utility
        agent['utility_going'] += agent_utility if decision == 1 else 0
        agent['utility_staying'] += agent_utility if decision == 0 else 0
        
        agent['history'].append({
            'day': day,
            'decision': decision,
            'n_going': n_going,
            'utility': agent_utility
        })

# Calculate average utilities for agents going and staying
average_utilities_going = [agent['utility_going'] for agent in agent_profiles]
average_utilities_staying = [agent['utility_staying'] for agent in agent_profiles]
cumulative_utilities = [agent['cumulative_utility'] for agent in agent_profiles]

# Determine the colors and shapes for each cluster
colors = ['blue', 'green', 'orange']
shapes = ['o', 's']  # Circles for going, squares for staying


# Convert history to a NumPy array for easier statistical calculations
history_array = np.array(history)

# Calculate statistical information
mean = np.mean(history_array)
median = np.median(history_array)
std_dev = np.std(history_array)
variance = np.var(history_array)

# Print the results
print("Statistical Information on History Array:")
print(f"Mean: {mean}")
print(f"Median: {median}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")

# Plotting average utilities for agents going and staying
plt.figure(figsize=(12, 6))

for i in range(len(strategy_p)):
    start_index = (n_agents // len(strategy_p)) * i
    end_index = (n_agents // len(strategy_p)) * (i + 1)
    # plt.scatter(range(start_index, end_index), average_utilities_going[start_index:end_index], c=colors[i], label=f'(Strategy p={strategy_p[i]})', marker=shapes[0])
    # plt.scatter(range(start_index, end_index), average_utilities_staying[start_index:end_index], c=colors[i], marker=shapes[1])
    plt.scatter(range(start_index, end_index), cumulative_utilities[start_index:end_index], c=colors[i], marker='x')

plt.xlabel('Agent ID')
plt.ylabel('Average Utility')
plt.title(f'Cumulative Utility of Agents over {days} Days. All with Strategy p={strategy_p[0]}. {n_going} Agents at bar. {n_agents-n_going} Agents not at bar')
plt.legend()
plt.grid(True)
plt.show()
