import matplotlib.pyplot as plt
import numpy as np

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
days = 1000  # Number of times the simulation is run for testing
n_agents = 100  # Total number of agents
threshold_crowded = 50  # Threshold for the bar to be considered crowded

# Strategy probabilities
strategy_p = [0.5]  # Adjusted to contain only one strategy for demonstration

# Initialize agent profiles with unique IDs and strategies
agent_profiles = [{
    'id': i,
    'p': strategy_p[0],  # Example logic for assigning strategies
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

# Plotting cumulative utility over the days
cumulative_utilities = [agent['cumulative_utility'] for agent in agent_profiles]
plt.figure(figsize=(12, 6))
plt.scatter(range(n_agents), cumulative_utilities, c='blue')
plt.xlabel('Agent ID')
plt.ylabel('Cumulative Utility')
plt.title(f'Cumulative Utility of Agents over {days} Days')
plt.grid(True)
plt.show()
