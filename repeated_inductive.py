strategies:
pure
- Look at X last days, take average of people going, do the same check as above and change p
# - Look at the last X days, try to predict the next day, see how that compares to threshold, change p accordingly
mixed form. Given a current p
- Look at X days ago, if higher/lower than threshold reduce/increase p by a threshold*factor
- Look at X days ago, if higher/lower than threshold reduce/increase p by 0.1*factor
- Look at X days ago, see difference to threshold, make p += difference
- Look at X days ago, see difference to threshold, make p = 1-threshold

I would like these strategeis to be randomly allocated to each agent, and record which strategy is allocated to which agent
I would like their strategies to be displayed in the graph based on color, and make sure that the key reflects this. 
I would like them to change their strategy when ever they lose (when their utility is negative)

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

def choose_strategy(strategy_id):
    implement a switch case to choose the strategy based on the strategy_id
    return strategy as a callable function

# PURE STRATEGIES

def strategy_1(X=1):
    look at the last X values of the array history. 
    take the average of the values, check if that is higher than threshold
    return 0 if higher, 1 if lower

def strategy_2(X=1):
    look at the last X values of the array history. 
    try to predict the next day, see how that compares to threshold
    return 1 if higher, 0 if lower

# MIXED STRATEGIES

def strategy_3(X=1, factor=0.1, p=0.5):
    look at the last X values of the array history. 
    get difference to threshold.
    increase p by difference*(1/(n_agents-threshold)) if lower, decrease p by difference*factor if higher
    return p

def strategy_4(X=1, factor=0.1):
    look at the last X values of the array history. 
    get difference to threshold.
    return 1-

history = []
# Simulation parameters
days = 1000  # Number of times the simulation is run for testing
n_agents = 100  # Total number of agents
threshold_crowded = 50  # Threshold for the bar to be considered crowded

# Strategy probabilities
# strategy_p = [0.5]  # Adjusted to contain different strategies
# strategy_p = [0.25, 0.5, 0.75]  # Adjusted to contain different strategies
strategy_p = [0.96, 0.35, 0.75]  # Adjusted to contain different strategies

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

# Plotting average utilities for agents going and staying
plt.figure(figsize=(12, 6))

for i in range(len(strategy_p)):
    start_index = (n_agents // len(strategy_p)) * i
    end_index = (n_agents // len(strategy_p)) * (i + 1)
    plt.scatter(range(start_index, end_index), average_utilities_going[start_index:end_index], c=colors[i], label=f'(Strategy p={strategy_p[i]})', marker=shapes[0])
    plt.scatter(range(start_index, end_index), average_utilities_staying[start_index:end_index], c=colors[i], marker=shapes[1])
    plt.scatter(range(start_index, end_index), cumulative_utilities[start_index:end_index], c=colors[i], marker='x')

plt.xlabel('Agent ID')
plt.ylabel('Average Utility')
plt.title(f'Average Utility of Agents Going and Staying over {days} Days')
plt.legend()
plt.grid(True)
plt.show()
