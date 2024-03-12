# strategies:
# pure
# - Look at X last days, take average of people going, do the same check as above and change p
# # - Look at the last X days, try to predict the next day, see how that compares to threshold, change p accordingly
# mixed form. Given a current p
# - Look at X days ago, if higher/lower than threshold reduce/increase p by a threshold*factor
# - Look at X days ago, if higher/lower than threshold reduce/increase p by 0.1*factor
# - Look at X days ago, see difference to threshold, make p += difference
# - Look at X days ago, see difference to threshold, make p = 1-threshold

# I would like these strategeis to be randomly allocated to each agent, and record which strategy is allocated to which agent
# I would like their strategies to be displayed in the graph based on color, and make sure that the key reflects this. 
# I would like them to change their strategy when ever they lose (when their utility is negative)

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

# Choose strategy function
def choose_strategy(strategy_id):
    strategies = {
        1: strategy_1,
        2: strategy_2,
        3: strategy_3,
        # 4: strategy_4,
    }
    return strategies.get(strategy_id, strategy_1)  # Default to strategy_1 if not found

# Pure Strategies
def strategy_1(X=1, p=0.561):
    if len(history) < X:
        return p
    avg_n_going = np.mean(history[-X:]) if len(history) >= X else 0
    return 0 if avg_n_going > threshold_crowded else 1

def strategy_2(X=1,  p=0.561, factor=0.1):
    if len(history) < X:
        return p
    last_n_going = np.mean(history[-X:]) if len(history) >= X else 0
    difference = last_n_going - threshold_crowded
    adjustment = -difference * (1 / (n_agents - threshold_crowded))  * factor
    new_p = min(1, max(0, p + adjustment))
    return new_p

def strategy_3(X=1,  p=0.561):
    if len(history) < X:
        return p
    last_n_going = np.mean(history[-X:]) if len(history) >= X else 0
    difference = threshold_crowded - last_n_going

    # Ensure that the probability stays within [0, 1]
    probability = max(0, min(1, (difference + threshold_crowded) / (2 * threshold_crowded)))
    return probability

history = []
# Simulation parameters
days = 200  # Number of times the simulation is run for testing
n_agents = 100  # Total number of agents
threshold_crowded = 50  # Threshold for the bar to be considered crowded
n_strats = 3  # Number of strategies

# Strategy probabilities
# strategy_p = [0.5]  # Adjusted to contain different strategies
# strategy_p = [0.25, 0.5, 0.75]  # Adjusted to contain different strategies
# strategy_p = [0.96, 0.35]  # Adjusted to contain different strategies

# Initialize agent profiles with unique IDs and strategies
agent_profiles = []
for i in range(n_agents):
    if i < n_agents // 3:
        strategy_id = 1  # First third gets strategy 1
    elif i < 2 * n_agents // 3:
        strategy_id = 2  # Second third gets strategy 2
    else:
        strategy_id = 3  # Last third gets strategy 3
    
    agent_profiles.append({
        'id': i,
        'strategy_id': strategy_id,
        'strategy_factor': np.random.rand(),  # Randomly choose a strategy probability between 0 and 1
        'strategy_days': 1,  # Randomly choose a strategy days between 1 and 4
        # 'strategy_days': np.random.randint(1, 5),  # Randomly choose a strategy days between 1 and 4
        'p': 0.561,  # Initial probability
        # 'p': np.random.rand(),  # Initial probability
        'history': [],
        'cumulative_utility': 0,
        'utility_going': 0,
        'utility_staying': 0
    })

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
        strategy_func = choose_strategy(agent['strategy_id'])
        if callable(strategy_func):  # Check if strategy function is callable
            if (agent['strategy_id'] == 2):
                agent['p'] = strategy_func(agent['strategy_days'], agent['p'], agent['strategy_factor'])
            else: 
                agent['p'] = strategy_func(agent['strategy_days'], agent['p']) 

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

# Plotting average utilities for agents going and staying
plt.figure(figsize=(12, 6))

for i in range(n_strats):
    start_index = (n_agents // n_strats) * i
    end_index = (n_agents // n_strats) * (i + 1)
    # plt.scatter(range(start_index, end_index), average_utilities_going[start_index:end_index], c=colors[i], label=f'(Strategy {i})', marker=shapes[0])
    # plt.scatter(range(start_index, end_index), average_utilities_staying[start_index:end_index], c=colors[i], marker=shapes[1])
    plt.scatter(range(start_index, end_index), cumulative_utilities[start_index:end_index], c=colors[i], marker='x')

plt.xlabel('Agent ID')
plt.ylabel('Average Utility')
plt.title(f'Average Utility of Agents Going and Staying over {days} Days')
plt.legend()
plt.grid(True)
plt.show()
# print(agent_profiles)

# Plotting the history of the number of agents going to the bar over time
plt.figure(figsize=(12, 6))
plt.plot(range(1, days + 1), history, color='black', marker='o', linestyle='-', linewidth=2)
plt.xlabel('Day')
plt.ylabel('Number of Agents at Bar')
plt.title('Bar attendance over time')
plt.grid(True)
plt.show()

# Calculate cumulative utilities for each day
# Calculate cumulative utilities by strategy
# Calculate cumulative utilities by strategy
cumulative_utilities_by_strategy = {i: [0] * days for i in range(1, n_strats + 1)}  # Initialize with zeros for each day

for agent in agent_profiles:
    strategy_id = agent['strategy_id']
    cumulative_utility = 0
    for day in range(days):
        total_utility = sum(agent['history'][d]['utility'] for d in range(day + 1) if agent['strategy_id'] == strategy_id)
        cumulative_utility += total_utility
        cumulative_utilities_by_strategy[strategy_id][day] = total_utility

# Plotting the cumulative utilities by strategy over time
plt.figure(figsize=(12, 6))

for strategy_id, cumulative_utilities in cumulative_utilities_by_strategy.items():
    plt.plot(range(1, days + 1), cumulative_utilities, label=f'Strategy {strategy_id}')

plt.xlabel('Day')
plt.ylabel('Cumulative Utility')
plt.title('Cumulative Utility by Strategy Over Time')
plt.legend()
plt.grid(True)
plt.show()


# # Plotting the cumulative utilities over time
# plt.figure(figsize=(12, 6))
# plt.plot(range(1, days + 1), cumulative_utilities_by_day, color='orange', marker='o', linestyle='-', linewidth=2)
# plt.xlabel('Day')
# plt.ylabel('Cumulative Utility')
# plt.title('Cumulative Utility of All Agents Over Time')
# plt.grid(True)
# plt.show()
