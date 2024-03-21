
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

# Choose strategy function
def choose_strategy(strategy_id):
    strategies = {
        1: strategy_2_01,
        2: strategy_2_02,
        # 3: strategy_2_03,
    }
    return strategies.get(strategy_id)  # Default to strategy_1 if not found

factorList = {
    # 1: 0.548-0.05,
    1:0.74,
    2: 0.548,
    # 2: 0.548+0.2
    2:0.34
}

def strategy_2_01(X=3,  p=0.548, factor=factorList[1]):
    return factorList[1]
def strategy_2_02(X=3,  p=0.548, factor=factorList[2]):
    return factorList[2]


# def strategy_2_03(X=3,  p=0.548, factor=factorList[3]):
#     return factorList[3]


# Simulation parameters
days = 1000  # Number of times the simulation is run for testing
n_agents = 101  # Total number of agents
threshold_crowded = 50  # Threshold for the bar to be considered crowded
n_strats = 3 # Number of strategies

history = []
history_seperated = {i: [] for i in range(1, n_strats + 1)}

# Initialize agent profiles with unique IDs and strategies
agent_profiles = []
agents_per_strategy = n_agents // n_strats


for i in range(n_agents):
    strategy_id = (i // agents_per_strategy) % n_strats + 1  # Ensure strategy_id ranges between 1 and n_strats
    
    agent_profiles.append({
        'id': i,
        'strategy_id': strategy_id,
        'threshold_adjust': 0,
        'strategy_days': 2, 
        'p': 0.548,  # Initial probability
        'history': [],
        'cumulative_utility': 0,
        'utility_going': 0,
        'utility_staying': 0
    })

# Create a list of agent IDs
agent_ids = [agent['id'] for agent in agent_profiles]

# Simulation
for day in range(days):

    for i in range(1,n_strats+1):
        history_seperated[i].append(0)

    decisions = []
    # Randomize the order of agent IDs
    np.random.shuffle(agent_ids)
    
    for agent_id in agent_ids:
        agent = next(filter(lambda x: x['id'] == agent_id, agent_profiles))

        # Determine decision based on strategy
        strategy_func = choose_strategy(agent['strategy_id'])
        if callable(strategy_func):  # Check if strategy function is callable
            agent['p'] = strategy_func(agent['strategy_days'], agent['p']) # strat 2

        decision = np.random.binomial(1, agent['p'])
        decisions.append(decision)
        
        history_seperated[agent['strategy_id']][day]+=decision

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
            'utility': agent_utility,
            'p': agent['p'],
            'threshold_adjust': agent['threshold_adjust'],
            'cumulative_utility': agent['cumulative_utility'],
        })

# Calculate average utilities for agents going and staying
average_utilities_going = [agent['utility_going'] for agent in agent_profiles]
average_utilities_staying = [agent['utility_staying'] for agent in agent_profiles]
cumulative_utilities = [agent['cumulative_utility'] for agent in agent_profiles]

# Determine the colors and shapes for each cluster
color_mapping = {
    1: 'blue',
    2: 'green',
    3: 'orange'
}
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

for agent in agent_profiles:
    color = color_mapping[agent['strategy_id']]  # Get the color based on the agent's strategy ID
    plt.scatter(agent['id'], agent['cumulative_utility'], c=color, marker='x')

plt.xlabel('Agent ID')
plt.ylabel('Cumulative Utility')
plt.title(f'Cumulative Utility of Agents by Strategy over {days} Days')
plt.grid(True)
plt.show()

cumulative_utilities_by_strategy = {i: []  for i in range(1, n_strats + 1)}  # Initialize with zeros for each day
cumulative_utilities_by_strategy_day = {i: [0] * days for i in range(1, n_strats + 1)}  # Initialize with zeros for each day

for agent in agent_profiles:
    strategy_id = agent['strategy_id']
    cumulative_utilities_by_strategy[strategy_id].append(agent['cumulative_utility'])
    for d in range(days):
        cumulative_utilities_by_strategy_day[strategy_id][d] += agent['history'][d]['cumulative_utility']

# Calculate mean and standard deviation for each strategy's cumulative utility
mean_utilities_per_strategy = {}
std_dev_per_strategy = {}

for strategy_id, cumulative_utilities in cumulative_utilities_by_strategy.items():
    mean_utilities_per_strategy[strategy_id] = np.mean(cumulative_utilities)
    std_dev_per_strategy[strategy_id] = np.std(cumulative_utilities)

# Print mean and standard deviation per strategy
for strategy_id, mean_utility in mean_utilities_per_strategy.items():
    std_dev = std_dev_per_strategy[strategy_id]
    print(f"Strategy {factorList[strategy_id]}: Mean Utility = {mean_utility}, Standard Deviation = {std_dev}")


# Plotting the history of the number of agents going to the bar over time
plt.figure(figsize=(12, 6))

# Plotting only the first 200 points
plt.plot(range(1, 201), history[:200], label='Overall', color='black', linewidth=1)
plt.axhline(y=threshold_crowded, color='red', linestyle='--', label='Threshold Capacity')

# Fit a polynomial of degree 2 (you can change the degree as needed)
degree = 2
pfit = Polynomial.fit(range(1, 200 + 1), history[:200], degree)
trendline = pfit(range(1, 200 + 1))

# Plot the trendline
plt.plot(range(1, 200 + 1), trendline, label='Trendline', color='blue', linestyle='-')

plt.xlabel('Day')
plt.ylabel('Number of Agents at Bar')
plt.title('Bar attendance over time')
plt.legend()
plt.grid(True)
plt.show()
    

    

# bottom = np.zeros(days)  # Initialize the bottom of the bars to zero

# # Plotting a stacked bar chart for each strategy
# for strategy_id, counts in history_seperated.items():
#     color = color_mapping[strategy_id]  # Get the color based on the strategy ID
#     plt.bar(range(1, len(counts) + 1), counts, label=f'Strategy {strategy_id}', color=color, alpha=0.6, bottom=bottom)
#     bottom += np.array(counts)  # Update the bottom for the next set of bars

# plt.xlabel('Day')
# plt.ylabel('Number of Agents at Bar')
# plt.title('Bar attendance over time by strategy')
# plt.legend()
# plt.grid(True)
# plt.show()

# factorList = {
#     1: "Nash Equilibrium",
#     2: "Change p factor:0.05",
#     3: "Change threshold factor:0.05"
# }


# for agent in agent_profiles:
#     strategy_id = agent['strategy_id']
#     cumulative_utility = 0
#     for day in range(days):
#         total_utility = sum(agent['history'][d]['utility'] for d in range(day + 1) if agent['strategy_id'] == strategy_id)
#         cumulative_utility += total_utility
#         cumulative_utilities_by_strategy_day[strategy_id][day] = total_utility

for strategy_id, cumulative_utilities in cumulative_utilities_by_strategy_day.items():
    color = color_mapping[strategy_id]  # Get the color based on the strategy ID
    plt.plot(range(1, days + 1), cumulative_utilities, label=f'Strategy {factorList[strategy_id]}', color=color)

plt.xlabel('Day')
plt.ylabel('Cumulative Utility')
plt.title('Cumulative Utility by Strategy Over Time (p adjustment)')
plt.legend()
plt.grid(True)
plt.show()


# Keep track of which strategies have already been plotted
plotted_strategies = set()

# Iterate over agent profiles
for agent in agent_profiles:
    strategy_id = agent['strategy_id']
    color = color_mapping[strategy_id]  # Get color based on strategy ID
    p_values = [history_point['p'] for history_point in agent['history']]
    if strategy_id not in plotted_strategies:
        plt.plot(range(1, len(p_values) + 1), p_values, label=f'Strategy {factorList[strategy_id]}', color=color)
        
        # Print the results
        print(f"Statistical Information on P ADJUST {strategy_id} :")

        history_array = np.array(p_values)

        # Calculate statistical information
        mean = np.mean(history_array)
        median = np.median(history_array)
        std_dev = np.std(history_array)
        variance = np.var(history_array)

        print(f"Mean: {mean}")
        print(f"Median: {median}")
        print(f"Standard Deviation: {std_dev}")
        print(f"Variance: {variance}")
    plotted_strategies.add(strategy_id)

plt.xlabel('Day')
plt.ylabel('p value')
plt.title('Change in p values for each agent over time')
plt.legend()
plt.grid(True)
plt.show()