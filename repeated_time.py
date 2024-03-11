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
n_agents = 100  # Total number of agents
threshold_crowded = 50  # Threshold for the bar to be considered crowded
strategy_p = 0.5  # Constant probability for simplicity

# Tracking average cumulative utility over days
average_cumulative_utilities_over_days = []

for days in range(1, 1001):  # From 1 to 1000 days
    agent_profiles = [{'cumulative_utility': 0} for _ in range(n_agents)]  # Reset agents for each simulation length
    
    for day in range(days):
        decisions = np.random.binomial(1, strategy_p, n_agents)
        n_going = np.sum(decisions)
        
        for i, decision in enumerate(decisions):
            agent_profiles[i]['cumulative_utility'] += payoff(n_going, decision)
    
    # Calculate the average cumulative utility at the end of this simulation
    average_cumulative_utility = np.mean([agent['cumulative_utility'] for agent in agent_profiles])
    average_cumulative_utilities_over_days.append(average_cumulative_utility)

# Plotting the effect of changing days on the average cumulative utility
plt.figure(figsize=(12, 6))
plt.plot(range(1, 1001), average_cumulative_utilities_over_days, color='blue')
plt.xlabel('Number of Days')
plt.ylabel('Average Cumulative Utility')
plt.title('Effect of Number of Days on Average Cumulative Utility')
plt.grid(True)
plt.show()
