import matplotlib.pyplot as plt
import numpy as np

# Simulation parameters
n_simulations = 1000  # Number of times the simulation is run
n_agents = 100  # Total number of agents
threshold_crowded = 50  # Threshold for the bar to be considered crowded

# Define the utility function
def utility(n_going):
    is_crowded = n_going > threshold_crowded
    if not is_crowded:
        if n_going == 0:
            return 0
        elif n_going == threshold_crowded:
            return 1
        else:
            return (1 / 50) * n_going
    else:
        if n_going == 0:
            return 0
        elif n_going == threshold_crowded:
            return -1
        else:
            return - (1 / (n_agents - threshold_crowded)) * (n_going - threshold_crowded)

def payoff(n_going, decision):
    if decision == 1:
        return utility(n_going)
    else:
        return -utility(n_going)

# Initialize arrays to store results
# p_values = np.arange(0.4, 0.61, 0.01)  # Range of p values
p_values = np.arange(0, 1.01, 0.01)  # 
# p_values = np.arange(0.3, 0.71, 0.001)  # 

average_utilities_going = []  # Average utilities for going
average_utilities_staying = []  # Average utilities for staying
average_utilities_total = []  # Average total utilities

ratios_going_staying = []  # Ratio of number of people going to staying
average_n_going = []  # Average number of people going

for p in p_values:
    utilities_going = []
    utilities_staying = []
    n_going_total = 0  # Initialize total number of people going
    
    for _ in range(n_simulations):
        decisions = np.random.binomial(1, p, n_agents)
        n_going = np.sum(decisions)
        n_staying = n_agents - n_going  # Number of agents staying
        n_going_total += n_going  # Accumulate the number of people going
        
        # Calculate utilities for going and staying
        utility_going = utility(n_going)
        utility_staying = -utility(n_going)  # Use the same function for simplicity
        
        
        if n_going > 0:  # Avoid division by zero
            utilities_going.extend([utility_going] * n_going)
        if n_staying > 0:  # Avoid division by zero
            utilities_staying.extend([utility_staying] * n_staying)
    
    # Calculate average utilities and ratio for this p value
    average_utilities_going.append(np.mean(utilities_going) if utilities_going else 0)
    average_utilities_staying.append(np.mean(utilities_staying) if utilities_staying else 0)
    average_utilities_total.append(np.mean(utilities_going*n_going + utilities_staying*(n_agents-n_going)))
    
    average_n_going.append(n_going_total / n_simulations)  # Calculate the average number of people going

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:red'
ax1.set_xlabel('Probability of Going to the Bar (p)')
ax1.set_ylabel('Average Utility', color=color)
ax1.plot(p_values, average_utilities_going, label='Average Utility Going', color='red', marker='o')
ax1.plot(p_values, average_utilities_staying, label='Average Utility Staying', color='blue', marker='x')
ax1.plot(p_values, average_utilities_total, label='Average Total Utility', color='black', marker='s')
ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
# color = 'tab:blue'
# ax2.set_ylabel('Average Number of People Going', color=color)  # We already handled the x-label with ax1
# ax2.plot(p_values, average_n_going, label='Average Number of People Going', color='green', linestyle='--')
# ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # To ensure the right y-label is not slightly clipped
fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
plt.title('Utility and Average Number of People Going at Different Probabilities (p)')
plt.grid(True)
plt.show()
