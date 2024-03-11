import matplotlib.pyplot as plt
import numpy as np

# Simulation parameters
n_simulations = 100  # Number of times the simulation is run
n_agents = 100  # Total number of agents
threshold_crowded = 50  # Threshold for the bar to be considered crowded

# Define the utility function
def utility(n_going):
    is_crowded = n_going > threshold_crowded

    if not is_crowded:
        # Calculate utility for uncrowded scenario
        if n_going == 0:
            return 0  # If no one is going, utility is 0
        elif n_going == threshold_crowded:
            return 1
        else:
            return (1 / 50) *  (n_going) 

    else:
       # Calculate utility for uncrowded scenario
        if n_going == 0:
            return 0  # If no one is going, utility is 0
        elif n_going == threshold_crowded:
            return -1
        else:
            return - (1 / (n_agents-threshold_crowded)) *  (n_going-threshold_crowded) 

def payoff(n_going, decision):
    if decision == 1:
        return utility(n_going)
    else:
        return -utility(n_going)

# Initialize arrays to store results
p_values = np.arange(0, 1.01, 0.01)  # 
# p_values = np.arange(0.3, 0.71, 0.001)  # 
sum_average_utilities_fine = []  # Sum of average utilities for the fine p values

# Calculate the sum of average utilities for each fine p value
for p in p_values:
    total_utilities = []
    
    for _ in range(n_simulations):
        decisions = np.random.binomial(1, p, n_agents)
        n_going = np.sum(decisions)
        
                # Calculate total utility based on payoff function
        total_utility = sum([payoff(n_going, decision) for decision in decisions])
        total_utilities.append(total_utility)
        
    # Average total utility for this p
    sum_average_utilities_fine.append(np.mean(total_utilities))

# Plot the sum of average utilities for the finer range of p values
plt.figure(figsize=(12, 6))
plt.plot(p_values, sum_average_utilities_fine, label='Sum of Average Utilities', marker='o', color='orange')
plt.xlabel('Probability of Going to the Bar (p)')
plt.ylabel('Sum of Average Utilities')
plt.title('Sum of Average Utilities for All Agents at Different Probabilities (p) - Fine Sampling')
plt.legend()
plt.grid(True)
plt.show()
