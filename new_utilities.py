import matplotlib.pyplot as plt
import numpy as np

# Simulation parameters
n_agents = 101  # Total number of agents
threshold_crowded = 50  # Threshold for the bar to be considered crowded

def utility(n_going):
    is_crowded = n_going >= threshold_crowded

    if not is_crowded:
        # Calculate utility for uncrowded scenario
        if n_going == 0:
            return 0  # If no one is going, utility is 0
        elif n_going == threshold_crowded:
            return 1
        else:
            return (1 / threshold_crowded) * n_going

    else:
        # Calculate utility for crowded scenario
        if n_going == n_agents:
            return -1  # If all agents are going, utility is -1
        else:
            return -((1 / (n_agents - threshold_crowded)) * (n_going - threshold_crowded))

def payoff(n_going, decision):
    if decision == 1:
        return utility(n_going)
    else:
        return -utility(n_going)

# Generate n_going values from 0 to 100
n_going_values = np.arange(0, 101)

# Calculate payoff for each decision (going or staying) for each n_going value
payoff_going = [payoff(n_going, 1) for n_going in n_going_values]
payoff_staying = [payoff(n_going, 0) for n_going in n_going_values]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(n_going_values, payoff_going, label='Payoff for Going', color='blue')
plt.plot(n_going_values, payoff_staying, label='Payoff for Staying', color='red')
plt.xlabel('Number of Agents Going')
plt.ylabel('Payoff')
plt.title('Payoff Function Based on Number of Agents Going')
plt.legend()
plt.grid(True)
plt.show()
