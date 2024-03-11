import matplotlib.pyplot as plt
import numpy as np

# Simulation parameters
n_simulations = 1000  # Number of times the simulation is run
n_agents = 101  # Total number of agents
threshold_crowded = 50  # Threshold for the bar to be considered crowded

# Initialize arrays to store results
p_values = np.arange(0, 1.1, 0.1)  # Probability values from 0 to 1 in increments of 0.1
# p_values = np.arange(0.3, 0.71, 0.02)  # Probability values from 0 to 1 in increments of 0.1
times_crowded_by_p = []  # Times the bar becomes crowded for each p
average_going_by_p = []  # Average number of agents going to the bar for each p

# Run simulations for each p value
for p in p_values:
    times_crowded = 0
    go_decisions = []
    
    for _ in range(n_simulations):
        decisions = np.random.binomial(1, p, n_agents)
        n_going = np.sum(decisions)
        go_decisions.append(n_going)
        
        if n_going >= threshold_crowded:
            times_crowded += 1
    
    times_crowded_by_p.append(times_crowded)
    average_going_by_p.append(np.mean(go_decisions))

# Plot the results
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:red'
ax1.set_xlabel('Probability of Going to the Bar (p)')
ax1.set_ylabel('Times Bar Became Crowded', color=color)
ax1.plot(p_values, times_crowded_by_p, color=color, marker='o', label='Times Bar Became Crowded')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis
color = 'tab:blue'
ax2.set_ylabel('Average Number Going', color=color)  # We already handled the x-label with ax1
ax2.plot(p_values, average_going_by_p, color=color, marker='x', label='Average Number Going')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # Otherwise the right y-label is slightly clipped
plt.title('Impact of Decision Probability (p) on Bar Crowdedness and Attendance')
plt.show()
