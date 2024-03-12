
import matplotlib.pyplot as plt
import numpy as np

def choose_strategy(strategy_id):
    strategies = {
        1: strategy_1,
        2: strategy_2,
        3: strategy_3,
        4: strategy_4,
    }
    return strategies.get(strategy_id, strategy_1)  # Default to strategy_1 if not found


# Pure Strategies
def strategy_1(X=1):
    avg_n_going = np.mean(history[-X:]) if len(history) >= X else 0
    return 0 if avg_n_going > threshold_crowded else 1

def strategy_2(X=1, factor=0.1, p=0.5):
    last_n_going = np.mean(history[-X:]) if len(history) >= X else 0
    difference = last_n_going - threshold_crowded
    adjustment = -difference * (1 / (n_agents - threshold_crowded))  * factor
    new_p = min(1, max(0, p + adjustment))
    return new_p

def strategy_3(X=1, factor=0.1):
    last_n_going = np.mean(history[-X:]) if len(history) >= X else 0
    difference = threshold_crowded - last_n_going

    # Ensure that the probability stays within [0, 1]
    probability = max(0, min(1, (difference + threshold_crowded) / (2 * threshold_crowded)))
    return probability

history = [0, 0, 70, 80]
days = 1000  # Number of times the simulation is run for testing
n_agents = 100  # Total number of agents
threshold_crowded = 50  # Threshold for the bar to be considered crowded

print(strategy_3(2))
print(strategy_3(4))