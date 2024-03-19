function equilibrium_probability = find_equilibrium_probability()
    total_players = 101; % Number of players excluding the player of interest
    threshold = 50; % Threshold for the bar being crowded
    initial_guess = 0.3; % Initial guess for the equilibrium probability

    % Define the function for the difference in expected payoffs
    payoff_difference = @(p) calculate_payoff_difference(p, total_players, threshold);

    % Find the root of the equation using fzero
    equilibrium_probability = fzero(payoff_difference, initial_guess);

    % Plot utility function to verify its correctness
    plot_utility_function(total_players, threshold);

    % Plot the expected payoffs function to find the equilibrium probability visually
    plot_expected_payoffs(total_players, threshold);
end

function diff = calculate_payoff_difference(p, total_players, threshold)
    n_agents = total_players; % Including the player of interest

    % Calculate the expected utility for going and staying
    E_G = 0; E_S = 0;
    for n_going = 0:total_players
        prob = custom_binopdf(n_going, total_players, p); % Probability of n_going players going
        utility_going = utility(n_going, n_agents, threshold);
        utility_staying = -utility(n_going, n_agents, threshold);

        E_G = E_G + prob * utility_going; % Expected utility for going
        E_S = E_S + prob * utility_staying; % Expected utility for staying
    end

    % Difference between expected payoffs
    diff = E_G - E_S;
end

function u = utility(n_going, n_agents, threshold_crowded)
    if n_going >= threshold_crowded
        if n_going == n_agents
            u = -1; % If all agents are going, utility is -1
        else
            u = -((1 / (n_agents - threshold_crowded)) * (n_going - threshold_crowded));
        end
    else
        if n_going == 0
            u = 0; % If no one is going, utility is 0
        elseif n_going == threshold_crowded
            u = 1; % If exactly the threshold number is going, utility is 1
        else
            u = (1 / threshold_crowded) * n_going;
        end
    end
end

function p = custom_binopdf(k, n, p_success)
    p = arrayfun(@(x) nchoosek(n, x) * p_success^x * (1 - p_success)^(n - x), k);
end

function plot_utility_function(n_agents, threshold_crowded)
    n_going_values = 0:n_agents;
    utilities_going = arrayfun(@(x) utility(x, n_agents, threshold_crowded), n_going_values);
    utilities_staying = -utilities_going;

    figure;
    plot(n_going_values, utilities_going, 'b', 'LineWidth', 2); % Utility for going (blue)
    hold on;
    plot(n_going_values, utilities_staying, 'r', 'LineWidth', 2); % Utility for staying (red)
    hold off;
    xlabel('Number of Agents Going to Bar');
    ylabel('Utility');
    title('Utility Function for Going and Staying');
    legend('Utility for Going', 'Utility for Staying');
    grid on;
end

function plot_expected_payoffs(total_players, threshold)
    probabilities = 0:0.01:1;
    expected_payoffs_going = zeros(size(probabilities));
    expected_payoffs_staying = zeros(size(probabilities));

    for i = 1:length(probabilities)
        p = probabilities(i);
        [expected_payoffs_going(i), expected_payoffs_staying(i)] = calculate_expected_payoffs(p, total_players, threshold);
    end

    payoff_difference = @(p) calculate_payoff_difference(p, total_players, threshold);
    equilibrium_p = fzero(payoff_difference, 0.5);

    figure;
    plot(probabilities, expected_payoffs_going, 'b-', 'LineWidth', 2);
    hold on;
    plot(probabilities, expected_payoffs_staying, 'r-', 'LineWidth', 2);
    plot(equilibrium_p, calculate_payoff_difference(equilibrium_p, total_players, threshold), 'ko', 'MarkerSize', 10);
    hold off;
    xlabel('Probability of Going to the Bar (p)');
    ylabel('Expected Payoff');
    title('Expected Payoffs for Going and Staying vs. Probability');
    legend('Expected Payoff for Going', 'Expected Payoff for Staying', 'Equilibrium Probability');
    grid on;
end

function [E_G, E_S] = calculate_expected_payoffs(p, total_players, threshold)
    n_agents = total_players;
    E_G = 0; E_S = 0;

    for n_going = 0:total_players
        prob = custom_binopdf(n_going, total_players, p);
        utility_going = utility(n_going, n_agents, threshold);
        utility_staying = -utility(n_going, n_agents, threshold);

        E_G = E_G + prob * utility_going;
        E_S = E_S + prob * utility_staying;
    end
end