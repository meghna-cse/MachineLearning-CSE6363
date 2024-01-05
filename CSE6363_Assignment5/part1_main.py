# -------------------------------------------------------------------------
#  Part 1: Q-Learning and Policy Iteration on the Frozen Lake Environment
# -------------------------------------------------------------------------

# The implementation of Q-Learning algorithm is based on the tutorial 
# from the following link: https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/


# Import necessary libraries
from pathlib import Path
from typing import NamedTuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# Set up the visual theme for plots
sns.set_theme()

# Define parameters for the Q-Learning algorithm and environment setup
class Params(NamedTuple):
    total_episodes: int         # Total episodes
    learning_rate: float        # Learning rate
    gamma: float                # Discounting rate
    epsilon: float              # Exploration probability
    map_size: int               # Number of tiles of one side of the squared environment
    seed: int                   # Define a seed so that we get reproducible results
    is_slippery: bool           # If true the player will move in intended direction with probability of 1/3 else will move in either perpendicular direction with equal probability of 1/3 in both directions
    n_runs: int                 # Number of runs
    action_size: int            # Number of possible actions
    state_size: int             # Number of possible states
    proba_frozen: float         # Probability that a tile is frozen
    savefig_folder: Path        # Root folder where plots are saved

# Initialize environment parameters with specific values
params = Params(
    total_episodes=2000,
    learning_rate=0.8,
    gamma=0.95,
    epsilon=0.1,
    map_size=5,
    seed=123,
    is_slippery=False,
    n_runs=20,
    action_size=None,
    state_size=None,
    proba_frozen=0.9,
    savefig_folder = Path(__file__).parent,
)

# Set up a random number generator with a seed for reproducibility
rng = np.random.default_rng(params.seed)

# Ensure the folder for saving figures exists
params.savefig_folder.mkdir(parents=True, exist_ok=True)

# Create the Frozen Lake environment with custom settings
env = gym.make(
    "FrozenLake-v1",
    is_slippery=params.is_slippery,
    render_mode="rgb_array",
    desc=generate_random_map(
        size=params.map_size, p=params.proba_frozen, seed=params.seed
    ),
)

# Update the action and state sizes in parameters based on the environment
params = params._replace(action_size=env.action_space.n)
params = params._replace(state_size=env.observation_space.n)
#print(f"Action size: {params.action_size}")
#print(f"State size: {params.state_size}")

# -------------------------------------------------------------------------
# Q-Learning Class: Implements the Q-Learning algorithm
# -------------------------------------------------------------------------
class Qlearning:
    # Initialization method
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    # Update method - Implements the core Q-learning update rule
    def update(self, state, action, reward, new_state):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        q_update = self.qtable[state, action] + self.learning_rate * delta
        return q_update

    # Method to reset the Q-table to its initial state
    def reset_qtable(self):
        """Reset the Q-table."""
        self.qtable = np.zeros((self.state_size, self.action_size))


# -----------------------------------------------------------------------------------------
# Epsilon-Greedy Strategy Class: Balances exploration and exploitation in action selection
# -----------------------------------------------------------------------------------------
class EpsilonGreedy:
    # Initialization method
    def __init__(self, epsilon):
        self.epsilon = epsilon

    # Method to choose an action based on the current state and Q-table
    def choose_action(self, action_space, state, qtable):
        """Choose an action `a` in the current world state (s)."""
        # First we randomize a number
        explor_exploit_tradeoff = rng.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()

        # Exploitation (taking the biggest Q-value for this state)
        else:
            # Break ties randomly
            # If all actions are the same for this state we choose a random one
            # (otherwise `np.argmax()` would always take the first one)
            if np.all(qtable[state, :]) == qtable[state, 0]:
                action = action_space.sample()
            else:
                action = np.argmax(qtable[state, :])
        return action


# -------------------------------------------------------------------------
# Running the Environment: Function to run Q-Learning on the environment
# -------------------------------------------------------------------------
# This function iterates through episodes and runs, updating the Q-table and recording rewards and steps
def run_env():
    rewards = np.zeros((params.total_episodes, params.n_runs))
    steps = np.zeros((params.total_episodes, params.n_runs))
    episodes = np.arange(params.total_episodes)
    qtables = np.zeros((params.n_runs, params.state_size, params.action_size))
    all_states = []
    all_actions = []

    for run in range(params.n_runs):           # Run several times to account for stochasticity
        learner = Qlearning(
            learning_rate=params.learning_rate,
            gamma=params.gamma,
            state_size=params.state_size,
            action_size=params.action_size,
        )
        explorer = EpsilonGreedy(
            epsilon=params.epsilon,
        )

        for episode in tqdm(
            episodes, desc=f"Q-Learning Algorithm Run {run}/{params.n_runs} - Episodes", leave=False
        ):
            state = env.reset(seed=params.seed)[0]  # Reset the environment
            step = 0
            done = False
            total_rewards = 0

            while not done:
                action = explorer.choose_action(
                    action_space=env.action_space, state=state, qtable=learner.qtable
                )

                # Log all states and actions
                all_states.append(state)
                all_actions.append(action)

                # Take the action (a) and observe the outcome state(s') and reward (r)
                new_state, reward, terminated, truncated, info = env.step(action)

                done = terminated or truncated
                learner.qtable[state, action] = learner.update(state, action, reward, new_state)
                total_rewards += reward
                step += 1

                # Our new state is state
                state = new_state

            # Log all rewards and steps
            rewards[episode, run] = total_rewards
            steps[episode, run] = step
        qtables[run, :, :] = learner.qtable

    return rewards, steps, episodes, qtables, all_states, all_actions



# -------------------------------------------------------------------------
# Policy Iteration Function: Iteratively improves policy until convergence
# -------------------------------------------------------------------------
def policy_iteration(env_parameters, environment):
    # Initialize a policy with arbitrary actions (1 for each state) and a value function with zeros
    optimal_policy = np.ones(environment.observation_space.n, dtype=int)
    state_value_function = np.zeros(environment.observation_space.n)

    discount_factor = env_parameters.gamma      # gamma
    convergence_threshold = 1e-5                # theta

    while True:
        # Update value function based on current policy
        while True:
            value_change = 0
            for state in range(environment.observation_space.n):
                prev_value = state_value_function[state]
                action = optimal_policy[state]

                # Calculate expected return for the current state-action pair
                expected_return = sum(
                    prob * (reward + discount_factor * state_value_function[next_state])
                    for prob, next_state, reward, _ in environment.P[state][action]
                )

                state_value_function[state] = expected_return
                value_change = max(value_change, abs(prev_value - state_value_function[state]))
            
            # Check for convergence
            if value_change < convergence_threshold:
                break
        
        # Update policy by choosing the best action in each state
        policy_stable = True
        for state in range(environment.observation_space.n):
            old_action = optimal_policy[state]

            # Calculate the value for each action and choose the best one
            action_values = [
                sum(
                    prob * (reward + discount_factor * state_value_function[next_state])
                    for prob, next_state, reward, _ in environment.P[state][action]
                )
                for action in range(environment.action_space.n)
            ]

            new_best_action = np.argmax(action_values)
            optimal_policy[state] = new_best_action

            # Check if policy has changed
            if old_action != new_best_action:
                policy_stable = False

        # Check if policy is stable (no change)
        if policy_stable:
            break

    return optimal_policy, state_value_function


# -------------------------------------------------------------------------
# Function to run Policy Iteration on the environment
# -------------------------------------------------------------------------
# This function runs the policy iteration algorithm for multiple episodes and runs
def run_policy_iteration(env_parameters, environment):
    rewards_all_runs = np.zeros((env_parameters.total_episodes, env_parameters.n_runs))
    steps_all_runs = np.zeros((env_parameters.total_episodes, env_parameters.n_runs))
    episode_numbers = np.arange(env_parameters.total_episodes)
    all_states_recorded = []
    all_actions_recorded = []

    for run in range(params.n_runs):
        # Perform policy iteration to get the optimal policy
        optimal_policy, _ = policy_iteration(env_parameters, environment)

        for episode in tqdm(episode_numbers, desc=f"Policy Iteration Run {run}/{env_parameters.n_runs}", leave=False):
            state = environment.reset(seed=env_parameters.seed)[0]
            step_count = 0
            done = False
            total_episode_reward = 0

            while not done:
                action = optimal_policy[state]
                all_states_recorded.append(state)
                all_actions_recorded.append(action)
                new_state, reward, terminated, truncated, _ = environment.step(action)
                done = terminated or truncated
                total_episode_reward += reward
                step_count += 1
                state = new_state
                
            rewards_all_runs[episode, run] = total_episode_reward
            steps_all_runs[episode, run] = step_count

    return rewards_all_runs, steps_all_runs, episode_numbers, all_states_recorded, all_actions_recorded


# -------------------------------------------------------------------------
# Postprocessing Function: Process and structure the results for analysis
# -------------------------------------------------------------------------
# This function prepares the data for visualization, which is an important step for analyzing the performance of RL algorithms
def postprocess(episodes, params, rewards, steps, map_size):
    """Convert the results of the simulation in dataframes."""
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, reps=params.n_runs),
            "Rewards": rewards.flatten(),
            "Steps": steps.flatten(),
        }
    )
    res["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")
    res["map_size"] = np.repeat(f"{map_size}x{map_size}", res.shape[0])

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": steps.mean(axis=1)})
    st["map_size"] = np.repeat(f"{map_size}x{map_size}", st.shape[0])
    return res, st


# -------------------------------------------------------------------------
# Visualization Functions: Plotting steps-rewards analysis
# -------------------------------------------------------------------------
# These functions create visualizations of the rewards/steps over episodes, which is crucial for understanding the behavior and performance of the algorithms
def plot_steps_and_rewards(agg_rewards_df, agg_steps_df, img_name):
    """Plot the steps and rewards for different hyperparameter settings."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    
    # Ensure the DataFrame contains the 'hyperparameters' column
    if 'hyperparameters' not in agg_rewards_df.columns:
        raise ValueError("DataFrame must contain a 'hyperparameters' column.")
    
    # Plotting cumulative rewards
    sns.lineplot(data=agg_rewards_df, x="Episodes", y="cum_rewards", hue="hyperparameters", ax=ax[0])
    ax[0].set_title("Cumulative Rewards")
    ax[0].set_ylabel("Cumulated rewards")

    # Plotting average steps
    sns.lineplot(data=agg_steps_df, x="Episodes", y="Steps", hue="hyperparameters", ax=ax[1])
    ax[1].set_title("Average Steps per Episode")
    ax[1].set_ylabel("Averaged steps number")

    for axi in ax:
        axi.legend(title="Hyperparameters")
    fig.tight_layout()
    img_title = img_name + ".png"
    fig.savefig(params.savefig_folder / img_title, bbox_inches="tight")
    plt.show()


# -------------------------------------------------------------------------
# Hyperparameter Analysis: Iterates over different hyperparameters
# -------------------------------------------------------------------------
# This section tests different combinations of learning rates, gammas, and 
# epsilons to analyze their effects on the learning process

# Run experiments for a specific hyperparameter, keeping others fixed.
def run_hyperparameter_analysis(params, hyperparam_name, hyperparam_range, fixed_params):
    # Aggregated results DataFrames
    agg_results_q_learning = pd.DataFrame()
    agg_results_policy_iter = pd.DataFrame()

    for value in hyperparam_range:
        # Update the params with the new value for the hyperparameter and fixed values for the others
        if hyperparam_name == 'learning_rate':
            params = params._replace(learning_rate=value, gamma=fixed_params['gamma'], epsilon=fixed_params['epsilon'])
        elif hyperparam_name == 'gamma':
            params = params._replace(learning_rate=fixed_params['learning_rate'], gamma=value, epsilon=fixed_params['epsilon'])
        elif hyperparam_name == 'epsilon':
            params = params._replace(learning_rate=fixed_params['learning_rate'], gamma=fixed_params['gamma'], epsilon=value)
        else:
            raise ValueError(f"Invalid hyperparameter name: {hyperparam_name}")

        # Run Q-Learning and Policy Iteration experiments
        rewards, steps, episodes, qtables, all_states, all_actions = run_env()
        rewards_pi, steps_pi, episodes_pi, all_states_pi, all_actions_pi = run_policy_iteration(params, env)
        
        # Add hyperparameter info to the results
        hyperparam_str = f"{hyperparam_name}: {value}"
        res, st = postprocess(episodes, params, rewards, steps, params.map_size)
        res['hyperparameters'] = hyperparam_str
        st['hyperparameters'] = hyperparam_str

        res_pi, st_pi = postprocess(episodes_pi, params, rewards_pi, steps_pi, params.map_size)
        res_pi['hyperparameters'] = hyperparam_str
        st_pi['hyperparameters'] = hyperparam_str
        
        # Aggregate the results
        agg_results_q_learning = pd.concat([agg_results_q_learning, res])
        agg_results_policy_iter = pd.concat([agg_results_policy_iter, res_pi])
    
    # Plot results for Q-Learning and Policy Iteration

    # Separate rewards and steps data for Q-Learning
    agg_rewards_q_learning = agg_results_q_learning[['Episodes', 'cum_rewards', 'hyperparameters']]
    agg_steps_q_learning = agg_results_q_learning[['Episodes', 'Steps', 'hyperparameters']]

    # Separate rewards and steps data for Policy Iteration
    agg_rewards_policy_iter = agg_results_policy_iter[['Episodes', 'cum_rewards', 'hyperparameters']]
    agg_steps_policy_iter = agg_results_policy_iter[['Episodes', 'Steps', 'hyperparameters']]

    # Plot results for Q-Learning
    plot_steps_and_rewards(agg_rewards_q_learning, agg_steps_q_learning, f"part1_hyperparameter_analysis_{hyperparam_name}_q_learning")

    # Plot results for Policy Iteration
    plot_steps_and_rewards(agg_rewards_policy_iter, agg_steps_policy_iter, f"part1_hyperparameter_analysis_{hyperparam_name}_policy_iteration")


    # Print results for analysis
    print(f"------ Analysis for varying {hyperparam_name}: ------")
    print("Q-Learning:")
    print(f"\tAverage Rewards: {agg_results_q_learning.groupby('hyperparameters')['Rewards'].mean()}")
    print(f"\tAverage Cumulated Rewards: {agg_results_q_learning.groupby('hyperparameters')['cum_rewards'].mean()}")
    print(f"\tAverage Steps: {agg_results_q_learning.groupby('hyperparameters')['Steps'].mean()}")
    print("Policy Iteration:")
    print(f"\tAverage Rewards: {agg_results_policy_iter.groupby('hyperparameters')['Rewards'].mean()}")
    print(f"\tAverage Cumulated Rewards: {agg_results_policy_iter.groupby('hyperparameters')['cum_rewards'].mean()}")
    print(f"\tAverage Steps: {agg_results_policy_iter.groupby('hyperparameters')['Steps'].mean()}")



# Define ranges for hyperparameters that will vary
learning_rates = [0.1, 0.01, 0.001]
gammas = [0.8, 0.9, 0.99]
epsilons = [0.01, 0.1, 0.5]

# Define fixed hyperparameters
fixed_hyperparams = {'learning_rate': 0.1, 'gamma': 0.8, 'epsilon': 0.01}

# Varying learning rate
run_hyperparameter_analysis(params, 'learning_rate', learning_rates, fixed_hyperparams)

# Varying gamma
run_hyperparameter_analysis(params, 'gamma', gammas, fixed_hyperparams)

# Varying epsilon
run_hyperparameter_analysis(params, 'epsilon', epsilons, fixed_hyperparams)

env.close()