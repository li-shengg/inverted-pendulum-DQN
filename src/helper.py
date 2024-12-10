# Imports
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_rewards(rewards, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.legend()
    plt.grid(True)

    plot_filename = 'reward_plot.png'  # Choose the desired file name
    save_path = os.path.join(save_dir, plot_filename) 

    plt.savefig(save_path)

def compute_moving_average_and_plot(results, save_dir, window_size = 5):
    """Compute the moving average of a list of data and plot it."""
    # Compute the moving average
    moving_avg = np.convolve(results, np.ones(window_size) / window_size, mode='valid')
    
    # Plot the data and moving average
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(window_size-1, len(results)), moving_avg, color='orange', label='Moving Average')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Moving Average')
    plt.legend()
    plt.grid(True)

    plot_filename = 'reward_moving_average_plot.png'  # Choose the desired file name
    save_path = os.path.join(save_dir, plot_filename) 

    plt.savefig(save_path)



def save_model_weights(agent, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    weights_path = os.path.join(model_dir, 'tensorflow_dqn_weights.weights.h5')
    agent.model.save_weights(weights_path)