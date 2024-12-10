from src.train import main
from src.helper import plot_rewards, compute_moving_average_and_plot

# Run the model
rewards = main()


# Plot the rewards
save_dir = 'visualizations/images'
plot_rewards(rewards, save_dir)
compute_moving_average_and_plot(rewards, save_dir)