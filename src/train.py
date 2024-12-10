####################################################
# Imports
####################################################
from .agent import Agent
from .helper import save_model_weights
import gym

import os
import imageio

# Main function to run the training
def main():
    env_string = 'Pendulum-v0'
    num_actions = 5
    state_size = gym.make(env_string).observation_space.shape[0]
    agent = Agent(env_string, num_actions, state_size)

    episodes = 25
    rewards = []  # Initialize rewards list to store total rewards per episode
    for ep in range(episodes):
        state = agent.env.reset()
        done = False
        total_reward = 0

        # Capture frames for GIF
        frames = []
        
        while not done:
            
            # Render the environment and capture frames
            frames.append(agent.env.render(mode='rgb_array'))
            
            action_index, _ = agent.select_action(state)
            action = [agent.action_scope[action_index]] 
            next_state, reward, done, _ = agent.env.step(action)
            agent.store_transition(state, action_index, reward, next_state)
            state = next_state
            total_reward += reward
            
            agent.learn()
            agent.update_target_model()

        rewards.append(total_reward)  # Append the total reward of the episode to the rewards list
        
        print(f"Episode: {ep+1}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        if agent.is_best_model(total_reward):
            # Update and save the best model weights
            agent.update_best_model(total_reward)
            model_dir = '../weights'
            save_model_weights(agent, model_dir)
            print("Best model weights saved.")

        # Directory where you want to save the files
        save_dir = '../visualizations/gif/'

        # Create the directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

                
        # Save frames as GIF
        gif_filename = os.path.join(save_dir, f'episode_{ep+1}.gif')
        imageio.mimsave(gif_filename, frames, duration=0.00333, loop=0) 

    agent.env.close()
    return rewards