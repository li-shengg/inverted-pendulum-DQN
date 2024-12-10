####################################################
# Imports
####################################################
import numpy as np
import gym
from .memory import Memory
from .net import Net

import tensorflow as tf
from tensorflow.keras.optimizers import Adam


class Agent:
    # this constructor initializes the environment, model, memory, and other variables required for the agent
    def __init__(self, env_string, num_actions, state_size, batch_size=32, learning_rate=0.01, gamma=0.98, epsilon=1.0, epsilon_decay=0.98, epsilon_min=0.01, tau=0.01, memory_capacity=10000):
        self.env_string = env_string
        self.env = gym.make(env_string)
        self.env.reset()
        
        self.state_size = state_size
        self.num_actions = num_actions
        self.action_scope = [i * 4.0 / (num_actions - 1) - 2.0 for i in range(num_actions)] # Adjusted to match action scaling in dueling DQN
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.tau = tau
        self.memory = Memory(memory_capacity)
        self.best_total_reward = float('-inf')
        
        self.model = Net(state_size, num_actions)
        self.target_model = Net(state_size, num_actions)
        self.optimizer = Adam(learning_rate)
        self.loss_fn = tf.losses.MeanSquaredError()

    '''
    Implements the epsilon-greedy policy. With probability epsilon, a random action is selected, otherwise the action chosen will
    be the one with the highest Q value. The epsilon value is decayed over time to reduce the exploration as the agent learns.
    It also converts the state to a tensor before passing through to the model.
    '''
    def select_action(self, state):
        # if random number is less than epsilon, return random action 
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions), None
        # convert to tensor
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.model(state)
        # else, return action with highest Q value
        return np.argmax(q_values.numpy()), None

    # Store experience tuple into the memory buffer, function is above
    def store_transition(self, state, action, reward, next_state):
        self.memory.update((state, action, reward, next_state, False))

    '''
    Learn function performs single step training on a batch of experiences sampled from the memory buffer.
    '''
    def learn(self):
        if len(self.memory.memory) < self.batch_size:
            return

        # sample a batch of transitions from the memory
        transitions = self.memory.sample(self.batch_size)
        # extract the states, actions, rewards, next states, from the batch
        state_batch, action_batch, reward_batch, next_state_batch, _ = zip(*transitions)
        
        # convert the s, a, r, s_' to tensors
        state_batch = tf.convert_to_tensor(state_batch, dtype=tf.float32)
        action_batch = tf.convert_to_tensor(action_batch, dtype=tf.int32)
        reward_batch = tf.convert_to_tensor(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(next_state_batch, dtype=tf.float32)

        '''
        Tensorflow GradientTape is an API for automatic differentiation. It records operations for automatic differentiation.
        This function will calculate the predicted Q-values from the current state and actions.
        '''
        with tf.GradientTape() as tape:
            '''
            Forward pass through the DQN model to get the Q-values for all actions given the current batch of states.
            q-values contains the predicted q-values for each action in each state of the batch
            '''
            q_values = self.model(state_batch)
            
            '''
            'action_indices' calculates the indices of the actions taken in the Q-value matrix. 
            Since q_values contains Q-values for all actions, this step is necessary to select only the Q-values corresponding to the actions that were actually taken.
            '''
            action_indices = tf.range(self.batch_size) * self.num_actions + action_batch
            
            '''
            Reshapes the Q-value matrix to a single vector, then selects the Q-values for the actions taken using the 'action_indices' calculated above.
            '''
            predicted_q = tf.gather(tf.reshape(q_values, [-1]), action_indices)
            
            '''
            Forward pass through the target DQN model to get Q-values for all actions given the next states. 
            '''
            next_q_values = self.target_model(next_state_batch)
            
            # Finds the maximum Q-value among all actions for each next state, which represents the best possible future reward achievable from the next state.
            max_next_q = tf.reduce_max(next_q_values, axis=1)
            
            '''
            Calculates the target Q-values using the immediate reward received (reward_batch) and the 
            discounted maximum future reward (self.gamma * max_next_q). This forms the update target for the Q-value of the action taken.
            '''
            target_q = reward_batch + self.gamma * max_next_q
            
            # Calculates the loss between the predicted Q-values and the target Q-values, basically MSE loss
            loss = self.loss_fn(target_q, predicted_q)
        
        # Calculate the gradients of the loss with respect to the model parameters
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        self.update_epsilon()

    # update the epsilon value
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)
    
    # update the model
    def update_target_model(self):
        main_weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = self.tau * main_weights[i] + (1 - self.tau) * target_weights[i]
        self.target_model.set_weights(target_weights)

    # checks if current model is the best model based on total reward
    def is_best_model(self, total_reward):
        return total_reward > self.best_total_reward

    # updates the best model with the current model
    def update_best_model(self, total_reward):
        self.best_total_reward = total_reward