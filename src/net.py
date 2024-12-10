# This is the Deep Neural Network for the Model


####################################################
# Imports
####################################################
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input


class Net(tf.keras.Model):
    # constructor initializes the layers
    def __init__(self, input_size, num_actions):
        super(Net, self).__init__()
        self.dense1 = Dense(128, activation='relu', input_shape=(input_size,))
        self.dense2 = Dense(128, activation='relu')
        self.output_layer = Dense(num_actions, activation='linear')
    
    '''
    This function is the forward pass of the model. It takes the state as input and returns the Q values for each action.
    'x' is the input to the model, which is the state from the environment. 'x' is then passed through the 2 dense
    layers and the output layer to get the predicted values for each action. We use a linear activation function 'relu' to
    predict a wide range of values, to estimate action values in the RL model
    '''
    def call(self, x):
        # first dense layer
        x = self.dense1(x)
        # second dense layer
        x = self.dense2(x)
        return self.output_layer(x)