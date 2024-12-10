####################################################
# Imports
####################################################
import random
from collections import deque, namedtuple

class Memory:
    # Constructor, the capacity is the maximum size of the memory. Once capacity is reached, the old memories are removed.  
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    '''
    Adds transition to the memory. The transition is a tuple of (state, action, reward, next_state).
    If the maximum capacity of memory is reached, the old memories are overrided.
    '''
    def update(self, transition):
        self.memory.append(transition)
    
    '''
    Retrieve a random sample from memory, batch size indicates the number of samples to be retrieved.
    '''
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)