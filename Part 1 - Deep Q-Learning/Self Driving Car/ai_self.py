 # AI for Self-Driving Car

# Importing the Libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating arcitecture of Neural Net
# 5 input neurons in input layer, 30 hidden neurons in 1 hidden layer, 
# 3 output neurons in the output layer

# Class  network must inherit nn.Module class
class Network(nn.Module):
    
    '''
    input_size = number of input neurons
    will be 5 in this case as input vector has 5 dimensions
    3 signals, +orientation, -orientation.
    
    nb_action = number of output neurons
    will be 3 in this case as output will be go left,
    straight or right.
    
    Note: we can add more hidden layers by making more full connections
    '''
    def __init__(self, input_size, nb_action):
        # calling the constructor of the nn module
        super(Network, self).__init__()
        
        self.input_size = input_size
        self.nb_action = nb_action
        
        # makes first full connection between the input layer
        # and first hidden layer
        
        # full connection/fc -> all nodes of input layer are connected 
        # to all nodes in the hidden layer
        
        # Parameters -> Input nodes, output nodes(no. of nodes in hidden layer)
        # and bias(= True). After parameter tuning output nodes value was set = 30 
        self.fc1 = nn.Linear(input_size, 30)
        
        # fc2 -> connects hidden to output layer
        self.fc2 = nn.Linear(30, nb_action)
        
    '''
    state = input of our neural network
    function will return the Q values for the input state provided
    
    Note: will need to add more variables like x if we add more layers
    '''    
    def forward(self, state):
        # x represents the hidden neurons
        # fc1 takes input states to go from input neurons to hidden neurons
        x = F.relu(self.fc1(state)) #Gives us activated hidden neurons
        
        # Output neurons, these will give us Q-values
        # input to fc2 is the activated hidden neurons 
        q_values = self.fc2(x)
        
        return q_values
    
# Experience Replay
# We put the last 100 transitions in memory

class ReplayMemory(object):
    '''
    capacity is the number of last transitions we store
    in the memory[]. In this case we consider 100,000
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        
        # list containing last 100 transitions/events
        self.memory=[]
        
    '''
    event is the latest transition that occered and needs to be saved
    event is a tuple of (last-state(st), new-state(st+1), 
    last-action(at), last-reward(rt))
    Appends last transition/event to the memory list
    and makes sure that the list is as big as the capacity 
    '''
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
            
    '''
    batch_size = number of samples considered
    takes some random samples from the memory of events
    note: random.sample() is an inbuilt funcn, dont confuse with funcn name 
    '''
    def sample(self, batch_size):
        
        # random.sample(self.memory, batch_size) this function  
        # takes some random samples from the list provided that 
        # have a fixed size of batch_size
        
        # l =[(1,2),(3,4)] then zip(*l) = [(1,3), (2,4)]
        # This will give us a list [(st1,st2), (st1+1,st2+1), (at1,at2), (rt1,rt2)]
        # if we take batch_size =2
        samples = zip(*random.sample(self.memory, batch_size))
        
        # We cant return samples directly, so we map them onto torch variables
        # lambda funcn will take the samples, concatinate them w.r.t 
        # the first dimension, then converts the tensors into torch variables 
        # that contain both a tensor and a gradient.
        # This is done to make it easier to uodate the weights while using stocastic GD.
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
# Implementing Deep Q-Learning

class Dqn():
    
    '''
    gamma parameter is the discount factor in the Q-Learning algo
    reward_window is the sliding window of evolving mean of last 100 rewards
    '''
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = [] # will append mean of rewards over time
        self.model = Network(input_size, nb_action) # Creating our NN
        self.memory = ReplayMemory(100000) # Object of ReplayMemory class, capacity as param
        
        # Create an optimizer and give it the parameters of the NN to optimize
        # lr is the Learning Rate 
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        
        # In this case last_state is a vector of 5 dimensions & inp_size is 5
        # For use with pytorch last_state needs to be a torch tensor
        # The network will only accept a batch of input observations
        # We need to add the batch size as well, it is also called the fake dimension
        # 0 as the fake dim goes in the beginning of the input 
        self.last_state = torch.Tensor(input_size).unsqueeze(0) 

        self.last_action = 0 #it corresponds to the action2rotation present in map.py
        self.last_reward = 0 # reward is a float between 0 & 1
        
    def select_action(self, state):
        # pass the NN to the softmax function, as softmax operates on the final 
        # Q-valies which are obtained as an output of our NN
        # states is the input to the NN, this needs to be a torch variable like last_state
        # So convert the state into torch Vriable
        # To specify that we dont want the gradient associated to input state 
        # in the graph add volatile param as true. This trick saves memory
        
        # temperature param allows us to modulate how the NN will be sure 
        # of which action it will decide to play
        # lower temperature param -> AI not very sure to act
        # higher temperature param -> AI very sure to act
        # can be added my muliplying the output by the temp parameter
        # here T = 7
        # ex softmax([1,2,3]) => [0.04,0.11,0.85]. Applying T=3
        # softmax([1,2,3]*3) => [0, 0.02, 0.98] higher probabilities, more surity
        # Increasing the temperature leads to more surity while taking decisions
        probs = F.softmax(self.model(Variable(state, volatile = True))*100)
        
        #now we m=need a random draw from this distribution
        action = probs.multinomial() # returns pytorch variable with fake dim(which is the batch_size)
        return action.data[0,0] # 0,0 index holds the action to be taken
 
    '''
    our inputs are now in batches as created by sample function
    '''
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        
        # .gather(1,batch_action) returns the action which was chosen
        # unsqueez is used as batch state has the fake dim but batch_action doesnt
        # index will be 1 as 0 corrosponds to the fake dim of state.
        outputs = self.model(batch_state).gather(1,batch_action.unsqueeze(1))
        
        # Now that we are out of the NN we dont need batches anymore, so we 
        # squeez it back and remove the fake dimension of the action.
        # Index is 1 as we need to kill the fake dim of the batch_action
        outputs.squeeze(1)
        
        # detach is used to separate the Q values as the output is all q values of all the next states
        # then we find the max of those Q-values. 
        # index 1 is specified as we are taking the max Q-values w.r.t action, 
        # and the index of action is 1. 
        # We also have to specify that we are taking the Q-values of of the next_state
        # and next_state corresponds to 0 so we need to add [0]
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        
        # calculate target according to Q-learning algorithm
        target = self.gamma*next_outputs + batch_reward
        
        # calculate temporal-difference, Huber loss function is used
        # represented by smooth_l1_loss. Best Loss funcn for Deep Q-Learning
        # arguments needed are predictions which in this case is outputs and target
        td_loss = F.smooth_l1_loss(outputs, target)
        
        # Loss has been calculated, now we need to backpropagate the error
        # we can do this using the optimizer we created earlier
        # When working with pytorch we should re-initialize the optimizer 
        # at each iteration of the loop of stocastic gradient descent
        # This can be done using zero_grad
        self.optimizer.zero_grad()
        #retain variables = true is to free some memory
        td_loss.backward(retain_variables = True)

        # Finally, update the values of the weights
        self.optimizer.step()
        
        
    '''
    updates when AI discovers new state
    the siganl parameter is computed in the map using the sensors
    it is the state in the form of a list. We will need to convert it 
    into a torch variable before we can use it
    
    Returns which action was taken
    '''
    def update(self, reward, new_signal):
        
        # convert and add a fkane dim at the beginning
        new_state = torch.Tensor(new_signal).unsqueeze(0)
        
        # add the latest trasition to the replay memory
        # we created the push function above, it takes one argument the event
        # we create the event tuple and feed it to the push function.
        # all the items in the tuple need to be tensors.
        # torch tensors take lists is input, so convert when required
        self.memory.push((self.last_state, new_state, 
                          torch.LongTensor([int(self.last_action)]), 
                          torch.Tensor([self.last_reward])))
        # play the new action after reaching the state
        action = self.select_action(new_state)
        
        # memory in Dqn has an attribute memory
        # if 100+ samples exist, then learning starts and 100 samples are taken
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        
        # Update last action, last state and last_reward
        self.last_action = action 
        self.last_state = new_state
        self.last_reward = reward
        
        # Update reward window, it is used to see how training is going
        # if most of the rewards are positive, then trining is going well
        self.reward_window.append(reward)
        
        # reward window has fixed size
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
            
        # return the action selected    
        return action

    # Functions to keep score, save the ai, reset field etc 

    def score(self):
        # add 1 to avoid denominator being 0
        return sum(self.reward_window)/(len(self.reward_window)+1.)

    def save(self):
        # state_dict() -> saves the parameters of the model
        # last_brain.pth -> Name of file where the last brain is saved
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth') 

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print('Loading Checkpoint.......')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done...")
        else:
            print("No checkpoint found !!!")
            