# AI to play doom

# Importing the Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing OpenAi packages and Doom
import gym
# SkpipWrapper is basically responsible for all the environments 
from gym.wrappers import SkipWrapper
# ToDiscrete contains the actions like move left, right, straight etc
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Import other python files
# experience_replay handles the experience rplay concept for our agent
# image_preprocessing is used to preprocess the images and converting them into np arrays 
import experience_replay, image_preprocessing
 

# Part 1: Building the AI

# Making the Brain of the AI

# Inherits the nn.Module class
class CNN(nn.Module):
    
    '''
        number_actions: gets the number of actions from the doom environment
        and makes it easier for us to switch environments as it gets the 
        actions the agent is allowed to perform, so we wont have to perform 
        in case we change the env. 
        
        Architecture of the CNN: we start with 3 convolutions layers(give us high 
        level of feature detection) fc1 and fc2 are the full connections connecting 
        the 3rd convolution layer to the hidden and hidden to output layer respectively. 
        
    '''
    def __init__(self, number_actions):
        # Invoke parent class constructor
        super(CNN, self).__init__()
        self.number_actions = number_actions
        
        # Convolution Layers
        # in channel=1 if we deal with b/w images and =3 if color images
        # AI can recognize who to shoot based on shape so color is not needed.
        # out_channels = no. of features you want to detect, i.e output of the conv layer
        # common practise is to use 32 feature detectors
        # This means convolution was applied to the input image to get 32 processed images
        # kernel_size is the dimensions of the square that will go through 
        # the original image (feature detector/filter). common practise is to 
        # use 2X2, 3X3 or 5X5
        # we slowly reduce the kernel size in each layer, as we start by detecting 
        # the big features and the deeper we get the more accurate features.
         
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        
        # as we need to detetct more features as we go deeper, we increase number of 
        # output images as each convuled image will detect only one feature.
        
        self.convolution4 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        
        # Hidden Layer
        # in_features = no. of pixels we get after flattening the 64 output images
        # we will compute it using the count_neurons(input_dim)  
        # out_features is a number of our choosing. general practise is to choose 40
        # we may increase it iff training is not very slow
        
        self.fc1 = nn.Linear(in_features=self.count_neurons((1,80,80)), out_features =60)
        
        # Hidden Layer 2
        self.fc2 = nn.Linear(in_features=60, out_features =60)
        
        # Hidden Layer 3
        self.fc3 = nn.Linear(in_features=60, out_features =60)
        
        # Hidden Layer 4
        self.fc4 = nn.Linear(in_features=60, out_features =60)
        
        # Hidden Layer 5
        self.fc5 = nn.Linear(in_features=60, out_features =60)
        
        # Hidden Layer 6
        self.fc6 = nn.Linear(in_features=60, out_features =60)
        
        # Hidden Layer 7
        self.fc7 = nn.Linear(in_features=60, out_features =60)
        
        # Hidden Layer 8
        self.fc8 = nn.Linear(in_features=60, out_features =60)
        
        # Hidden Layer 9
        self.fc9 = nn.Linear(in_features=60, out_features =60)
        
        # Output Layer
        # out_features in this case is the final o/p which is the actions
    
        self.fc10 = nn.Linear(in_features=60, out_features=number_actions)
        
        
    '''
        image_dim = (1,80,80). 1-> b/w image, 80X80 will be the dimension
        *image_dim passes the tuple as a list
    '''    
    def count_neurons(self, image_dim):
        
        # creating a fake image, used to calculate the number_neurons
        x =Variable(torch.rand(1, *image_dim))
        
        # Now we prpagate the fake image 'x' through the CNN upto the 
        # 3rd convolution layer to get the number_neurons in the flatenning layer
        # Applying the convolution1, then max-pooling(kernel_size = 3, stride = 2)
        # Then apply Relu
        
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        
        # Now, we flatten the last layer which will give us the number_neurons
        # x is a pytorch variable and we can use its properties to achieve our goal
        # size() gives all the pixels of all the channels and puts them in a
        # huge vector which will be the input to fc1
        
        return x.data.view(1, -1).size(1)
        
    '''
        function to propagate the signals in all layers of the CNN
        Logic is similar to that of the count neurons as we propagate the data
        through all the layers but instead of the fake image we use the input 
        doom images
        
        x: in this case is the input images at first, it will then be updated as 
        the  signal is propagated through the layers 
    '''
    def forward(self, x):
        
        # Propagating through the convolution layers
        
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        
        # Flattening. These flattening tricks can be found in the pytorch tutorials
        
        x = x.view(x.size(0), -1)

        # Propagating x through the fully connected layer and using relu to 
        # break the linearity
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.fc8(x)
        x = self.fc9(x)
        x = self.fc10(x)
        
        # return the output neurons with the Q-values
        
        return x
    
# Making the Body

class SoftmaxBody(nn.Module):
    
    '''
        T: temperature
        We can use T to configure the exploration. 
        Higher T -> lower exploration and vice versa.
        NOTE: After the Agent has been trained increase the Temperature 
        for best performance
    '''
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T
        
    '''
        funcn propagate the output signal from the brain of the AI to the body
        outputs: the outpt signals from the brain
    '''
    def forward(self, outputs):
        
        # distribution of proabilities for all the Q-values(one for each action)    
        # which depend on the input image 
        # We multiply the Temperature parameter so that the higher probabilities
        # get selected.  
        
        probs = F.softmax(outputs * self.T)
        actions = probs.multinomial()
        return actions
    
# Making the AI

class AI:
    
    '''
       brain is the neural net
       body is the softmax
    '''
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body 
    
    '''
        Takes inputs and propagates through the brain and body of the AI
        inputs : input images
    '''    
    def __call__(self, inputs):
        
        # convert the inputs into numpy arrays and  then convert the array into
        # torcch tensors
        # Then convert the tensor into a torch Variable having both a tensor 
        # and a gradient
        
        input = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        
        output = self.brain(input) 
        actions = self.body(output)
        
        # actions are in torch format, convert to np array and return
        return actions.data.numpy()
         
    
# Part 2 - Training the AI using Deep Q-Learning

# getting the Doom environment
# we can change the environment by changing the argument to gym.make
# set image width and height to 80X80 as we set our input format to 80X80
# set grayscale to true as we take black and white input
        
doom_env = image_preprocessing.PreprocessImage(SkipWrapper(4)(ToDiscrete("minimal")(gym.make("ppaquette/DoomCorridor-v0"))), width = 80, height = 80, grayscale = True)
doom_env = gym.wrappers.Monitor(doom_env, "videos", force = True)

# Gets the number of actions allowed in that environment
number_actions = doom_env.action_space.n

# Building an AI

# Creating a Brain
cnn = CNN(number_actions)

# Creating a Body
softmax_body = SoftmaxBody(T = 1.0)

# Creating an AI
ai = AI(brain = cnn, body = softmax_body)

# Setting up Experience Replay with n-step eligibility trace
n_steps = experience_replay.NStepProgress(env = doom_env, ai = ai, n_step = 10)
memory = experience_replay.ReplayMemory(n_steps = n_steps, capacity = 10000)

# Implementing Eligibility Trace

'''
    Implementing Asynchronus n-step Q-learning algorithm (with 1 agent so not really asynchronus)
    and instead of epsilon-greedy we are using softmax. 
    This algorithm is also known as sarsa algorithm.
    
    Takes batch as i/p and returns the inputs and targets 
    inputs and targets will be fed in batches
'''
def eligibility_trace(batch):
    gamma = 0.99
    inputs = []
    targets = []
    
    for series in batch:
        # taking the first and last element of the batch and make torch Variable 
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        output = cnn(input)
        # if last transition of the series is done reward = 0.0 else reward = max(Q-values)
        # output[1] will give the Q-value and we take its data and finally find the max
        cumul_reward = 0.0 if series[-1].done else output[1].data.max()
        
        # loop in reverse upto the elemnt before the last element and compute the cumulative reward
        for step in reversed(series[:-1]):
            cumul_reward = step.reward + gamma * cumul_reward
            
        state = series[0].state
        target = output[0].data
        target[series[0].action] = cumul_reward
        
        inputs.append(state)
        targets.append(target)
    
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)

# Making moving avg on 100 steps
# size will be = 100
class MA:
    
    def __init__(self, size):
        self.list_of_rewards = []
        self.size = size
        
    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards += rewards
        else:
            self.list_of_rewards.append(rewards)
            
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]
            
    def average(self):
        return np.mean(self.list_of_rewards)
    
# Creating obj of MA class with size 100
ma = MA(100)

# Training the AI
loss = nn.MSELoss() # mean squared error
optimizer = optim.Adam(cnn.parameters(), lr = 0.001)
nb_epocs = 100

for epoc in range(1, nb_epocs + 1):
    # 200 runs in each epoc
    memory.run_steps(200)
    
    # 128 -> batch size
    # every 128 steps AI memory will give us a batch of size 128, which contains
    # the last 128 steps taken by the AI
    for batch in memory.sample_batch(128):
        inputs, targets = eligibility_trace(batch)
        inputs, targets = Variable(inputs), Variable(targets)
        
        # making the predictions and calculating the loss
        predictions = cnn(inputs)
        loss_error = loss(predictions, targets)
        
        # initializing the optimizer
        optimizer.zero_grad()
        # back-propagating the loss error
        loss_error.backward()
        # performing stocastic gradient descent
        optimizer.step()
    
    # calculating average reward        
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps)
    avg_reward = ma.average()
    
    print('Epoc: %s, Average Reward: %s' % (str(epoc), str(avg_reward)))
    
    # Change the reward target w.r.t the environment.
    # Else it will run for 100 iterations as we set nb_epocs = 100
    if avg_reward >= 1500:
        print("Congratulations your AI wins")
        break # stop training when AI wins

# Closing the Doom environment
doom_env.close()
