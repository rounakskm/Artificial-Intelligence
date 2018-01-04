# Artificial-Intelligence
# Reinforcement Learning

The objective is to build an AI for simulating a self driving car

The AI is created using the Deep-Q-Learning algorithm.
The file ai_self.py contains the commented code, which has been explained step by step.
The file map_commented.py contains the commented code for the environment in which our AI will be tested.

In order to see the AI in action simply run the map.py or map_commented.py file

The Car is represented by a rectangle and the three circles around its front are the sensors. 
It's goal is to go from the top left to the bottom right corner.
We can draw roads or other ways with sand by clicking on the field.

The AI will learn and drive around properly in the roads we create. 

We can save the learned AI using the save button and load a previously trained AI bu using the load button.
The first time we dont have the option for loading.


The hypothetical sensors of the car are used to check if the car runs into any sand.
If it does then the algorithm penalizes it and the car learns from this.