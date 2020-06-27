# lunar_lander_per

Implementation of the Prioritized Experience Replay for the Deep Q-Network algorithm, following the publication from Tom Schaul, John Quan, Ioannis Antonoglou and David Silver : https://arxiv.org/pdf/1511.05952.pdf

Dependencies: 
- Python 3.6.4
- Gym
- Torch
- Numpy
- Matplotlib

This implementation refers to the Rank-based prioritization mentionned in the paper rather than the Proportional prioritization. This means that the priorities associated with each environment state are stored in a conventional container (Here a dictionnary) rather than in a sum tree.

This implementation with PER only can solve the lunar-lander environment in about 1200 episodes. It could be further improved by adding a the dueling Q-Network implementation. The computation of the weights necessary for the dueling Q-Network are already implemented here. 

To run the training and observed the trained agent : `python launch.py`
