# lunar_lander_per

Implementation of the Prioritized Experience Replay for the Deep Q-Network algorithm, following the publication from Tom Schaul, John Quan, Ioannis Antonoglou and David Silver : https://arxiv.org/pdf/1511.05952.pdf

Dependencies: 
- Python 3.6.4
- Gym
- Torch
- Numpy
- Matplotlib

This implementation refers to the Rank-based prioritization mentionned in the paper rather than the Proportional prioritization. This means that the priorities associated with each environment state are stored in a conventional container (Here a dictionnary) rather than in a sum tree.

This implementation with PER only can solve the lunar-lander environment in about 1200 episodes. It could be further improved by adding a the dueling Q-Network implementation. The computation of the weights necessary for the dueling Q-Network in combination with PER are already implemented here. 

To run the training and observed the trained agent : `python launch.py`

To be able to visualize the agents from WSL (Windows Sub-sysytem for Linux) or WSL 2:
- in your bash terminal, you'll need to install `sudo apt install ubuntu-desktop mesa-utils` 
- open a XLaunch server in Windows. Use default parameters except for `Extra Settings`, un-tick Native OpenGL and tick Disable access control.
- export the WSL display to Windows: `export DISPLAY=localhost:0`
