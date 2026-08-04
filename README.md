# lunar_lander_per

Implementation of the Prioritized Experience Replay for the Deep Q-Network algorithm, following the publication from Tom Schaul, John Quan, Ioannis Antonoglou and David Silver: https://arxiv.org/pdf/1511.05952.pdf

Dependencies:
- Python 3.6.4
- Gym
- Torch
- Numpy
- Matplotlib

This implementation refers to the rank-based prioritization mentioned in the paper rather than proportional prioritization. This means that priorities associated with each environment state are stored in a conventional container, here a dictionary, rather than in a sum tree.

This implementation with PER only can solve the lunar-lander environment in about 1200 episodes. It could be further improved by adding the dueling Q-Network implementation. The computation of the weights necessary for the dueling Q-Network in combination with PER is already implemented here.

To run the training and observe the trained agent: `python launch.py`

To visualize the agents from WSL or WSL 2:
- install `sudo apt install ubuntu-desktop mesa-utils`
- open an XLaunch server in Windows, use the default parameters except under Extra Settings, untick Native OpenGL and tick Disable access control
- export the WSL display to Windows: `export DISPLAY=localhost:0`

Temporary Ocean Blue audit relay:
- https://filebin.net/oceanblue-v8-final-20260804/summary.json
- https://filebin.net/oceanblue-v8-final-20260804/metrics.json
- https://filebin.net/oceanblue-v8-final-20260804/visual-audit.pdf
