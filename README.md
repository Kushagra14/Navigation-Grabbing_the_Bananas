## GRAB THE BANANAS

In this project Reinforcement Learning agent navigate (and collect bananas!) in a large, square world. Below is the snapshot of the game:

![GRAB THE BANANAS](images/banana.gif)


### Description
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

    0 - move forward.
    1 - move backward.
    2 - turn left.
    3 - turn right.

The task is episodic and termination happens if the agent achieves average score of 13 over 100 consecutive episodes

### Dependencies
1. Python 3
2. Pytorch
3. Jupyter-notebook
4. Untiy Environment

### Getting Started
1. Clone this repository on your local machine
2. Open the project using jupyter-notebook
3. [Shift + Ent] to execute every cell. This particular code will start learning neural network and will run one episode in backgorund
4. You can also load the weights provided in this repositroy using code below:
   ```device = torch.device('cpu')
      model = TheModelClass(*args, **kwargs)
      model.load_state_dict(torch.load(PATH, map_location=device))

    ```
    [Using pytorch to load weights in your algorithm](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
    

