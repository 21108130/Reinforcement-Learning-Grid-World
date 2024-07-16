# Reinforcement Learning Grid World

This project simulates a grid world environment where a reinforcement learning agent learns to navigate from a start position to a goal while avoiding obstacles. The agent uses Q-learning to update its action-value function and improve its policy over time.

## Features

- Grid world environment with customizable grid size and maximum steps.
- Q-learning agent with epsilon-greedy policy for exploration and exploitation.
- Visualization using Tkinter for GUI and Pygame for sound effects.
- Obstacles that penalize the agent upon collision.
- Goal state with a reward upon reaching it, with additional penalties for reaching it too early.

## Prerequisites

- Python 3.x
- Visual Studio (recommended for development)
- Required Python libraries:
  - NumPy
  - Tkinter (standard library for GUI)
  - Pillow (Python Imaging Library) or equivalent
  - Pygame (for sound effects)

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/reinforcement-learning-grid-world.git
   cd reinforcement-learning-grid-world
   ```

2. Open the project in Visual Studio:
   - Launch Visual Studio.
   - Navigate to `File > Open > Project/Solution` and select the `reinforcement-learning-grid-world.sln` file.

3. Set up the Python environment:
   - Ensure Python interpreter is correctly set up in Visual Studio.
   - Install required Python packages: `numpy`, `pillow`, `pygame`.

4. Run the simulation:
   - Open `main.py` in Visual Studio.
   - Adjust parameters such as grid size and number of training episodes.
   - Start debugging (`F5`) or run without debugging (`Ctrl+F5`).

## Usage

- The GUI will display the grid world environment, the agent's movement, and rewards obtained.
- Customize simulation parameters directly in the `main.py` script.
- Explore different configurations to observe how the agent learns and navigates the environment.

## Structure

- `main.py`: Entry point of the program, initializes the environment and agent, and starts the visualization.
- `RLVisualizer.py`: Contains the `RLVisualizer` class for rendering the grid world and handling GUI interactions.
- `gridworld.py`: Defines the `GridWorld` class representing the grid environment and its dynamics.
- `qlearning.py`: Implements the `QLearningAgent` class for the reinforcement learning agent using Q-learning.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or create a pull request.

