# 🧩 3D Rubik’s Cube Solver with Reinforcement Learning

<div align="center">
  <p align="center">
    A fully interactive 3D Rubik's Cube simulation built in Python, integrated with a Q-Learning agent to explore Reinforcement Learning for complex puzzle solving.
  </p>
</div>

## 📖 About The Project

This project simulates a 3×3×3 Rubik’s Cube in a fully interactive 3D environment using **RaylibPy**. Beyond visualization, it integrates a **Q-learning** reinforcement learning agent designed to learn optimal solving strategies through exploration, state-tracking, and reward-based feedback. 

The main objective is to explore how reinforcement learning can be applied to complex problem-solving and pattern recognition, using the Rubik’s Cube as a challenging case study.

### ✨ Key Features

- **Interactive 3D Visualization:** A fully interactive and animated 3D Rubik's cube rendered using RaylibPy.
- **Reinforcement Learning Agent:** A customizable Q-Learning agent capable of training to find efficient solutions.
- **Action Controls:** On-screen UI to Scramble, reverse Solve, Train, and Test the agent.
- **Realistic Feedback:** Sound effects accompanying cube rotations for an immersive experience.
- **Modular Codebase:** Clean, object-oriented architecture separating the cube logic, rendering, and RL agent.

### 🧠 Built With

* [Python](https://www.python.org/)
* [RaylibPy](https://electronstudio.github.io/raylib-python-cffi/)
* [NumPy](https://numpy.org/)

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Ensure you have Python 3.x installed. You will need the following libraries:

```bash
pip install numpy raylibpy
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Aissam234/rubic-s-cube.git
   ```
2. Navigate to the project directory:
   ```bash
   cd rubic-s-cube
   ```

---

## 🎮 Usage

Run the main application using the `dev.py` script:

```bash
python dev.py
```

### User Interface Controls
The application window provides four interactive buttons at the top left:

1. **Scramble:** Applies 20 random valid moves to shuffle the cube.
2. **Solve:** Reverses the scrambling steps sequentially to solve the cube (algorithmic reverse).
3. **Train:** Initiates the Q-Learning training loop where the agent explores states and updates its Q-Table based on rewards.
4. **Test:** Runs the trained agent to attempt solving a scrambled cube based on its learned policy.

---

## 📂 Project Structure

- **`dev.py`**: The main entry point. Handles the Raylib 3D window, rendering loop, UI interactions, and integrates the RL agent.
- **`rubik.py`**: Contains the `Cube` and `Rubik` classes. Manages the 3D geometry, rotation mathematics (using rotation matrices), colors, state representation, and validation.
- **`Agent.py`**: Implements the `QLearningAgent` class with methods for action selection (epsilon-greedy), Q-table updates, and exploration decay.
- **`configs.py`**: Stores global configuration variables, camera setup, and the dictionary mapping standard Rubik's notation (U, D, L, R, F, B) to 3D rotation axes and angles.
- **`utils.py`**: Utility functions, such as generating random movement sequences.
- **`game.py`**: A supplementary script demonstrating Q-learning on the OpenAI Gym `FrozenLake-v1` environment (useful for understanding the underlying RL principles).

---

## 🛠 How It Works (Reinforcement Learning)

The project models the Rubik's Cube as an environment for a Reinforcement Learning agent:
- **State Space (`state_size = 72`):** The state is derived from the current color configuration of the cube's faces.
- **Action Space:** 12 possible actions corresponding to the standard 90-degree and -90-degree rotations of the 6 faces.
- **Rewards:** The agent receives a positive reward `(+1)` upon successfully reaching the solved state, and a slight negative penalty `(-0.01)` for each move to encourage finding shorter paths.
- **Algorithm:** Uses Q-Learning with an epsilon-greedy policy, balancing random exploration with exploiting learned values.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).
