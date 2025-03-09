# agent.py
import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay_rate=0.005):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_table = np.zeros((state_size, action_size))

    def get_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state, :])
            
    def update_q_table(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state, :])
        self.q_table[state, action] = predict + self.learning_rate * (target - predict)

    def decay_exploration_rate(self):
        self.exploration_rate = max(0.01, self.exploration_rate - self.exploration_decay_rate)
        
    def reset_exploration_rate(self):
        self.exploration_rate = 1