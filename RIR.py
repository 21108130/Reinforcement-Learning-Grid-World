import tkinter as tk
import numpy as np
import time
from PIL import Image, ImageTk
import pygame

class GridWorld:
    def __init__(self, grid_size=5, max_steps=50):
        self.grid_size = grid_size
        self.state = (0, 0)
        self.goal_state = (grid_size-1, grid_size-1)
        self.obstacles = [(1, 1), (2, 2), (3, 3)]  # Adjusted number of obstacles
        self.actions = ['up', 'down', 'left', 'right']
        self.max_steps = max_steps
        self.steps = 0

    def reset(self):
        self.state = (0, 0)
        self.steps = 0
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 'up' and x > 0:
            x -= 1
        elif action == 'down' and x < self.grid_size - 1:
            x += 1
        elif action == 'left' and y > 0:
            y -= 1
        elif action == 'right' and y < self.grid_size - 1:
            y += 1

        self.state = (x, y)
        self.steps += 1

        reward = -100 if self.state in self.obstacles else 10 if self.state == self.goal_state else -0.1
        if self.state == self.goal_state and self.steps < self.max_steps / 2:
            reward -= 50  # Increased penalty for reaching the goal too quickly

        done = self.state == self.goal_state or self.state in self.obstacles or self.steps >= self.max_steps
        return self.state, reward, done

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        x, y = self.state
        gx, gy = self.goal_state
        grid[x, y] = -1  # Robot
        grid[gx, gy] = 1  # Goal
        for ox, oy in self.obstacles:
            grid[ox, oy] = -2  # Obstacle
        return grid

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((env.grid_size, env.grid_size, len(env.actions)))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)
        else:
            x, y = state
            return self.env.actions[np.argmax(self.q_table[x, y])]

    def update_q_table(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        action_idx = self.env.actions.index(action)
        best_next_action = np.max(self.q_table[next_x, next_y])
        td_target = reward + self.gamma * best_next_action
        td_error = td_target - self.q_table[x, y, action_idx]
        self.q_table[x, y, action_idx] += self.alpha * td_error

    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

class RLVisualizer(tk.Tk):
    def __init__(self, agent, env):
        super().__init__()
        self.agent = agent
        self.env = env
        self.grid_size = env.grid_size
        self.cell_size = 100
        self.title("Reinforcement Learning Grid World")
        self.geometry(f"{self.grid_size * self.cell_size}x{self.grid_size * self.cell_size + 50}")
        self.canvas = tk.Canvas(self, width=self.grid_size * self.cell_size, height=self.grid_size * self.cell_size)
        self.canvas.pack()
        self.info_label = tk.Label(self, text="")
        self.info_label.pack()
        
        self.robot_image = Image.open("Robot.png").resize((self.cell_size, self.cell_size))
        self.charging_station_image = Image.open("charging station.png").resize((self.cell_size, self.cell_size))
        self.obstacle_image = Image.open("obstacle.png").resize((self.cell_size, self.cell_size))
        
        self.robot_photo = ImageTk.PhotoImage(self.robot_image)
        self.charging_station_photo = ImageTk.PhotoImage(self.charging_station_image)
        self.obstacle_photo = ImageTk.PhotoImage(self.obstacle_image)
        
        self.cumulative_reward = 0
        self.current_episode = 0  # Track current episode number
        self.after(0, self.run_episode)

        # Initialize pygame mixer
        pygame.mixer.init()

        # Load sound effects
        self.obstacle_sound = pygame.mixer.Sound("Voicy_Sorry.mp3")
        self.goal_sound = pygame.mixer.Sound("mixkit-sci-fi-robot-speaking-289.wav")

        # Policy variables
        self.explore_steps = 2000  # Number of steps to explore before exploiting
        self.steps_taken = 0

    def render_grid(self):
        grid = self.env.render()
        self.canvas.delete("all")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x0, y0 = j * self.cell_size, i * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size
                if grid[i, j] == -1:
                    self.canvas.create_image((x0 + self.cell_size // 2, y0 + self.cell_size // 2), image=self.robot_photo)
                elif grid[i, j] == 1:
                    self.canvas.create_image((x0 + self.cell_size // 2, y0 + self.cell_size // 2), image=self.charging_station_photo)
                elif grid[i, j] == -2:
                    self.canvas.create_image((x0 + self.cell_size // 2, y0 + self.cell_size // 2), image=self.obstacle_photo)
                else:
                    self.canvas.create_rectangle(x0, y0, x1, y1, outline="black")

    def run_episode(self):
        self.current_episode += 1
        state = self.env.reset()
        done = False
        episode_reward = 0
        while not done:
            self.render_grid()
            self.update_idletasks()
            self.update()
            time.sleep(0.2)  # Adjusted sleep time for smoother animation
            
            # Choose action
            if self.steps_taken < self.explore_steps:
                action = np.random.choice(self.env.actions)  # Exploration phase
            else:
                action = self.agent.choose_action(state)  # Exploitation phase
            
            # Take action
            next_state, reward, done = self.env.step(action)
            
            # Update Q-table
            self.agent.update_q_table(state, action, reward, next_state)
            state = next_state
            
            # Play sound effects
            if reward == -100:
                self.obstacle_sound.play()
            elif reward == 10 and self.env.steps < self.env.max_steps / 2:
                self.goal_sound.play()
            
            # Update rewards
            episode_reward += reward
            self.cumulative_reward += reward
            self.info_label.config(text=f"Episode: {self.current_episode} | Current Reward: {episode_reward:.2f} | Cumulative Reward: {self.cumulative_reward:.2f}")
            
            # Update steps taken
            self.steps_taken += 1
        
        # Reset steps taken after each episode
        self.steps_taken = 0
        
        # Render final state
        self.render_grid()
        
        # Schedule next episode after a delay
        self.after(1000, self.run_episode)

if __name__ == "__main__":
    env = GridWorld(grid_size=5, max_steps=50)  # Adjust grid size and maximum steps
    agent = QLearningAgent(env)
    agent.train(num_episodes=30000)  # Increase number of training episodes
    
    app = RLVisualizer(agent, env)
    app.mainloop()
