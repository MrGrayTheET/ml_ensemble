import random
from collections import deque
import numpy as np
import pandas as pd
import tensorflow as tf
import keras._tf_keras as keras
import tf_keras.backend as K
from tf_keras.models import Sequential,load_model, clone_model
from tf_keras.layers import Dense, InputLayer
from tf_keras.optimizers import Adam

device = tf.config
def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber's loss - Custom Loss Function for Q Learning

    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))

class DQNAgent:
    def __init__(self, state_dim, action_dim,loss='mse', lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = 64
        self.loss_func = loss
        self.memory = deque(maxlen=10000)
        self.lr = lr
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def _build_model(self):

        model = Sequential([
            InputLayer(input_shape=self.state_dim),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_dim, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.lr), loss=self.loss_func)

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        q_values = self.model.predict(state[np.newaxis, :], verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        target_qs = self.model.predict(states, verbose=0)
        next_qs = self.target_model.predict(next_states, verbose=0)

        for i in range(self.batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.max(next_qs[i])
            target_qs[i][actions[i]] = target

        self.model.fit(states, target_qs, epochs=1, verbose=0)

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train(env, agent, num_episodes):
    all_episodes = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        agent.update_target_model()
        df = env.simulator.result().copy()
        print(f'End Results:\n {df.iloc[-1]}')
        df['step'] = df.index
        df['episode'] = episode
        df.set_index(['episode', 'step'], inplace=True)
        all_episodes.append(df)

        print(f"Episode {episode+1}:  Reward{ total_reward:.4f}")

    results_df = pd.concat(all_episodes)

    return results_df


