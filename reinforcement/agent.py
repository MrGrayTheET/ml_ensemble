'''Based on Double Deep Q Network agent from '''
import os.path
import random
from collections import deque
import numpy as np
import pandas as pd
import tensorflow as tf
import keras._tf_keras as keras
from time import time
import tf_keras.backend as K
from tf_keras.regularizers import l2
from tf_keras.models import Sequential,load_model, clone_model, save_model
from tf_keras.layers import Dense, InputLayer, Dropout
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
            print(f"Skipping training. Memory: {len(self.memory)}")
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

class DDQNAgent:
    def __init__(self, state_dim,
                 num_actions,
                 learning_rate=0.001,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay_steps=250,
                 epsilon_exponential_decay=.99,
                 replay_capacity=int(1e6),
                 architecture=(256,256),
                 l2_reg=1e-6,
                 tau=100,
                 batch_size=1024):

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.experience = deque([], maxlen=replay_capacity)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg

        self.online_network = self.build_model()
        self.target_network = self.build_model(trainable=False)
        self.update_target()

        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []

        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.tau = tau
        self.losses = []
        self.idx = tf.range(batch_size)
        self.train = True

    def build_model(self, trainable=True):
        layers = []
        n = len(self.architecture)
        for i, units in enumerate(self.architecture, 1):
            layers.append(Dense(units=units,
                                input_dim=self.state_dim if i == 1 else None,
                                activation='relu',
                                kernel_regularizer=l2(self.l2_reg),
                                name=f'Dense_{i}',
                                trainable=trainable))
        layers.append(Dropout(.1))
        layers.append(Dense(units=self.num_actions,
                            trainable=trainable,
                            name='Output'))
        model = Sequential(layers)
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target(self):
        self.target_network.set_weights(self.online_network.get_weights())

    def epsilon_greedy_policy(self, state):
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        q = self.online_network.predict(state)
        return np.argmax(q, axis=1).squeeze()

    def memorize_transition(self, s, a, r, s_prime, not_done):
        if not_done:
            self.episode_reward += r
            self.episode_length += 1
        else:
            if self.train:
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        self.experience.append((s, a, r, s_prime, not_done))

    def replay(self):
        if self.batch_size > len(self.experience):
            return
        minibatch = map(np.array, zip(*random.sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states, not_done = minibatch

        next_q_values = self.online_network.predict_on_batch(next_states)
        best_actions = tf.argmax(next_q_values, axis=1)

        next_q_values_target = self.target_network.predict_on_batch(next_states)
        target_q_values = tf.gather_nd(next_q_values_target,
                                       tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))

        targets = rewards + not_done * self.gamma * target_q_values

        q_values = self.online_network.predict_on_batch(states)
        q_values[self.idx, actions] = targets

        loss = self.online_network.train_on_batch(x=states, y=q_values)
        self.losses.append(loss)

        if self.total_steps % self.tau == 0:
            self.update_target()

def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)
def track_results(episode, nav_ma_100, nav_ma_10,
                  market_nav_100, market_nav_10,
                  win_ratio, total, epsilon):
    episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], []
    time_ma = np.mean([episode_time[-100:]])
    T = np.sum(episode_time)

    template = '{:>4d} | {} | Agent: {:>6.1%} ({:>6.1%}) | '
    template += 'Market: {:>6.1%} ({:>6.1%}) | '
    template += 'Wins: {:>5.1%} | eps: {:>6.3f}'
    print(template.format(episode, format_time(total),
                          nav_ma_100 - 1, nav_ma_10 - 1,
                          market_nav_100 - 1, market_nav_10 - 1,
                          win_ratio, epsilon))
def train_dqn(env, agent, num_episodes):
    all_episodes = []
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
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

def train_ddqn(trading_environment, ddqn, max_episodes, state_dim, save_fp='ddqn_model.h5'):
    episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], []

    start = time()
    results = []
    max_episode_steps = trading_environment.spec.max_episode_steps
    for episode in range(1, max_episodes + 1):
        this_state = trading_environment.reset()
        for episode_step in range(max_episode_steps):
            action = ddqn.epsilon_greedy_policy(this_state.reshape(-1, state_dim))
            next_state, reward, done, _ = trading_environment.step(action)
            ddqn.memorize_transition(this_state,
                                     action,
                                     reward,
                                     next_state,
                                     0.0 if done else 1.0)
            if ddqn.train:
                ddqn.experience_replay()
            if done:
                break
            this_state = next_state


        # get DataFrame with seqence of actions, returns and nav values
        result = trading_environment.env.simulator.result()

        # get results of last step
        final = result.iloc[-1]

        # apply return (net of cost) of last action to last starting nav
        nav = final.nav * (1 + final.strategy_return)
        navs.append(nav)

        # market nav
        market_nav = final.market_nav
        market_navs.append(market_nav)
        print(f'Episode: {episode}, Market Nav: {market_nav}, Strategy Nav: {nav}')
        # track difference between agent an market NAV results
        diff = nav - market_nav
        diffs.append(diff)


        if episode % 10 == 0:
            track_results(episode,
                          # show mov. average results for 100 (10) periods
                          np.mean(navs[-100:]),
                          np.mean(navs[-10:]),
                          np.mean(market_navs[-100:]),
                          np.mean(market_navs[-10:]),
                          # share of agent wins, defined as higher ending nav
                          np.sum([s > 0 for s in diffs[-100:]]) / min(len(diffs), 100),
                          time() - start, ddqn.epsilon)
        if episode % 50 == 0:
            print(f'Saving model to {save_fp}')
            save_model(ddqn , save_fp)

        if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):
            print(result.tail())
            break
    result_data = pd.DataFrame(data={'Episode':range(1, max_episodes+1), 'Market':market_navs, 'Diff': diffs})
    trading_environment.close()

    return result_data, ddqn
