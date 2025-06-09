""""
The MIT License (MIT)

Copyright (c) 2016 Tito Ingargiola
Copyright (c) 2019 Stefan Jansen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modified by Mr Gray
"""
import os
import sys
import toml
import logging
import tempfile
import gym
import numpy as np
import pandas as pd
import yfinance as yf

from gym import spaces
from gym.utils import seeding
from technical_prep import FeaturePrep
from sklearn.preprocessing import StandardScaler
from feature_engineering import extract_time_features
from volatility.regimes import WKFi
from sc_loader import sierra_charts as sch

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)

if sys.platform != 'win32':
    SC_CFG_FP = '/content/ml_ensemble/configs/colab/colab_sc.toml'
    FEATURES_CONFIG_FP = '/content/ml_ensemble/configs/env_config.toml'
else:
    SC_CFG_FP = 'C:\\Users\\nicho\PycharmProjects\ml_trading\configs\loader_config.toml'
    FEATURES_CONFIG_FP = "C:\\Users\\nicho\PycharmProjects\ml_trading\configs\env_config.toml"

sc = sch(SC_CFG_FP)
tf = '1d'


class DataSource:
    """
    Data source for TradingEnvironment

    Loads & preprocesses daily price & volume data
    Provides data for each new episode.

    Uses Either Yfinance or Formatted Sierra Charts data
    """

    def __init__(self, trading_days=252,
                 ticker='AAPL',
                 start_date='2014-01-01',
                 data_source='yf',
                 data_timeframe='1d',
                 ohlc=False,
                 normalize=True,
                 ):
        self.timestamps = None
        self.ticker = ticker
        self.trading_days = trading_days
        self.normalize = normalize
        self.tf = data_timeframe
        self.additional_features = []
        self.scaler = StandardScaler()

        if ohlc:
            self.additional_features += ['Open', 'High', 'Low', 'Close']

        self.data = self.load_data(ticker, start_date, data_source)
        self.preprocess_data()

        self.min_values = self.data.min().values
        self.max_values = self.data.max().values
        self.step = 0
        self.offset = None

    def load_data(self, ticker, start_date, data_source='sc'):
        if data_source == 'yf':
            df = yf.download(ticker, start=start_date, interval=self.tf)
        else:
            print(start_date)
            df = sc.get_chart(ticker, formatted=True, start_date=start_date)

        log.info('got data for {}...'.format(self.ticker))

        return df

    def preprocess_data(self):
        """calculate returns and percentiles, then removes missing values"""
        with open(FEATURES_CONFIG_FP, 'r') as f:
            feature_params = toml.load(f)

        pre_model = FeaturePrep(self.data, intraday_tfs=[self.tf])

        if feature_params['types']['Volatility']:
            pre_model.volatility_signals(timeframe=self.tf,
                                         normalize_atr=self.normalize,
                                         normalize=self.normalize,
                                         **feature_params['Volatility'])

        if feature_params['types']['Volume']:
            pre_model.volume_features(timeframe=self.tf, **feature_params['Volume'])

        if feature_params['types']['Trend']:
            if feature_params['Trend']['KAMAs']:
                kama_params = feature_params['Trend']['kama_params']
                feature_params['Trend']['kama_params'] = [
                    tuple(kama_params[n:n + 3]) for n in range(0, len(kama_params), 3)
                ]

            pre_model.trend_indicators(timeframe=self.tf, normalize_features=self.normalize, **feature_params['Trend'])

        if feature_params['cluster']['use_cluster']:
            if feature_params['cluster']['method'] == 'wkmeans':
                model = WKFi(pre_model.dfs_dict[self.tf].copy(), **feature_params['cluster']['wkmeans'])
                model.fit_windows()
                df = model.predict_clusters()
                pre_model.dfs_dict[self.tf]['cluster'] = df['cluster']
                self.additional_features += ['cluster']

        training_types = [key for key, v in feature_params['types'].items() if v]
        pre_model.feats_dict[self.tf]['Additional'] += self.additional_features
        self.training_df_data = pre_model.dfs_dict[self.tf]

        pre_model.prepare_for_training(self.tf, feature_types=training_types, **feature_params['Training'])
        cols = ['target_returns'] + pre_model.features

        self.data = pre_model.training_df.copy()

        if isinstance(self.data.index, pd.DatetimeIndex):
            self.data = extract_time_features(self.data, set_index=True, hour=True, dayofweek=True)
            self.timestamps = self.data['datetime']
            self.data.index = np.arange(len(self.data))
            self.data.drop(columns='datetime', inplace=True)

        self.data = pd.DataFrame(data=self.scaler.fit_transform(self.data[pre_model.features]),
                                 columns=pre_model.features)
        self.data.insert(0, 'returns', pre_model.training_df['target_returns'])
        log.info(self.data.info())

        return self.data

    def reset(self):
        """Provides starting index for time series and resets step"""
        high = len(self.data.index) - self.trading_days
        self.offset = np.random.randint(low=0, high=high)
        self.step = 0

    def take_step(self):
        """Returns data for current trading day and done signal"""
        obs = self.data.iloc[self.offset + self.step].values
        timestamp = self.timestamps.iloc[self.offset + self.step]
        self.step += 1
        done = self.step > self.trading_days
        return obs, timestamp, done


class TradingSimulator:
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, steps, trading_cost_bps, time_cost_bps):
        # invariant for object life
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps

        # change every step
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.market_navs = np.ones(self.steps)
        self.strategy_returns = np.ones(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.market_returns = np.zeros(self.steps)
        self.timestamps = np.empty(self.steps, dtype='datetime64[ns]')

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.market_navs.fill(1)
        self.strategy_returns.fill(0)
        self.positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.market_returns.fill(0)

    def take_step(self, action, market_return, timestamp):
        """ Calculates NAVs, trading costs and reward
            based on an action and latest market return
            and returns the reward and a summary of the day's activity. """

        start_position = self.positions[max(0, self.step - 1)]
        start_nav = self.navs[max(0, self.step - 1)]
        start_market_nav = self.market_navs[max(0, self.step - 1)]
        self.market_returns[self.step] = market_return
        self.actions[self.step] = action
        self.timestamps[self.step] = timestamp

        end_position = action - 1  # short, neutral, long
        trade = end_position - start_position

        if action != 1 and end_position != 0:
            n_trades = self.trades[self.step - 1] + 1
            if action == 2:
                print(f'Buy 1 at {timestamp}\n Cumulative trades: {n_trades}\n\n')
            if action == 0:
                print(f'Sell 1 at {timestamp}\n Cumulative trades: {n_trades}\n\n')


        else:
            n_trades = self.trades[self.step - 1]

        self.positions[self.step] = end_position
        self.trades[self.step] = n_trades

        # roughly value based since starting NAV = 1
        trade_costs = abs(trade) * self.trading_cost_bps
        time_cost = 0 if trade else self.time_cost_bps
        self.costs[self.step] = trade_costs + time_cost
        reward = start_position * market_return - self.costs[max(0, self.step - 1)]
        self.strategy_returns[self.step] = reward

        if self.step != 0:
            self.navs[self.step] = start_nav * (1 + self.strategy_returns[self.step])
            self.market_navs[self.step] = start_market_nav * (1 + self.market_returns[self.step])

        info = {'reward': reward,
                'nav': self.navs[self.step],
                'costs': self.costs[self.step]}

        self.step += 1
        return reward, info

    def result(self):
        """returns current state as pd.DataFrame """
        return pd.DataFrame({'timestamp': self.timestamps,
                             'action': self.actions,
                             'timestamp': self.timestamps,  # current action
                             'nav': self.navs,  # starting Net Asset Value (NAV)
                             'market_nav': self.market_navs,
                             'market_return': self.market_returns,
                             'strategy_return': self.strategy_returns,
                             'position': self.positions,  # eod position
                             'cost': self.costs,  # eod costs
                             'trade': self.trades,
                             })  # eod trade)


class TradingEnvironment(gym.Env):
    """A simple trading environment for reinforcement learning.

    Provides daily observations for a stock price series
    An episode is defined as a sequence of 252 trading days with random start
    Each day is a 'step' that allows the agent to choose one of three actions:
    - 0: SHORT
    - 1: HOLD
    - 2: LONG

    Trading has an optional cost (default: 10bps) of the change in position value.
    Going from short to long implies two trades.
    Not trading also incurs a default time cost of 1bps per step.

    An episode begins with a starting Net Asset Value (NAV) of 1 unit of cash.
    If the NAV drops to 0, the episode ends with a loss.
    If the NAV hits 2.0, the agent wins.

    The trading simulator tracks a buy-and-hold strategy as benchmark.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 trading_days=252,
                 trading_cost_bps=1e-3,
                 time_cost_bps=1e-4,
                 data_timeframe='1d',
                 data_source='sc',
                 start_date='2014-01-01',
                 include_ohlc=True,
                 ticker='ES_F'):
        self.trading_days = trading_days
        self.trading_cost_bps = trading_cost_bps
        self.ticker = ticker
        self.time_cost_bps = time_cost_bps
        self.data_source = DataSource(trading_days=self.trading_days,
                                      data_source=data_source,
                                      start_date=start_date,
                                      ticker=ticker, ohlc=include_ohlc)

        self.timestamps = self.data_source.timestamps

        self.simulator = TradingSimulator(steps=self.trading_days,
                                          trading_cost_bps=self.trading_cost_bps,
                                          time_cost_bps=self.time_cost_bps)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.data_source.min_values,
                                            self.data_source.max_values)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Returns state observation, reward, done and info"""

        assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action))

        observation, timestamp, done = self.data_source.take_step()

        reward, info = self.simulator.take_step(action=action,
                                                market_return=observation[0],
                                                timestamp=timestamp)

        return observation, reward, done, info

    def reset(self):
        """Resets DataSource and TradingSimulator; returns first observation"""
        self.data_source.reset()
        self.simulator.reset()
        return self.data_source.take_step()[0]
