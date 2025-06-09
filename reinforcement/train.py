from agent import DQNAgent, train
from trading_env import TradingEnvironment
import sys
import os
import logging
import pandas as pd
import numpy as np
from sc_loader import sierra_charts as sc
from technical_prep import FeaturePrep as FP
import toml
from argparse import Namespace

self = Namespace

t_env = TradingEnvironment(1250,source='sc', ticker='ES_F',data_timeframe='4h')

t_agent = DQNAgent(state_dim=t_env.reset().shape[0], action_dim=t_env.action_space.n)

train(t_env, t_agent, 100)