import pandas as pd
import numpy as np
import os
from pathlib import Path
from configparser import ConfigParser
import yfinance as yf
import toml

sc_config = {
    'paths': {
        'unformatted': 'E:\\charts\\', 'formatted': 'E:\\charts\\formatted\\'
    },
    'unformatted_aliases':
        {
            'GC_F': 'gc_25.csv', 'ES_F': 'es_25.csv'
        }
}


def multi_sc_df(df_ticker_dict, columns=['Open', 'High', 'Low', 'Close'], ticker_first=False):
    if ticker_first:
        new_cols = pd.MultiIndex.from_product([df_ticker_dict.keys(), columns])
    else:
        new_cols = pd.MultiIndex.from_product([columns, df_ticker_dict.keys()])

    new_df = pd.DataFrame(columns=new_cols)


class sierra_charts:

    def __init__(self, config_file="C:\\Users\\nicho\PycharmProjects\ml_trading\example_config.toml", config_data=None):
        self.resample_logic = {'Open': 'first',
                               'High': 'max',
                               'Low': 'min',
                               'Last': 'last',
                               'Close': 'last',
                               'Volume': 'sum',
                               'BidVolume': 'sum',
                               'AskVolume': 'sum'}

        if not config_data:
            with open(config_file) as f:
                config_data = toml.load(f)

        self.config_fp = config_file
        self.config_data = config_data

        self.unformatted_charts = config_data['paths']['unformatted']
        self.formatted_charts = config_data['paths']['formatted']
        self.intraday_charts = config_data['paths']['intraday']
        self.unformatted_aliases = config_data['aliases']['unformatted']
        self.formatted_aliases = config_data['aliases']['formatted']
        self.intraday_aliases = config_data['aliases']['intraday']
        self.yf_tickers = config_data['yf_tickers']

        self.tickers = list(self.unformatted_aliases.keys())

    def format_from_dict(self, dict):
        """dict : dict structured in the format
        {alias : (load file_path, save_file_name) }"""
        for alias, (load_loc, save_loc) in dict.items():
            self.format_chart(load_loc, alias=alias, new_file=save_loc, save=True)

    def format_chart(self, file_name, date_col='Date', time_col='Time', close_col='Last', resample=False,
                     resample_len='30min', save=False, type='formatted', alias=None, new_file=None):
        df = pd.read_csv(file_name)
        df.columns = df.columns.str.replace(' ', '')
        datetime_series = df[date_col].astype(str) + df[time_col].astype(str).replace(' ', '')
        df['Datetime'] = pd.to_datetime(datetime_series, format='mixed')
        df = df.drop(columns=[date_col, time_col]).set_index(df['Datetime'])
        df['Close'] = df[close_col]
        if resample:
            df = df.resample(resample_len).apply(self.resample_logic)
        if save:
            self.save_formatted_chart(df, new_file, alias, type)

        return df

    def get_chart(self, alias, formatted=True, save_formatted=False, new_file=None, new_alias=None, close_col='Last',
                  date_col='Date', time_col='Time',start_date='2013-01-01', resample=False, resample_period='30min',
                  resample_offset="-8h"):

        if not formatted:
            df = self.format_chart(self.unformatted_charts + self.unformatted_aliases[alias], resample=resample,
                                   resample_len=resample_period)
            df = df.loc[start_date:]


        else:
            df = pd.read_parquet(self.formatted_charts + self.formatted_aliases[alias], engine='pyarrow').dropna()
            df = df.loc[start_date:]

        if save_formatted:
            if new_file and new_alias:
                self.save_formatted_chart(df, file_name=new_file, alias=new_alias)
            else:
                self.save_formatted_chart(df, file_name=self.unformatted_aliases[alias], alias=alias)

        return df

    def update_ohlcv(self, alias, update_data):

        return

    def load_multiple_files(self, ticker_file_dict, resample=True, period='1h', ohlcv_only=True, formatted=False):

        if ohlcv_only:
            ohlc = ['Open', 'High', 'Low', 'Close', 'Volume']
        else:
            ohlc = self.resample_logic.keys()

        new_cols = pd.MultiIndex.from_product([ohlc, ticker_file_dict.keys()])
        new_df = pd.DataFrame(columns=new_cols)

        for i in ticker_file_dict.keys():
            df = pd.read_parquet(self.unformatted_charts + ticker_file_dict[i])[ohlc]
            new_df[[(col, i) for col in ohlc]] = df[ohlc]
            del df

        return new_df

    def save_formatted_chart(self, df, file_name, alias, type='formatted', save_cfg=False, cfg_file=None):
        f_path = self.config_data['paths'][type] + file_name
        df.to_parquet(f_path, engine='pyarrow')
        self.write_config('aliases', type, alias, file_name)

        return print('Saved to: ' + f'{f_path}')

    def format_files(self, resample_period='5min'):
        [self.get_chart(i, save_formatted=True, new_file=i + '_' + resample_period, new_alias=i + resample_period,
                        resample=True, resample_period=resample_period) for i in self.tickers]

    def file_list(self, unformatted=True):
        dir_path = Path(self.unformatted_charts)
        return [f.name for f in dir_path.iterdir() if f.is_file()]

    def open_formatted_files(self):

        files = {}
        [files.update({i: self.get_chart(i, formatted=True)}) for i in self.tickers]

        return files

    def load_intraday(self, alias):
        return pd.read_parquet(self.intraday_charts + self.intraday_aliases[alias])

    def write_config(self, section, sub_section, key, value=None):
        if not value:
            self.config_data[section][sub_section] = key
        else:
            self.config_data[section][sub_section][key] = value

        with open(self.config_fp, 'w') as f:
            toml.dump(self.config_data, f)
        return

    def update_from_dict(self, file_dict, type='intraday', new_file_format=''):
        if type == 'intraday':
            path = self.config_data['paths']['unformatted_tick']
        else:
            path = self.config_data['paths']['unformatted']

        for basket, entries in file_dict.items():
            for i in range(len(entries['aliases'])):
                self.format_chart(path + entries['files'][i], save=True, type=type, alias=entries['aliases'][i],
                                  new_file=entries['files'][i] + new_file_format)

            self.write_config('baskets', basket, entries['aliases'])

        return self.config_data
