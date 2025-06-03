
from linear_models import multivariate_regression
import fredapi as Fred

fredkeyfile = "F:\\Macro\\fredapi.txt"
with open(fredkeyfile) as fredkey:
    key = fredkey.read()

fred = Fred(key)

class MacroModel:

    def __init__(self, tickers=[], data_dir='F:\\factors\\', loading_method='yf', model_name='factor_example',
                 interval='1mo', start='2020-01-01', end='2025-04-09'):
        self.macro_df = None
        self.new_col = lambda col_name: [(col_name, ticker) for ticker in tickers]
        self.ticker_ohlc = lambda ticker: tuple(zip([ticker * 4], ['Open', 'High', 'Low', 'Close']))

        if loading_method == 'sc':
            import sc_loader as sc
            loader = sc()
            self.load = loader.open_formatted_files
            self.data = self.load()
            columns = pd.MultiIndex.from_product([['Open', 'High', 'Low', 'Close'], tickers])
            df = pd.DataFrame(columns=columns, index=pd.date_range(start, end, freq=interval))




        else:
            self.load = yf.download
            if (interval.endswith('w')):
                yf_interval = '1wk'

            elif interval.endswith('min'):
                yf_interval = interval.strip('in')

            else:
                yf_interval = interval

            self.data = self.load(tickers, start=start, end=end, interval=yf_interval)
            self.cols = lambda col_name: [(col_name, ticker) for ticker in tickers]

        if not data_dir.endswith('\\'):
            self.dir = data_dir + '\\'
        else:
            self.dir = data_dir

        self.data[self.cols('returns')] = np.log(self.data.Close.values / self.data.Close.shift(1).values)

        if not os.path.isdir(data_dir):
            os.mkdir(data_dir + '\\')
            os.mkdir(data_dir + model_name)
        elif not os.path.isdir(data_dir + model_name):
            os.mkdir(data_dir + model_name)
        else:
            overwrite = input(
                f"Directory {self.dir} already contains a model named {model_name}\n Would you like to overwrite? Y/N").strip().lower()
            if overwrite != 'y':
                pass
            else:
                exit()

        return

    def price_index(self, weight_dict: dict, type='Fred', data=None, normalize=False):
        series_names = [*weight_dict.keys()]
        if type == 'Fred':
            _df = get_macro_vars(series_names, transformation=lambda x:np.log(x/x.shift(1)))
        else:
            _df = self.load(series_names)

        index_df = weighted_index(_df, weight_dict, normalize=normalize)

        return index_df

    def benchmark_regression(self, feature_col=None, benchmark_ticker='^GSPC'):
        if feature_col is None:
            y = yf.download(benchmark_ticker, start=self.data.index[0], end=self.data.index[-1])['Close']
        else:
            y = self.data[feature_col]

        for i, col in self.data['returns'].columns:
            x = self.data['returns'][col]

        return
