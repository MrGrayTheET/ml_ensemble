import pandas as pd
import os
import cot_reports

bar_data_PATH = "C:\\Users\\nicho\\charts\\"
alt_data_PATH = "C:\\Users\\nicho\\alt_data\\sentiment\\sentiment.csv"
files = [bar_data_PATH + i for i in os.listdir(bar_data_PATH)]
ticker_names = ['CL_F', 'ES_F', 'NG_F', 'VX_F', 'ZC_F', 'ZS_F'] # Creates a dictionary linked to  chart files
file_dict = dict(zip(ticker_names, files))
# Variable containing cleaned AAII sentiment data

sentiment = pd.read_csv('../hull_test/venv/Lib/site-packages/ml_build/data/sentiment.csv', index_col='Date', date_format='%YY-%mm-%dd', parse_dates=True)
sentiment = sentiment.replace('%', '', regex=True)
sentiment = sentiment.replace(',', '', regex=True)

sentiment.index = pd.to_datetime(sentiment.index)

sentiment.to_csv('sentiment_2')


class cot_data:

    def __init__(self):
        self.aggregated_reports = pd.read_csv(
            '../hull_test/venv/Lib/site-packages/ml_build/data/agg_legacy_reports', index_col='As of Date in Form YYYY-MM-DD')

        self.non_commercials = ["Market and Exchange Names", 'CFTC Contract Market Code',
                                                    "Noncommercial Positions-Long (All)",
                                                    "Noncommercial Positions-Short (All)",
                                                    "Change in Noncommercial-Long (All)",
                                                    "Change in Noncommercial-Short (All)",
                                                    "% of OI-Noncommercial-Long (All)",
                                                    "% of OI-Noncommercial-Short (All)",]



        self.code_dict = {'CL_F': ['067651'], 'ES_F': ['13874A'], 'NQ_F': ['209742'], 'LE_F': ['057642']}

        None

    def contract_data(self, contract, by_ticker=True):

        filtered_df = self.aggregated_reports[["Market and Exchange Names", 'CFTC Contract Market Code',
                                               "Noncommercial Positions-Long (All)",
                                               "Noncommercial Positions-Short (All)",
                                               "Change in Noncommercial-Long (All)",
                                               "Change in Noncommercial-Short (All)",
                                               "% of OI-Noncommercial-Long (All)",
                                               "% of OI-Noncommercial-Short (All)",
                                               'Change in Commercial-Long (All)',
                                               'Change in Commercial-Short (All)', '% of OI-Commercial-Long (All)',
                                               '% of OI-Commercial-Short (All)', '% of OI-Commercial-Long (Old)',
                                               '% of OI-Commercial-Short (Old)',
                                               '% of OI-Commercial-Long (Other)' ]]

        filter = filtered_df['CFTC Contract Market Code'].isin(self.code_dict.get(contract))

        return self.aggregated_reports[filter]

    def download_cot_year_list(self, year_list, report_type='legacy_fut'):
        for i in year_list:
            cot_i = cot_reports.cot_year(i)
            self.aggregated_reports = pd.concat([self.aggregated_reports, cot_i],axis=0, ignore_index=True)
        self.aggregated_reports = self.aggregated_reports.set_index(pd.to_datetime(self.aggregated_reports['As of Date in Form YYYY-MM-DD']))
        self.aggregated_reports.to_csv('agg_legacy_reports')

        return self.aggregated_reports.sort_values(by='CFTC Contract Market Code', axis=0)
