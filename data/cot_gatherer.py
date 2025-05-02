import cot_reports
import pandas as pd



reports11 = cot_reports.cot_year(2011)
reports12 = cot_reports.cot_year(2012)
reports13 = cot_reports.cot_year(2013)
reports14 = cot_reports.cot_year(2014)
reports15 = cot_reports.cot_year(2015)
reports16 = cot_reports.cot_year(2016)
reports17 = cot_reports.cot_year(2017)
reports18 = cot_reports.cot_year(2018)
reports19=cot_reports.cot_year(2019)
reports20 = cot_reports.cot_year(2020)
reports21 = cot_reports.cot_year(2021)
reports22 = cot_reports.cot_year(2022)
reports23 = cot_reports.cot_year(2023)
reports24 = cot_reports.cot_year(2024)
aggregated_reports  = pd.concat([reports24,reports23,reports22,reports21,reports20,reports19, reports18, reports17, reports16, reports15, reports14, reports13, reports12, reports11], ignore_index=True)

filtered_df = aggregated_reports[["Market and Exchange Names", 'CFTC Contract Market Code', "As of Date in Form YYYY-MM-DD", "Noncommercial Positions-Long (All)",
                  "Noncommercial Positions-Short (All)", "Change in Noncommercial-Long (All)", "Change in Noncommercial-Short (All)",
                  "% of OI-Noncommercial-Long (All)","% of OI-Noncommercial-Short (All)" ]]


cot_dict = {'CL_F':['067651'], 'ES_F': ['13874A'], 'NQ_F':['209742']}
aggregated_reports.index = pd.to_datetime(aggregated_reports['As of Date in Form YYYY-MM-DD'])

def return_COT_data(contract_ticker):
    filter = aggregated_reports['CFTC Contract Market Code'].isin(cot_dict.get(contract_ticker))
    return aggregated_reports.loc[filter].sort_index()

crude_cot = return_COT_data('CL_F')
es_f = return_COT_data('ES_F')
nq_f = return_COT_data('NQ_F')
