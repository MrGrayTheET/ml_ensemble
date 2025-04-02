# futures_ml

#Program for quickly building models. Models include Random Forest Regressors and Xgboost regressor/classifier. 
 
ml_build contains the ml_model module/class, which inputs data, features, and target upon initialization. The model then calls the clean_data function from utilities to automatically clean the data. 

There is also a dl_model class that focuses on deep learning,but it is not complete yet. The clean data function is still useful for preprocessing data, as it can convert to tensors and create sequential input/output data through the sequential and to_tensor arguments of clean_data

model_prep is a way of putting together trend/seasonal decomposition models. Features have a setter feature which reads which features to add from its settings . 
Trend features contains indicators like BBands, SMAs, and trend from seasonal decomposition. Seasonal is the residuals and seasonal patterns from seasonal decomp. I am currently working on other feature types like Volume
It can also input custom features/series using the custom_features method

model_prep also contains functions to load/save model parameters and data to a directory. It saves to a directory in the model_prep project_dir argument + the model name. In the directory it saves
the model_info data containing the added features. When loading data, it may be necessary to add the feature types to the load_model argument.

Currently there are "Volatility", "Trend", and "Seasonal" feature types, though more may be added over time