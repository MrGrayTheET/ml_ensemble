import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_build.utils import clean_data, evaluate_model, evaluate_clf, create_labels,save_model, load_model
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error

rfr_params = {'max_depth': [3,5],
              'max_features': [5,6,7],
              'n_estimators': [200]}

gbr_params = {'max_features': [5,6,7],
              'learning_rate':[0.1],
              'n_estimators':[200],
              'subsample':[0.6],
              'random_state':[42]}

xgb_params = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.8, 0.9, 1.0],
    'n_estimators':[200],
    'max_leaves': [5, 7, 8]

}

rf_clf_params = {'n_estimators':[200],
                 'subsample':[0.6],
                 'random_state':[42],
                 'learning_rate':[0.01, 0.05, 0.1, 0.2],
                 'max_depth':[3, 5, 7]}

xgb_clf_params = dict(n_estimators=[200],
                      max_depth=[3,5,7],
                      learning_rate=[0.01, 0.05, 0.1, 0.2],
                      subsample=[0.6,  0.8],
                      random_state=[42])

class ml_model:

    def __init__(self, data, features,target_column,train_test_size=0.8,scale_x=False,
                 scale_y=False, log_results=False, log_file_path='ml_log'):
        self.feats = features
        self.model = None
        self.train_predict, self.test_predict = \
            [None] * 2  # Vars for future use

        self.params = {}
        self.x_final = data[features].tail(50)


        if type(data.index) == pd.DatetimeIndex:
            data.index = range(0, data.shape[0])  # Cleans NAs and drops DateTimeIndex

        [self.x_train, self.y_train],  [self.x_test, self.y_test] =\
            clean_data(data, features, target_column, scale_x=scale_x, scale_y=scale_y, train_split=True, train_size=train_test_size)


    def plot_predictions(self, training_pred=False, test_pred=True):
        if training_pred and test_pred:
            fig, ax = plt.subplots(2)
            ax[0].scatter(self.y_train, self.train_predict, label='train')
            ax[1].scatter(self.y_test, self.test_predict, label='test')

        elif training_pred and not test_pred:
            plt.scatter(self.y_train, self.train_predict)

        elif test_pred and not training_pred: plt.scatter(self.y_test, self.test_predict)

        plt.show()

        return None

    def tree_model(self, parameter_dict, gbr=False, plot_pred=True, plot_importances=True, save_params=False,
                   params_file='rfr_params.csv', evaluate=True, eval_log='model_eval.csv'):
        test_scores = []
        rmse_scores = []

        if not gbr:
            model = RandomForestRegressor(n_estimators=200)
            name = 'rfr'
        else:
            model = GradientBoostingRegressor(n_estimators=200)
            name = 'gbr'

        for g in ParameterGrid(parameter_dict):
            model.set_params(**g)
            model.fit(self.x_train, self.y_train)
            test_scores.append(model.score(self.x_train, self.y_train))
            rmse_scores.append(root_mean_squared_error(self.y_test, model.predict(self.x_test)))

        best_idx = np.argmax(test_scores)
        best_rmse = np.argmin(rmse_scores)
        print(f'best score: {test_scores[best_idx]}\n')
        print(f'best settings {ParameterGrid(parameter_dict)[best_idx]}\n')
        print(f'remaining scores:{test_scores}')
        print(f'Best RMSE: {rmse_scores[best_rmse]}')
        print(f'best rmse settings: {ParameterGrid(parameter_dict)[best_rmse]}')
        self.params.update({name: ParameterGrid(parameter_dict)[best_idx]})

        model.set_params(**self.params[name])
        self.model = model
        self.train_predict = model.predict(self.x_train)
        self.test_predict = model.predict(self.x_test)

        if plot_pred:
            if plot_importances:
                self.plot_importances(subplots=True)

            else:
                self.plot_predictions(training_pred=True, test_pred=True)
        else:
            if plot_importances:
                self.plot_importances(self.model, subplots=False)

        if save_params:
            params = pd.DataFrame(data=self.params[name], index=[pd.Timestamp.today()]).to_csv(params_file)

        if evaluate:
            self.eval = evaluate_model(self.test_predict, self.y_test, self.feats,sorted_features=True, log_file=eval_log)

        return self.model

    def plot_importances(self, model, subplots=False):
        importance = model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        x = range(len(importance))
        labels = np.array(self.feats)[sorted_idx]
        self.feats = labels

        if subplots:
            fig, ax = plt.subplots(3)
            ax[2].bar(x, importance[sorted_idx], tick_label=labels)
            ax[0].scatter(self.y_train, self.train_predict, label='train')
            ax[0].axvline(x=0)
            ax[0].axhline(y=0)
            ax[1].scatter(self.y_test, self.test_predict, label='test')
            ax[1].axvline(x=0)
            ax[1].axhline(y=0)
        else:
            plt.bar(x=labels, height=importance)

        return plt.show()

    def neighbors_model(self, n_start, n_end, plot_pred=True,
                        evaluate=True, eval_log='model_eval.csv'):
        results_df = pd.DataFrame(columns=['train_r2', 'test_r2', 'n_neighbors'], index=range(n_start, n_end))
        for n in range(n_start, n_end):
            knn = KNeighborsRegressor(n_neighbors=n)
            knn.fit(self.x_train, self.y_train)
            results_df['n_neighbors'].loc[n] = n
            results_df['train_r2'].loc[n] = knn.score(self.x_train, self.y_train)
            results_df['test_r2'].loc[n] = knn.score(self.x_test, self.y_test)

        sorted_models = np.argsort(results_df['test_r2'])[::-1]
        best_neighbors = results_df['n_neighbors'].iloc[sorted_models].iloc[0]

        knn.set_params(**{'n_neighbors': best_neighbors})

        self.model = knn
        self.train_predict = self.model.predict(self.x_train)
        self.test_predict = self.model.predict(self.x_test)

        if evaluate:

            self.eval = evaluate_model(self.test_predict, self.y_test, self.feats, eval_log)


        if plot_pred:
            self.plot_predictions(test_pred=True, training_pred=True)

        return self.model

    def xgb_model(self, param_dict, plot_pred=True, plot_importances=True, save_params=True,
                  params_file='xgb_params.csv', evaluate=True, eval_log='xgb_eval.csv'):
        test_scores =[]
        rmse_scores = []

        param_grid = param_dict

        grid_search = GridSearchCV(XGBRegressor(), param_grid, cv=3)

        grid_search.fit(self.x_train, self.y_train)
        best_params = grid_search.best_params_

        self.model = XGBRegressor(**best_params)
        self.model.fit(self.x_train, self.y_train)

        self.train_predict = self.model.predict(self.x_train)
        self.test_predict = self.model.predict(self.x_test)

        if plot_pred:
            if plot_importances:
                self.plot_importances(self.model, subplots=True)
            else:
                self.plot_predictions(training_pred=True, test_pred=True)
        elif plot_importances and not plot_pred:
            self.plot_importances(self.model, subplots=True)

        if evaluate:
            self.eval = evaluate_model(self.test_predict, self.y_test,self.feats, log_file=eval_log, sorted_features=True)
            print(eval)
        else:
            pass

        return self.model

    def xgb_clf(self, high_p:int, low_p:int, parameter_dict, plot_pred=False, plot_importances=True, evaluate=True):
        '''high_p:int percentile used to decide the 1 class
            low_p: int percentile used to decide the -1 class
            paramter_dict:dict hyperparameter dictionary for GridSearchCV
        '''
        test_scores = []
        acc_scores = []

        self.clf_train, self.clf_test = create_labels(self.y_train, high_p, low_p),create_labels(self.y_test, high_p, low_p)

        #initiate grid_search
        grid_search = GridSearchCV(XGBClassifier(), parameter_dict, scoring='accuracy', cv=3)


        # Create labels for training

        grid_search.fit(self.x_train, self.clf_train)

        #Fit and test model
        self.model = XGBClassifier(**grid_search.best_params_)
        self.model.fit(self.x_train, self.clf_train)
        self.train_predict = self.model.predict(self.x_train)
        self.test_predict = self.model.predict(self.x_test)
        self.train_probs = self.model.predict_proba(self.x_train)
        self.test_probs = self.model.predict_proba(self.x_test)

        if plot_importances:
            if not plot_pred:
                self.plot_importances(self.model, subplots=False)
            else:
                self.plot_importances(self.model, subplots=True)

            if evaluate:
                model_eval = evaluate_clf(self.test_predict, self.test_probs, self.clf_test,  self.feats)
                print(model_eval)
                self.eval = model_eval

        return self.model

    def save_model(self,file_name:str, save_fp='F:\\ML\\models\\'):
        save_model(self.model, save_fp+file_name)
        print(f'Model successfully Saved to {save_fp+file_name}')
        return

    def load_model(self, model_fp):
        model = load_model(model_fp)
        self.model = model
        return self.model

    def forecast_data(self, n_periods_back=10):

        return self.model.predict(self.x_final[-n_periods_back:])






