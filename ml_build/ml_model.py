import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_build.utils import clean_data, evaluate_model, evaluate_clf, create_labels, create_label_dataset,save_model, load_model
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier as LGBClf, LGBMRegressor as LGBReg
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score,log_loss, root_mean_squared_error, classification_report, confusion_matrix


rfr_params = {'max_depth': [3,5],
              'max_features': [5,6,7],
              'n_estimators': [200]}

gbr_params = {'max_features': [5,6,7],
              'learning_rate':[0.1],
              'n_estimators':[200],
              'subsample':[0.6],
              'random_state':[42]}

lgb_clf_params ={
    'objective':'multiclass',
    'num_class':3,
    'metric':'multi_logloss',
    'num_leaves': 31,
    'max_depth': -1,
    'learning_rate':  0.01,
    'n_estimators': 100,
    "verbose": 1,
}

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
        self.eval = None
        self.clf_train, self.clf_test = None, None
        self.feats = features
        self.model = None
        self.train_predict, self.test_predict = \
            [None] * 2  # Vars for future use

        self.params = {}
        self.x_final = data[features].tail(50)

        self.target_data = data[target_column]
        self.train_size = 0.8


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


        if not gbr:
            model = RandomForestRegressor(n_estimators=200)
            name = 'rfr'
        else:
            model = GradientBoostingRegressor(n_estimators=200)
            name = 'gbr'

        test_scores = []
        rmse_scores = []

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

    def lgb_clf(self, params ,plot_importances=True,high_percentile=80, low_percentile=20,num_rounds=100  ):
        self.clf_train, self.clf_test = create_label_dataset(self.target_data,len(self.y_train),high_percentile, low_percentile)
        train_data = lgb.Dataset(self.x_train, label=self.clf_train, free_raw_data=False)
        valid_data = lgb.Dataset(self.x_test, label=self.clf_test, reference=train_data, free_raw_data=False)
        self.model = lgb.train(params, num_boost_round=num_rounds, callbacks=[lgb.early_stopping(10)],train_set=train_data,valid_sets=[valid_data])
        y_prob = self.model.predict(self.x_test, num_iteration=self.model.best_iteration)
        y_pred = np.argmax(y_prob, axis=1)
        self.eval  = evaluate_clf(y_pred, y_prob, self.clf_test, sorted_features=True)
        print(classification_report(self.clf_test, y_pred, labels=[0, 1, 2]))



        return self.model

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

    def neighbors_clf(self, k_start, k_end, evaluate=True, high_p=80, low_p=20):
        best_acc = 0

        self.clf_train, self.clf_test = create_labels(self.y_train, high_p, low_p), create_labels(self.y_test, high_p, low_p)

        results_df = pd.DataFrame(columns=['accuracy', 'log_loss'], index=np.arange(k_start, k_end+1))
        for k in range(k_start, k_end):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.x_train, self.clf_train)
            acc = knn.score(self.x_test, self.clf_test)
            if acc > best_acc: best_acc = acc
            results_df.loc[k, 0] = acc
            results_df.loc[k, 1] = log_loss(self.clf_test, knn.predict_proba(self.x_test))
            print(f'K : {k} ; Accuracy: {acc} ; Log Loss : {results_df.loc[k, 1]}')

        best_k = results_df.loc[results_df['accuracy'] == best_acc].index[0]
        knn.set_params(**{'n_neighbors':best_k})
        self.model = knn

        return results_df


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
            print(self.eval)
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






