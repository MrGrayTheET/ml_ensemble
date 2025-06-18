import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ml_build.utils import (clean_data, TimeSeriesCV,
                            evaluate_model,
                            evaluate_clf,
                            create_labels, create_label_dataset,
                            save_model, load_model, get_fi, format_time)

from feature_engineering import extract_time_features
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb
from time import time
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from scipy.stats import spearmanr
from itertools import product
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, log_loss, \
    root_mean_squared_error, classification_report, confusion_matrix

YEAR = 252
rfr_params = {'max_depth': [3, 5],
              'max_features': [5, 6, 7],
              'n_estimators': [200]}

gbr_params = {'max_features': [5, 6, 7],
              'learning_rate': [0.1],
              'n_estimators': [200],
              'subsample': [0.6],
              'random_state': [42]}

lgb_clf_params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'num_leaves': 31,
    'max_depth': -1,
    'learning_rate': 0.01,
    'n_estimators': 100,
    "verbose": 1,
}
max_depths = [2, 3, 5, 7]
lgb_reg_params = {
    'learning_rate': [0.01, .1, .3],
    'num_leaves': [2 ** i for i in max_depths],
    'feature_fraction': [.3, .6, .95],
    'min_data_in_leaf': [25, 50, 70]
}

xgb_params = {
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'subsample': [0.8, 0.9, 1.0],
    'n_estimators': [200],
    'max_leaves': [5, 7, 8]

}

rf_clf_params = {'n_estimators': [200],
                 'subsample': [0.6],
                 'random_state': [42],
                 'learning_rate': [0.01, 0.05, 0.1, 0.2],
                 'max_depth': [3, 5, 7]}

xgb_clf_params = dict(n_estimators=[200],
                      max_depth=[3, 5, 7],
                      learning_rate=[0.01, 0.05, 0.1, 0.2],
                      subsample=[0.6, 0.8],
                      random_state=[42])


class ml_model:

    def __init__(self, data, features, target_column, train_test_size=0.8, scale_x=False,
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

        [self.x_train, self.y_train], [self.x_test, self.y_test] = \
            clean_data(data, features, target_column, scale_x=scale_x, scale_y=scale_y, train_split=True,
                       train_size=train_test_size)

    def plot_predictions(self, training_pred=False, test_pred=True):
        if training_pred and test_pred:
            fig, ax = plt.subplots(2)
            ax[0].scatter(self.y_train, self.train_predict, label='train')
            ax[1].scatter(self.y_test, self.test_predict, label='test')

        elif training_pred and not test_pred:
            plt.scatter(self.y_train, self.train_predict)

        elif test_pred and not training_pred:
            plt.scatter(self.y_test, self.test_predict)

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
            self.eval = evaluate_model(self.test_predict, self.y_test, self.feats, sorted_features=True,
                                       log_file=eval_log)

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

    def lgb_clf(self, params, plot_importances=True, high_percentile=80, low_percentile=20, num_rounds=100):
        self.clf_train, self.clf_test = create_label_dataset(self.target_data, len(self.y_train), high_percentile,
                                                             low_percentile)
        train_data = lgb.Dataset(self.x_train, label=self.clf_train, free_raw_data=False)
        valid_data = lgb.Dataset(self.x_test, label=self.clf_test, reference=train_data, free_raw_data=False)
        self.model = lgb.train(params, num_boost_round=num_rounds, callbacks=[lgb.early_stopping(10)],
                               train_set=train_data, valid_sets=[valid_data])
        y_prob = self.model.predict(self.x_test, num_iteration=self.model.best_iteration)
        y_pred = np.argmax(y_prob, axis=1)
        self.eval = evaluate_clf(y_pred, y_prob, self.clf_test, sorted_features=True)
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

        self.clf_train, self.clf_test = create_labels(self.y_train, high_p, low_p), create_labels(self.y_test, high_p,
                                                                                                  low_p)

        results_df = pd.DataFrame(columns=['accuracy', 'log_loss'], index=np.arange(k_start, k_end + 1))
        for k in range(k_start, k_end):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.x_train, self.clf_train)
            acc = knn.score(self.x_test, self.clf_test)
            if acc > best_acc: best_acc = acc
            results_df.loc[k, 0] = acc
            results_df.loc[k, 1] = log_loss(self.clf_test, knn.predict_proba(self.x_test))
            print(f'K : {k} ; Accuracy: {acc} ; Log Loss : {results_df.loc[k, 1]}')

        best_k = results_df.loc[results_df['accuracy'] == best_acc].index[0]
        knn.set_params(**{'n_neighbors': best_k})
        self.model = knn

        return results_df

    def xgb_model(self, param_dict, plot_pred=True, plot_importances=True, save_params=True,
                  params_file='xgb_params.csv', evaluate=True, eval_log='xgb_eval.csv'):
        test_scores = []
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
            self.eval = evaluate_model(self.test_predict, self.y_test, self.feats, log_file=eval_log,
                                       sorted_features=True)
            print(self.eval)
        else:
            pass

        return self.model

    def xgb_clf(self, high_p: int, low_p: int, parameter_dict, plot_pred=False, plot_importances=True, evaluate=True):
        '''high_p:int percentile used to decide the 1 class
            low_p: int percentile used to decide the -1 class
            paramter_dict:dict hyperparameter dictionary for GridSearchCV
        '''
        test_scores = []
        acc_scores = []

        self.clf_train, self.clf_test = create_labels(self.y_train, high_p, low_p), create_labels(self.y_test, high_p,
                                                                                                  low_p)

        # initiate grid_search
        grid_search = GridSearchCV(XGBClassifier(), parameter_dict, scoring='accuracy', cv=3)

        # Create labels for training

        grid_search.fit(self.x_train, self.clf_train)

        # Fit and test model
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
                model_eval = evaluate_clf(self.test_predict, self.test_probs, self.clf_test, self.feats)
                print(model_eval)
                self.eval = model_eval

        return self.model

    def save_model(self, file_name: str, save_fp='F:\\ML\\models\\'):
        save_model(self.model, save_fp + file_name)
        print(f'Model successfully Saved to {save_fp + file_name}')
        return

    def load_model(self, model_fp):
        model = load_model(model_fp)
        self.model = model
        return self.model

    def forecast_data(self, n_periods_back=10):

        return self.model.predict(self.x_final[-n_periods_back:])


# noinspection PyTypeChecker
class lgb_optimizer:
    # noinspection PyDefaultArgument

    def __init__(self, dataset,
                 feature_cols,load_existing=False,
                 store_name='model.h5',
                 results_path='F:\\ML\\intraday\\',
                 date_col='date',
                 date_features=['month', 'weekday'],
                 year_col = 'year',
                 categoricals=None,
                 param_dict=None,
                 label_filter='target'):

        self.ic_group_col = None
        self.scope_params = ['lookahead', 'train_length', 'test_length']
        param_dict = lgb_reg_params if param_dict is None else param_dict
        self.param_names = [*param_dict.keys()]
        self.results_path = results_path
        self.lgb_store = Path(results_path, store_name)
        self.num_iterations = [10, 25, 50, 75] + list(
            range(100, 501, 50)) if 'num_iterations' not in param_dict.keys() else param_dict['num_iterations']
        self.num_boost_round = self.num_iterations[-1]
        self.labels = sorted(dataset.filter(like=label_filter).columns)
        self.metric_cols = ([*param_dict.keys()] + ['t', 'daily_ic_mean', 'daily_ic_mean_n', 'daily_ic_median',
                                                    'daily_ic_median_n'] + [str(n) for n in self.num_iterations])

        self.ic_cols = self.metric_cols[5:9]

        self.model = None
        self.cv_params = list(product(*param_dict.values()))
        self.n_params = len(self.cv_params)
        self.label_dict = None
        self.base_params = dict(boosting='gbdt',
                                objective='regression',
                                verbose=-1)
        self.date_idx = [date_col]

        if isinstance(dataset.index, pd.DatetimeIndex):
            self.data = extract_time_features(data=dataset, date_cols=self.date_idx + date_features + [year_col], reset_index=True)
        else:
            self.data = dataset

        self.categoricals = date_features if categoricals is None else categoricals + date_features
        self.date_features = date_features
        self.feature_cols = feature_cols + self.categoricals
        for feature in self.categoricals + self.date_idx + ['year']:
            self.data[feature] = pd.factorize(self.data[feature], sort=True)[0]

        self.keys = []

        print(self.data.info())
        print(f'Categoricals:{self.categoricals}\n')
        return

    def set_cv_params(self, lookaheads=[0], train_lengths=[180, 252], test_lengths=[63]):

        self.test_params = list(product(lookaheads, train_lengths, test_lengths))
        n = len(self.test_params)
        test_param_sample = np.random.choice(list(range(n)), size=int(n), replace=False)
        self.test_params = [self.test_params[i] for i in test_param_sample]
        print('Train Configs:', len(self.test_params))
        if len(lookaheads) < len(self.labels):
            self.label_dict = dict(zip(range(len(self.labels)), self.labels))
        else:
            self.label_dict = dict(zip(lookaheads, self.labels))


        return

    def run_CV(self, group_ic_by=['year','month'], cv_lookaheads=False):
        cv_params = self.cv_params
        n_params = len(self.cv_params)
        label_dict = self.label_dict
        data = self.data
        date_idx = self.date_idx
        features = self.feature_cols
        categoricals = self.categoricals
        base_params = self.base_params
        param_names = self.param_names
        num_boost_round = self.num_boost_round
        num_iterations = self.num_iterations
        self.keys = []
        self.ic_group_col = group_ic_by

        for lookahead, train_length, test_length in self.test_params:
            cvp = np.random.choice(list(range(n_params)),
                                   size=int(n_params / 2),
                                   replace=False)
            cv_params_ = [cv_params[i] for i in cvp]

            # TimeSeries cv
            n_splits = int(2 * 252 / test_length)
            print(f'Lookahead: {lookahead:2.0f} | '
                  f'Train: {train_length:3.0f} | '
                  f'Test: {test_length:2.0f} | '
                  f'Params: {len(cv_params_):3.0f} | '
                  f'Train configs: {len(self.test_params)}')

            cv = TimeSeriesCV(n_splits, train_length, test_length, lookahead=0)

            label = self.label_dict[lookahead]
            label = [label]
            outcome_data = data.loc[:, features + label + date_idx + ['year']].dropna()
            lgb_data = lgb.Dataset(data=outcome_data.drop(columns=label+date_idx+['year']),
                                   label=outcome_data[label],
                                   categorical_feature=categoricals,
                                   )
            print(lgb_data.data.info())
            T = 0
            predictions, metrics, feature_importance, rolling_ic = [], [], [], []

            for p, param_vals in enumerate(cv_params_):
                key = f'{lookahead}/{train_length}/{test_length}/' + '/'.join([str(p) for p in param_vals])
                params = dict(zip(param_names, param_vals))
                params.update(base_params)

                start = time()
                cv_preds, nrounds = [], []
                ic_cv = defaultdict(list)

                # iterate over folds
                for i, (train_idx, test_idx) in enumerate(cv.split(X=outcome_data)):

                    # select train subset
                    lgb_train = lgb_data.subset(used_indices=train_idx,
                                                params=params).construct()
                    # train model for num_boost_round
                    model = lgb.train(params=params,
                                      train_set=lgb_train,
                                      num_boost_round=num_boost_round)

                    # log feature importance
                    if i == 0:
                        fi = get_fi(model).to_frame()
                    else:
                        fi[i] = get_fi(model)

                    # capture prediction
                    test_set = outcome_data.iloc[test_idx[0]:test_idx[-1], :]
                    X_test = test_set.loc[:, features]
                    y_test = test_set.loc[:, label]
                    y_pred = {str(n): model.predict(X_test, num_iteration=n) for n in num_iterations}

                    # record predictions for each fold
                    cv_preds.append(y_test.assign(**y_pred).assign(**test_set[self.date_features + self.date_idx + ['year']]))

                cv_preds = pd.concat(cv_preds).assign(**params)
                predictions.append(cv_preds)

                # compute IC by period

                by_period = cv_preds.groupby(self.ic_group_col)

                ic_by_period = pd.concat([by_period.apply(lambda x: spearmanr(x[label], x[str(n)])[0]).to_frame(n)
                                       for n in num_iterations], axis=1)

                period_ic_mean = ic_by_period.mean()
                period_ic_mean_n = period_ic_mean.idxmax()
                period_ic_median = ic_by_period.median()
                period_ic_median_n = period_ic_median.idxmax()

                # compute IC across all predictions
                ic = [spearmanr(cv_preds[label], cv_preds[str(n)])[0] for n in num_iterations]
                t = time() - start
                T += t

                # collect metrics
                metrics = pd.Series(list(param_vals) +
                                    [t, period_ic_mean.max(), period_ic_mean_n, period_ic_median.max(),
                                     period_ic_median_n] + ic,
                                    index=self.metric_cols)

                msg = f'\t{p:3.0f} | {format_time(T)} ({t:3.0f}) | {params["learning_rate"]:5.2f} | '
                msg += f'{params["num_leaves"]:3.0f} | {params["feature_fraction"]:3.0%} | {params["min_data_in_leaf"]:4.0f} | '
                msg += f' {max(ic):6.2%} | {ic_by_period.mean().max(): 6.2%} | {period_ic_mean_n: 4.0f} | {ic_by_period.median().max(): 6.2%} | {period_ic_median_n: 4.0f}'
                print(msg)

                # persist results for given CV run and hyperparameter combination
                metrics.to_hdf(self.lgb_store, 'metrics/' + key)
                ic_by_period.assign(**params).to_hdf(self.lgb_store, 'daily_ic/' + key)
                fi.T.describe().T.assign(**params).to_hdf(self.lgb_store, 'fi/' + key)
                cv_preds.to_hdf(self.lgb_store, 'predictions/' + key)

        return
    def evaluate_results(self, results_store_file='lgb_eval1.h5', top_n=10, return_all_ic_groups=False):
        lgb_train_params = self.param_names
        eval_summary = {}
        eval_res_path = Path(self.results_path, results_store_file)
        if not eval_res_path.exists():
            lgb_metrics = self._get_metrics(store_fp=eval_res_path)
            lgb_ic = self._get_ic(store_file=eval_res_path)
            lgb_daily_ic = lgb_ic.groupby(self.scope_params + self.param_names + ['boost_rounds']).ic.mean().to_frame(
                'ic').reset_index()
            lgb_daily_ic.to_hdf(eval_res_path, 'lgb/daily_ic')
        else:
            with pd.HDFStore(eval_res_path) as eval_res:
                lgb_daily_ic = eval_res['lgb/daily_ic']
                lgb_ic = eval_res['lgb/ic'] if return_all_ic_groups else []
                lgb_metrics = eval_res['lgb/metrics']

        eval_summary['Metrics'] = lgb_metrics.sort_values(by='ic', ascending=False)
        eval_summary['lgb_ic'] = lgb_ic if return_all_ic_groups else []
        eval_summary['lgb_ic_total'] = lgb_daily_ic.sort_values(by='ic', ascending=False)
        self.summary = eval_summary
        top_params = lgb_daily_ic.groupby('lookahead', group_keys=False).apply(lambda x: x.nlargest(top_n, 'ic'))

        lookaheads = lgb_daily_ic.lookahead.unique().tolist()
        int_cols = ['boost_rounds', 'train_length', 'test_length', 'num_leaves', 'lookahead']

        for n in lookaheads:
            best_params = self.get_lgb_params(top_params, t=n, best=0)
            eval_summary[n]= dict(params=best_params,
                                  predictions=self._get_predictions(params=best_params, lookahead=n),
                                  feature_importance=self._get_fi(params=best_params, lookahead=n))
            rounds = str(best_params.boost_rounds)


        return eval_summary



    def get_lgb_params(self, data, t=0, best=0):
        int_cols = ['boost_rounds', 'train_length', 'test_length', 'num_leaves', 'lookahead']
        param_cols = self.scope_params[1:] + self.param_names + ['boost_rounds']
        df = data[data.lookahead == t].sort_values('ic', ascending=False).iloc[best]
        df[int_cols] = df[int_cols].astype(int)
        return df

    def get_key(self, t, p):
        key = f'{t}/{int(p.train_length)}/{int(p.test_length)}/{p.learning_rate}/'
        return key + f'{int(p.num_leaves)}/{p.feature_fraction}/{int(p.min_data_in_leaf)}'

    def _select_ic(self, params, ic_data, lookahead):
        filtered_ic =  ic_data.loc[(ic_data.lookahead == lookahead) &
                           (ic_data.train_length == params.train_length) &
                           (ic_data.test_length == params.test_length) &
                           (ic_data.learning_rate == params.learning_rate) &
                           (ic_data.num_leaves == params.num_leaves) &
                           (ic_data.feature_fraction == params.feature_fraction) &
                           (ic_data.boost_rounds == params.boost_rounds)]
        ic = filtered_ic.sort_values(self.ic_group_col)
        ic['ym'] = ic['year'].astype(str) + ic['month'].astype(str).str.zfill(2)

        return ic

    def _get_predictions(self, params, lookahead):
        key = self.get_key(lookahead, params)
        predictions = pd.read_hdf(self.lgb_store, 'predictions/'+key)
        return predictions

    def _get_fi(self, params, lookahead):
        key = self.get_key(lookahead, params)
        fi = pd.read_hdf(self.lgb_store, 'fi/'+key)
        return fi


    def _get_ic(self, store_file=None):
        lgb_ic = []
        int_cols = ['lookahead', 'train_length', 'test_length', 'boost_rounds']
        store_file = Path(self.results_path,str(time())[-3:]+'results.h5' ) if store_file is None else store_file #Time element added as a pseudo-number generator to avoid overwriting existing files/tables

        with pd.HDFStore(self.lgb_store) as store:
            keys = [k[1:] for k in store.keys() if len(k.split('/')) >3 ]
            for key in keys:
                _, t, train_length, test_length = key.split('/')[:4]
                if key.startswith('daily_ic'):
                    df = (store[key]
                          .drop(['boosting', 'objective', 'verbose'], axis=1)
                          .assign(lookahead=t,
                                  train_length=train_length,
                                  test_length=test_length))
                    lgb_ic.append(df)
            lgb_ic = pd.concat(lgb_ic).reset_index()
            id_vars = ['month', 'year'] + self.scope_params + self.param_names
            lgb_ic = pd.melt(lgb_ic,
                             id_vars=id_vars,
                             value_name='ic',
                             var_name='boost_rounds').dropna()
            lgb_ic.loc[:, int_cols] = lgb_ic.loc[:, int_cols].astype(int)
            lgb_ic.to_hdf(store_file, 'lgb/ic')
            lgb_ic.info()


            return lgb_ic

    def _get_metrics(self, store_fp=None,):
        with pd.HDFStore(self.lgb_store) as store:
            for i, key in enumerate([k[1:] for k in store.keys() if k[1:].startswith('metrics')]):
                _, t, train_length, test_length = key.split('/')[:4]
                attrs = {
                    'lookahead': t,
                    'train_length': train_length,
                    'test_length': test_length,
                }
                s = store[key].to_dict()
                s.update(attrs)
                if i == 0:
                    lgb_metrics = pd.Series(s).to_frame(i)
                else:
                    lgb_metrics[i] = pd.Series(s)
            id_vars = self.scope_params + self.param_names + self.ic_cols
            lgb_metrics = pd.melt(lgb_metrics.T.drop('t', axis=1),
                                  id_vars=id_vars,
                                  value_name='ic',
                                  var_name='boost_rounds').dropna().apply(pd.to_numeric)
            if store_fp is None:
                lgb_metrics.to_hdf(Path(self.results_path, 'results.h5'), 'lgb/metrics')
            else:
                lgb_metrics.to_hdf(store_fp, 'lgb/metrics')
            lgb_metrics.info()
            lgb_metrics.groupby(self.scope_params)

            return lgb_metrics







