from random import triangular
import numpy as np
import pandas as pd
import datetime
import warnings
import pickle
from colorama import Fore, Back, Style
import gc
import os
import yaml

from sklearn.model_selection import StratifiedKFold
from utils import *
import lightgbm as lgb
import optuna

warnings.filterwarnings('ignore')


def amex_metric(y_true: np.array, y_pred: np.array) -> float:

    # count of positives and negatives
    n_pos = y_true.sum()
    n_neg = y_true.shape[0] - n_pos

    # sorting by descring prediction values
    indices = np.argsort(y_pred)[::-1]
    preds, target = y_pred[indices], y_true[indices]

    # filter the top 4% by cumulative row weights
    weight = 20.0 - target * 19.0
    cum_norm_weight = (weight / weight.sum()).cumsum()
    four_pct_filter = cum_norm_weight <= 0.04

    # default rate captured at 4%
    d = target[four_pct_filter].sum() / n_pos

    # weighted gini coefficient
    lorentz = (target / n_pos).cumsum()
    gini = ((lorentz - cum_norm_weight) * weight).sum()

    # max weighted gini coefficient
    gini_max = 10 * n_neg * (1 - 19 / (n_pos + 20 * n_neg))

    # normalized weighted gini coefficient
    g = gini / gini_max

    return 0.5 * (g + d)

def custom_accuracy(preds, data):
    y_true = data.get_label()
    acc = amex_metric(y_true, preds)
    return ('custom_accuracy', acc, True)


def tuning(train, target, features, CFG, LOGGER = None):
    LOGGER.info('[Optuna parameter tuning]')
    kf = StratifiedKFold(n_splits=5)

    def objective(trial):
        score_list = []
        start_time = datetime.datetime.now()
        global trial_number
        trial_number += 1
        LOGGER.info(f'[ Trial {trial_number} Start ]')

        for fold, (idx_tr, idx_va) in enumerate(kf.split(train, target)):
            train_x = train.iloc[idx_tr][features]
            valid_x = train.iloc[idx_va][features]
            train_y = target[idx_tr]
            valid_y = target[idx_va]
            dtrain = lgb.Dataset(train_x, label=train_y)
            dvalid = lgb.Dataset(valid_x, label=valid_y)

            param = {
                'objective': 'binary',
                'metric': 'custom',
                'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                'num_leaves': trial.suggest_int('num_leaves', 100, 400),
                'learning_rate': trial.suggest_uniform('learning_rate', 0.005, 0.1),
                #'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
                #'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                #'bagging_freq': trial.suggest_int('bagging_freq', 1, 30),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'min_child_samples': trial.suggest_int('min_child_samples', 100, 3000),
                'force_col_wise': True
            }
    
            gbm = lgb.train(param, dtrain, valid_sets=[dvalid], 
                            num_boost_round=CFG['num_boost_round'],
                            early_stopping_rounds=50,
                            verbose_eval=False,
                            feval = [custom_accuracy])
            preds = gbm.predict(valid_x)
            score = amex_metric(valid_y, preds)
            score_list.append(score)

            if CFG['ONLY_FIRST_FOLD']: break # we only want the first fold

        cv_score = sum(score_list) / len(score_list)
        if LOGGER:
            LOGGER.info(f"CV Score | {str(datetime.datetime.now() - start_time)[-12:-7]} | Score = {cv_score:.5f}")
        else:
            print(f"{Fore.GREEN}{Style.BRIGHT}CV Score | {str(datetime.datetime.now() - start_time)[-12:-7]} |"
                  f" Score = {cv_score:.5f}{Style.RESET_ALL}")

        return cv_score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=CFG['optuna_trial'])
    best_params = study.best_params
    LOGGER.info(best_params)
    return best_params

def train_model(train, target, features, best_param, CFG, LOGGER = None):
    score_list = []
    kf = StratifiedKFold(n_splits=5)
    for fold, (idx_tr, idx_va) in enumerate(kf.split(train, target)):
        train_x = train.iloc[idx_tr][features]
        valid_x = train.iloc[idx_va][features]
        train_y = target[idx_tr]
        valid_y = target[idx_va]
        
        dtrain = lgb.Dataset(train_x, label=train_y)
        dvalid = lgb.Dataset(valid_x, label=valid_y)

        gbm = lgb.train(best_param, dtrain, valid_sets=[dvalid], 
                        num_boost_round=CFG['num_boost_round'],
                        callbacks=[lgb.log_evaluation(10)],
                        feval = [custom_accuracy])

        preds = gbm.predict(valid_x)
        score = amex_metric(valid_y, preds)
        score_list.append(score)
        file = CFG['logdir'] + f'/model_fold{fold}.pkl'
        pickle.dump(gbm, open(file, 'wb'))
    if LOGGER:
        LOGGER.info(f"OOF Score: {np.mean(score_list):.5f}")
    else:
        print(f"{Fore.GREEN}{Style.BRIGHT}OOF Score: {np.mean(score_list):.5f}{Style.RESET_ALL}")

def main():
    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)

    os.makedirs(CFG['logdir'], exist_ok=True)
    LOGGER = None
    if CFG['log'] == True:
        LOGGER = init_logger(log_file=CFG['logdir']+'/train.log')

    train = create_train(CFG['features_path'], CFG['using_features'], LOGGER)
    target = pd.read_csv(CFG['label_pth']).target.values
    features = train.columns
    #オンオフ
    best_params = {}
    best_params = tuning(train, target, features, CFG, LOGGER)
    best_params['force_col_wise'] = True
    train_model(train, target, features, best_params, CFG, LOGGER)

if __name__ == '__main__':
    # global variable
    trial_number = 0
    main()