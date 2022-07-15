import numpy as np
import pandas as pd
import datetime
import warnings
import pickle
import gc
import os
import yaml

from sklearn.model_selection import StratifiedKFold
from utils import *
import lightgbm as lgb
import optuna
import pickle
from train import *

warnings.filterwarnings('ignore')


class Dart_LGBM_baseline(LGBM_baseline):

    def tuning(self) -> dict:
        num_trial = self.CFG['OPTUNA_num_trial']
        num_boost_round = self.CFG['num_boost_round']
        only_first_fold = self.CFG['OPTUNA_only_first_fold']
        early_stopping = self.CFG['OPTUNA_early_stopping_rounds']

        self.STDOUT('[Optuna parameter tuning]')
        kf = StratifiedKFold(n_splits=5)

        def objective(trial):
            score_list = []
            start_time = datetime.datetime.now()
            self.STDOUT(f'[ Trial {trial._trial_id} Start ]')

            for fold, (idx_tr, idx_va) in enumerate(kf.split(self.train, self.target)):
                train_x = self.train.iloc[idx_tr][self.features]
                valid_x = self.train.iloc[idx_va][self.features]
                train_y = self.target[idx_tr]
                valid_y = self.target[idx_va]
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
                                num_boost_round=num_boost_round,
                                early_stopping_rounds=early_stopping,
                                verbose_eval=False,
                                feval = [custom_accuracy])
                preds = gbm.predict(valid_x)
                score = amex_metric(valid_y, preds)
                score_list.append(score)

                if only_first_fold: break

            cv_score = sum(score_list) / len(score_list)
            self.STDOUT(f"CV Score | {str(datetime.datetime.now() - start_time)[-12:-7]} | Score = {cv_score:.5f}")
            return cv_score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=num_trial)
        best_params = study.best_params
        self.STDOUT(best_params)
        return best_params


    def train_model(self) -> None:
        eval_interval = self.CFG['eval_interval']
        num_boost_round = self.CFG['num_boost_round']
        only_first_fold = self.CFG['only_first_fold']
        early_stopping = self.CFG['early_stopping_rounds']
        score_list = []
        kf = StratifiedKFold(n_splits=5)
        for fold, (idx_tr, idx_va) in enumerate(kf.split(self.train, self.target)):
            des = DartEarlyStopping("valid_0", 'custom_accuracy', early_stopping)
            train_x = self.train.iloc[idx_tr][self.features]
            valid_x = self.train.iloc[idx_va][self.features]
            train_y = self.target[idx_tr]
            valid_y = self.target[idx_va]
            dtrain = lgb.Dataset(train_x, label=train_y)
            dvalid = lgb.Dataset(valid_x, label=valid_y)

            gbm = lgb.train(self.best_params, dtrain, valid_sets=[dvalid], 
                            num_boost_round=num_boost_round,
                            callbacks=[lgb.log_evaluation(eval_interval),des],
                            feval = [custom_accuracy])
            gbm = des.best_model
            preds = gbm.predict(valid_x)
            score = amex_metric(valid_y, preds)
            score_list.append(score)
            file = self.output_dir + f'/model_fold{fold}.pkl'
            pickle.dump(gbm, open(file, 'wb'))
            if only_first_fold: break 

        self.STDOUT(f"OOF Score: {np.mean(score_list):.5f}")


def main():
    with open('config.yml', 'r', encoding="utf-8_sig") as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)

    os.makedirs(CFG['output_dir'], exist_ok=True)
    LOGGER = None
    if CFG['log'] == True:
        LOGGER = init_logger(log_file=CFG['output_dir']+'/train.log')

    baseline = Dart_LGBM_baseline(CFG, LOGGER)
    baseline.train_model()
    baseline.save_features()

if __name__ == '__main__':
    main()