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


class LGBM_baseline():
    def __init__(self, CFG, logger = None) -> None:
        self.STDOUT = set_STDOUT(logger)
        self.CFG = CFG
        self.features_path = CFG['features_path']
        self.feature_groups_path = CFG['feature_groups_path']
        self.using_features = CFG['using_features']
        self.output_dir = CFG['output_dir']

        self.train = self.create_train()
        self.target = self.create_target()
        self.features = self.train.columns
        self.best_params =  {'objective': 'binary', 'metric': 'custom'}
        if CFG['use_optuna']:
            self.best_params = self.tuning()
        if CFG['use_custom_params']:
            assert type(CFG['custom_params']) is dict
            self.best_params = CFG['custom_params']
        self.best_params['force_col_wise'] = True

        self.STDOUT(f'[ USING CUSTOM PARAMS ]')
        self.STDOUT('=' * 30)
        for key, value in self.best_params.items():
            self.STDOUT(f'{key} : {value}')
        self.STDOUT('=' * 30)


    def save_features(self) -> None:
        features_dict = {}
        
        for dirname, feature_name in self.using_features.items():
            if feature_name == 'all':
                feature_name = glob.glob(self.features_path + f'/{dirname}/train/*')
                feature_name = [os.path.splitext(os.path.basename(F))[0]
                 for F in feature_name if 'customer_ID' not in F]

            elif type(feature_name) == str:
                file = self.feature_groups_path + f'/{dirname}/{feature_name}.txt'
                feature_name = []
                with open(file, 'r') as f:
                        for line in f:
                            feature_name.append(line.rstrip("\n"))

            features_dict[dirname] = []
            for name in feature_name:
                features_dict[dirname].append(name)

        file = self.output_dir + f'/features.pkl'
        pickle.dump(features_dict, open(file, 'wb'))
        self.STDOUT(f'successfully saved feature names')


    def create_train(self) -> pd.DataFrame:
        df_dict = {}
        for dirname, feature_name in self.using_features.items():
            if feature_name == 'all':
                feature_name = glob.glob(self.features_path + f'/{dirname}/train/*')
                feature_name = [os.path.splitext(os.path.basename(F))[0] 
                                for F in feature_name if 'customer_ID' not in F]

            elif type(feature_name) == str:
                file = self.feature_groups_path + f'/{dirname}/{feature_name}.txt'
                feature_name = []
                with open(file, 'r') as f:
                        for line in f:
                            feature_name.append(line.rstrip("\n"))

            # oofを生成する都合でcustomer_id を入れる
            feature_name.insert(0, 'customer_ID')

            for name in feature_name:
                filepath = self.features_path + f'/{dirname}/train' + f'/{name}.pickle'
                one_df = pd.read_pickle(filepath)
                df_dict[one_df.name] = one_df.values
                self.STDOUT(f'loading : {name} of {dirname}')
        df = pd.DataFrame(df_dict)
        self.STDOUT(f'dataframe_info:  {len(df)} rows, {len(df.columns)} features')
        return df


    def create_target(self) -> pd.Series:
        label_pth = self.CFG['label_pth']
        target = pd.read_csv(label_pth).target.values
        return target


    def tuning(self) -> dict:
        num_trial = self.CFG['OPTUNA_num_trial']
        num_boost_round = self.CFG['OPTUNA_num_boost_round']
        only_first_fold = self.CFG['OPTUNA_only_first_fold']
        early_stopping = self.CFG['OPTUNA_early_stopping_rounds']
        boosting_type = self.CFG['OPTUNA_boosting_type']

        self.STDOUT('[Optuna parameter tuning]')
        kf = StratifiedKFold(n_splits=5)

        def objective(trial):
            score_list = []
            start_time = datetime.datetime.now()
            self.STDOUT(f'[ Trial {trial._trial_id} Start ]')

            for fold, (idx_tr, idx_va) in enumerate(kf.split(self.train, self.target)):
                train_x = self.train.iloc[idx_tr][self.features[1:]]
                valid_x = self.train.iloc[idx_va][self.features[1:]]
                train_y = self.target[idx_tr]
                valid_y = self.target[idx_va]
                dtrain = lgb.Dataset(train_x, label=train_y)
                dvalid = lgb.Dataset(valid_x, label=valid_y)

                param = {
                    'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
                    'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
                    'num_leaves': trial.suggest_int('num_leaves', 100, 400),
                    #'learning_rate': trial.suggest_uniform('learning_rate', 0.005, 0.1),
                    'feature_fraction': trial.suggest_uniform('feature_fraction', 0.1, 0.4),
                    #'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
                    #'bagging_freq': trial.suggest_int('bagging_freq', 1, 30),
                    #'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'min_child_samples': trial.suggest_int('min_child_samples', 100, 3000),
                    'force_col_wise': True,
                    'boosting' : boosting_type
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
        only_first_fold = self.CFG['only_first_fold']
        score_list = []
        kf = StratifiedKFold(n_splits=5)
        oofs = []
        for fold, (idx_tr, idx_va) in enumerate(kf.split(self.train, self.target)):
            train_x = self.train.iloc[idx_tr][self.features]
            valid_x = self.train.iloc[idx_va][self.features]
            train_y = self.target[idx_tr]
            valid_y = self.target[idx_va]
            dtrain = lgb.Dataset(train_x.drop(columns = 'customer_ID'), label=train_y)
            dvalid = lgb.Dataset(valid_x.drop(columns = 'customer_ID'), label=valid_y)

            gbm = lgb.train(self.best_params, dtrain, valid_sets=[dvalid], 
                            callbacks=[lgb.log_evaluation(eval_interval)],
                            feval = [custom_accuracy])

            preds = gbm.predict(valid_x.drop(columns = 'customer_ID'))
            score = amex_metric(valid_y, preds)
            score_list.append(score)
            # saving models
            file = self.output_dir + f'/model_fold{fold}.pkl'
            pickle.dump(gbm, open(file, 'wb'))

            # creating oof predictions
            preds_df = pd.DataFrame(preds, columns = ['predicted'])
            preds_09 = pd.DataFrame(np.where(preds > 0.9, 1, 0), columns = ['predicted_09'])
            preds_08 = pd.DataFrame(np.where(preds > 0.8, 1, 0), columns = ['predicted_08'])
            preds_07 = pd.DataFrame(np.where(preds > 0.7, 1, 0), columns = ['predicted_07'])
            preds_06 = pd.DataFrame(np.where(preds > 0.6, 1, 0), columns = ['predicted_06'])
            preds_05 = pd.DataFrame(np.where(preds > 0.5, 1, 0), columns = ['predicted_05'])
            oof = pd.concat([valid_x.reset_index(drop = True), preds_df, preds_09,
                            preds_08, preds_07, preds_06, preds_05], axis = 1)
            oof['fold'] = fold
            oofs.append(oof)

            if self.CFG['show_importance']:
                importance = pd.DataFrame(gbm.feature_importance(), index=self.features[1:], 
                                          columns=['importance']).sort_values('importance',ascending=False)
                importance.to_csv(self.output_dir + f'/importance_fold{fold}.csv')           

            if only_first_fold: break 
        
        # saving oofs
        if self.CFG['create_oofs']:
            oofs = pd.concat(oofs).reset_index()
            oofs.to_feather(self.output_dir + '/oofs.ftr')
            self.STDOUT('=' * 30)
            self.STDOUT('OOFs Info')
            f, t = oofs['predicted_09'].value_counts()
            self.STDOUT(f'threshold 0.9  True:{t} False:{f}')
            f, t = oofs['predicted_08'].value_counts()
            self.STDOUT(f'threshold 0.8  True:{t} False:{f}')
            f, t = oofs['predicted_07'].value_counts()
            self.STDOUT(f'threshold 0.7  True:{t} False:{f}')
            f, t = oofs['predicted_06'].value_counts()
            self.STDOUT(f'threshold 0.6  True:{t} False:{f}')
            f, t = oofs['predicted_05'].value_counts()
            self.STDOUT(f'threshold 0.5  True:{t} False:{f}')
            self.STDOUT('=' * 30)

        self.STDOUT(f"OOF Score: {np.mean(score_list):.5f}")


def main():
    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)

    os.makedirs(CFG['output_dir'], exist_ok=True)
    LOGGER = None
    if CFG['log'] == True:
        LOGGER = init_logger(log_file=CFG['output_dir']+'/train.log')

    baseline = LGBM_baseline(CFG, LOGGER)
    baseline.train_model()
    baseline.save_features()

if __name__ == '__main__':
    main()