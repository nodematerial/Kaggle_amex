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
from catboost import Pool, CatBoostClassifier
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


class CAT_baseline():
    def __init__(self, CFG, logger = None) -> None:
        self.STDOUT = set_STDOUT(logger)
        self.CFG = CFG
        self.features_path = CFG['features_path']
        self.using_features = CFG['using_features']
        self.output_dir = CFG['output_dir']

        self.train = self.create_train()[0]
        self.categorical = self.create_train()[1]
        self.target = self.create_target()
        self.features = self.train.columns
        self.best_params =  {}
        if CFG['use_optuna']:
            self.best_params = self.tuning()
        self.best_params['verbose'] = self.CFG['verbose']
        self.best_params['allow_writing_files'] = False
        if self.CFG['GPU']:
            self.STDOUT('###### GPU MODE ######')
            self.best_params['task_type'] = 'GPU'

    def save_features(self) -> None:
        features_dict = {}
        
        for dirname, feature_name in self.using_features.items():
            if feature_name == 'all':
                feature_name = glob.glob(self.features_path + f'/{dirname}/train/*')
                feature_name = [os.path.splitext(os.path.basename(F))[0] for F in feature_name if 'customer_ID' not in F]

            features_dict[dirname] = []
            for name in feature_name:
                features_dict[dirname].append(name)

        file = self.output_dir + f'/features.pkl'
        pickle.dump(features_dict, open(file, 'wb'))
        self.STDOUT(f'successfully saved feature names')


    def create_train(self) -> pd.DataFrame:
        categorical_used = []
        for dirname, feature_name in self.using_features.items():
            if feature_name == 'all':
                feature_name = glob.glob(self.features_path + f'/{dirname}/train/*')
                feature_name = [os.path.splitext(os.path.basename(F))[0] 
                                for F in feature_name if 'customer_ID' not in F]

            for name in feature_name:
                filepath = self.features_path + f'/{dirname}/train' + f'/{name}.pickle'
                one_df = pd.read_pickle(filepath)
                # categoricalに欠損値が入るとデータ型がおかしくなるため、最頻値で埋める
                if name in self.CFG['categorical']:
                    categorical_used.append(name)
                    one_df = one_df.fillna(one_df.mode().iloc[0])
                    try:
                        one_df = one_df.astype('int')
                    except:
                        pass
                    one_df = one_df.astype('category')

                if 'df' in locals():
                    df = pd.concat([df, one_df], axis=1)
                else:
                    df = one_df
                self.STDOUT(f'loading : {name} of {dirname}')

        self.STDOUT(f'dataframe_info:  {len(df)} rows, {len(df.columns)} features')
        self.STDOUT(f'used categorical_features : {categorical_used}')
        return df, categorical_used


    def create_target(self) -> pd.Series:
        label_pth = self.CFG['label_pth']
        target = pd.read_csv(label_pth).target.values
        return target


    def tuning(self) -> dict:
        num_trial = self.CFG['optuna_num_trial']
        only_first_fold = self.CFG['OPTUNA_ONLY_FIRST_FOLD']

        self.STDOUT('[Optuna parameter tuning]')
        kf = StratifiedKFold(n_splits=5)

        def objective(trial):
            score_list = []
            start_time = datetime.datetime.now()
            self.STDOUT(f'[ Trial {trial._trial_id} Start ]')
            categorical = self.categorical

            for fold, (idx_tr, idx_va) in enumerate(kf.split(self.train, self.target)):
                train_x = self.train.iloc[idx_tr][self.features]
                valid_x = self.train.iloc[idx_va][self.features]
                train_y = self.target[idx_tr]
                valid_y = self.target[idx_va]
                train_pool = Pool(train_x, train_y, cat_features=categorical)
                valid_pool = Pool(valid_x, valid_y, cat_features=categorical)

                param = {
                    'iterations' : trial.suggest_int('iterations', 200, 600),                         
                    'depth' : trial.suggest_int('depth', 4, 10),                                       
                    'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.1),               
                    'random_strength' :trial.suggest_int('random_strength', 0, 100),                       
                    'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00), 
                    'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
                    'od_wait' :trial.suggest_int('od_wait', 10, 50),
                    'verbose' : False,
                    'allow_writing_files' : False,
                }
                if self.CFG['GPU']:
                    param['task_type'] = 'GPU'

                if self.CFG['focal_loss']:
                    model = CatBoostClassifier(loss_function = FocalLoss(),
                                               eval_metric = 'Logloss', 
                                               **param)
                else:
                    model = CatBoostClassifier(**param)

                model.fit(train_pool)
                preds = model.predict(valid_pool)
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
        only_first_fold = self.CFG['ONLY_FIRST_FOLD']
        score_list = []
        kf = StratifiedKFold(n_splits=5)
        categorical = self.categorical

        for fold, (idx_tr, idx_va) in enumerate(kf.split(self.train, self.target)):
            train_x = self.train.iloc[idx_tr][self.features]
            valid_x = self.train.iloc[idx_va][self.features]
            train_y = self.target[idx_tr]
            valid_y = self.target[idx_va]
            train_pool = Pool(train_x, train_y, cat_features=categorical)
            valid_pool = Pool(valid_x, valid_y, cat_features=categorical)

            if self.CFG['focal_loss']:
                self.STDOUT('#### FOCAL LOSS ####')
                model = CatBoostClassifier(loss_function = FocalLoss(),
                                            eval_metric = 'Logloss', 
                                            **self.best_params)
            else:
                model = CatBoostClassifier(**self.best_params)

            model.fit(train_pool)
            preds = model.predict(valid_pool)
            score = amex_metric(valid_y, preds)
            score_list.append(score)
            self.STDOUT(f'fold {fold} score: {score:.4f}')
            file = self.output_dir + f'/model_fold{fold}.pkl'
            pickle.dump(model, open(file, 'wb'))
            if only_first_fold: break 

        self.STDOUT(f"OOF Score: {np.mean(score_list):.4f}")


def main():
    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)

    os.makedirs(CFG['output_dir'], exist_ok=True)
    LOGGER = None
    if CFG['log'] == True:
        LOGGER = init_logger(log_file=CFG['output_dir']+'/train.log')

    baseline = CAT_baseline(CFG, LOGGER)
    baseline.train_model()
    baseline.save_features()

if __name__ == '__main__':
    main()