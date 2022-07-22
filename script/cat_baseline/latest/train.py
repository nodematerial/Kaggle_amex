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

class CAT_baseline():
    def __init__(self, CFG, logger = None) -> None:
        self.STDOUT = set_STDOUT(logger)
        self.CFG = CFG
        self.features_path = CFG['features_path']
        self.feature_groups_path = CFG['feature_groups_path']
        self.using_features = CFG['using_features']
        self.output_dir = CFG['output_dir']

        self.train = self.create_train()[0]
        self.categorical = self.create_train()[1]
        self.target = self.create_target()
        self.features = self.train.columns
        assert type(CFG['training_params']) is dict
        self.best_params = CFG['training_params']
        if CFG['use_optuna']:
            optuna_params = self.tuning()
            for key, value in optuna_params.items():
                self.best_params[key] = value

        self.STDOUT(f'[ TRAINING PARAMS ]')
        if CFG['use_optuna']:
            self.STDOUT(f'params are gained from OPTUNA')
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
        categorical_used = []
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
            if dirname == 'Basic_Stat':
                feature_name.insert(0, 'customer_ID')

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

                df_dict[one_df.name] = one_df.values
                print(f'loading : {name} of {dirname}')

        df = pd.DataFrame(df_dict)
        self.STDOUT(f'dataframe_info:  {len(df)} rows, {len(df.columns)} features')
        self.STDOUT(f'used categorical_features : {categorical_used}')
        return df, categorical_used


    def create_target(self) -> pd.Series:
        label_pth = self.CFG['label_pth']
        target = pd.read_csv(label_pth).target.values
        return target


    def tuning(self) -> dict:
        num_trial = self.CFG['OPTUNA_num_trial']
        num_boost_round = self.CFG['OPTUNA_num_boost_round']
        only_first_fold = self.CFG['OPTUNA_only_first_fold']
        early_stopping = self.CFG['OPTUNA_early_stopping_rounds']

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
                train_pool = Pool(train_x.drop(columns = 'customer_ID'), train_y, cat_features=categorical)

                param = {
                    'num_boost_round' :  num_boost_round,
                    'depth' : trial.suggest_int('depth', 4, 10),              
                    'random_strength' :trial.suggest_int('random_strength', 0, 100),                       
                    'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00), 
                    'verbose' : True,
                    'allow_writing_files' : False,
                    'early_stopping_rounds' : early_stopping,  
                }
                if self.CFG['OPTUNA_GPU']:
                    param['task_type'] = 'GPU'

                if self.CFG['focal_loss']:
                    model = CatBoostClassifier(loss_function = FocalLoss(),
                                                eval_metric = 'Logloss',
                                                **param)
                else:
                    model = CatBoostClassifier(**param)

                model.fit(train_pool)
                preds = model.predict(valid_x.drop(columns = 'customer_ID'), prediction_type='Probability')[:, 1]
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
        only_first_fold = self.CFG['only_first_fold']
        score_list = []
        kf = StratifiedKFold(n_splits=5)
        oofs = []
        categorical = self.categorical

        for fold, (idx_tr, idx_va) in enumerate(kf.split(self.train, self.target)):
            train_x = self.train.iloc[idx_tr][self.features]
            valid_x = self.train.iloc[idx_va][self.features]
            train_y = self.target[idx_tr]
            valid_y = self.target[idx_va]
            train_pool = Pool(train_x.drop(columns = 'customer_ID'), train_y, cat_features=categorical)
            valid_pool = Pool(valid_x.drop(columns = 'customer_ID'), valid_y, cat_features=categorical)

            if self.CFG['focal_loss']:
                self.STDOUT('#### FOCAL LOSS ####')
                model = CatBoostClassifier(loss_function = FocalLoss(),
                                            **self.best_params)
            else:
                model = CatBoostClassifier(**self.best_params)

            model.fit(train_pool, eval_set=valid_pool)
            preds = model.predict(valid_x.drop(columns = 'customer_ID'), prediction_type='Probability')[:, 1]
            score = amex_metric(valid_y, preds)
            score_list.append(score)
            self.STDOUT(f'fold {fold} score: {score:.4f}')
            file = self.output_dir + f'/model_fold{fold}.pkl'
            pickle.dump(model, open(file, 'wb'))

            # creating oof predictions
            preds_df = pd.DataFrame(preds, columns = ['predicted'])
            preds_08 = pd.DataFrame(np.where(preds > 0.8, 1, 0), columns = ['predicted_08'])
            preds_07 = pd.DataFrame(np.where(preds > 0.7, 1, 0), columns = ['predicted_07'])
            preds_06 = pd.DataFrame(np.where(preds > 0.6, 1, 0), columns = ['predicted_06'])
            preds_05 = pd.DataFrame(np.where(preds > 0.5, 1, 0), columns = ['predicted_05'])
            oof = pd.concat([valid_x.reset_index(drop = True), preds_df,
                            preds_08, preds_07, preds_06, preds_05], axis = 1)
            oof['fold'] = fold
            oofs.append(oof)

            if self.CFG['show_importance']:
                importance = pd.DataFrame(model.get_feature_importance(train_pool,
                                        type="PredictionValuesChange"))
                importance.to_csv(self.output_dir + f'/importance_fold{fold}.csv')     

            if only_first_fold: break 

        # saving oofs
        if self.CFG['create_oofs']:
            oofs = pd.concat(oofs).reset_index()
            oofs.to_feather(self.output_dir + '/oofs.ftr')
            self.STDOUT('=' * 30)
            self.STDOUT('OOFs Info')
            f, t = oofs['predicted_08'].value_counts()
            self.STDOUT(f'threshold 0.8  True:{t} False:{f}')
            f, t = oofs['predicted_07'].value_counts()
            self.STDOUT(f'threshold 0.7  True:{t} False:{f}')
            f, t = oofs['predicted_06'].value_counts()
            self.STDOUT(f'threshold 0.6  True:{t} False:{f}')
            f, t = oofs['predicted_05'].value_counts()
            self.STDOUT(f'threshold 0.5  True:{t} False:{f}')
            self.STDOUT('=' * 30)

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