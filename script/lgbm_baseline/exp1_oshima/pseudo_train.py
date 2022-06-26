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

from train import *

warnings.filterwarnings('ignore')



class LGBM_baseline_pseudo(LGBM_baseline):
    #self.testも呼び出すように修正
    def __init__(self, CFG, logger) -> None:
        self.STDOUT      = set_STDOUT(logger)
        self.CFG         = CFG
        self.train       = self.create_dm(type = 'train')
        self.test        = self.create_dm(type = 'test')
        self.target      = self.create_target()
        self.features    = self.train.columns
        self.best_params =  {'objective': 'binary', 'metric': 'custom'}
        if CFG['use_optuna']:
            self.best_params = self.tuning()
        self.best_params['force_col_wise'] = True


    def create_dm(self , type : str) -> pd.DataFrame:
        features_path = self.CFG['features_path']
        using_features = self.CFG['using_features']

        for dirname, feature_name in using_features.items():
            if feature_name == 'all':
                feature_name = glob.glob(features_path + f'/{dirname}/{type}/*')
                feature_name = [os.path.splitext(os.path.basename(F))[0] for F in feature_name if 'customer_ID' not in F]

            for name in feature_name:
                filepath = features_path + f'/{dirname}/{type}' + f'/{name}.pickle'
                one_df = pd.read_pickle(filepath)

                if 'df' in locals():
                    df = pd.concat([df, one_df], axis=1)
                else:
                    df = one_df
                self.STDOUT(f'loading : {name} of {dirname}')

        self.STDOUT(f'dataframe_info:  {len(df)} rows, {len(df.columns)} features')
        return df



    def train_model(self) -> None:
        output_dir       = self.CFG['output_dir']
        num_boost_round  = self.CFG['num_boost_round']
        eval_interval    = self.CFG['eval_interval']
        only_first_fold  = self.CFG['ONLY_FIRST_FOLD']
        score_list       = []
        test_preds_list  = []
        kf = StratifiedKFold(n_splits=2)
        for fold, (idx_tr, idx_va) in enumerate(kf.split(self.train, self.target)):
            train_x = self.train.iloc[idx_tr][self.features]
            valid_x = self.train.iloc[idx_va][self.features]
            train_y = self.target[idx_tr]
            valid_y = self.target[idx_va]
            dtrain  = lgb.Dataset(train_x, label=train_y)
            dvalid  = lgb.Dataset(valid_x, label=valid_y)

            gbm = lgb.train(self.best_params, dtrain, valid_sets=[dvalid], 
                            num_boost_round=num_boost_round,
                            callbacks=[lgb.log_evaluation(eval_interval)],
                            feval = [custom_accuracy])

            preds = gbm.predict(valid_x)
            score = amex_metric(valid_y, preds)
            score_list.append(score)
            
            test_preds = gbm.predict(self.test)
            test_preds_list.append(test_preds)
            file = output_dir + f'/model_fold{fold}.pkl'
            pickle.dump(gbm, open(file, 'wb'))
            if only_first_fold: break
        self.test_preds = np.mean(test_preds_list , axis = 0)
        
        
        self.STDOUT(f"OOF Score: {np.mean(score_list):.5f}")

    def create_pseudo(self, upper:float)->None:
        
        rate            = len(self.target[self.target == 0]) / len(self.target[self.target == 1])
        pseudo1_dm      = self.test.iloc[self.test_preds > upper , :].copy()
        pseudo1_target  = np.ones((len(pseudo1_dm)))
        lower           = np.sort(self.test_preds)[int(rate * len(pseudo1_dm))]
        pseudo0_dm      = self.test.iloc[self.test_preds < lower , :].copy()
        pseudo0_target  = np.zeros((len(pseudo0_dm)))
        
        self.STDOUT(f'add pseudo-label1 : {len(pseudo1_dm)}')
        self.STDOUT(f'add pseudo-label0 : {len(pseudo0_dm)}')
        
        self.train  = pd.concat([self.train , pseudo1_dm])
        self.target = np.concatenate([self.target, pseudo1_target])
        self.train  = pd.concat([self.train , pseudo0_dm])
        self.target = np.concatenate([self.target, pseudo0_target])

def main():
    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)

    os.makedirs(CFG['output_dir'], exist_ok=True)
    LOGGER = None
    if CFG['log'] == True:
        LOGGER = init_logger(log_file=CFG['output_dir']+'/train.log')

    baseline = LGBM_baseline_pseudo(CFG, LOGGER)
    baseline.train_model()
    baseline.create_pseudo( upper = CFG['upper'])
    baseline.train_model()

if __name__ == '__main__':
    main()