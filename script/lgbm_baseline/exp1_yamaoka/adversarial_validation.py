import numpy as np
import pandas as pd
import datetime
import warnings
import pickle
import gc
import os
import yaml
from tqdm.auto import tqdm
import json
import argparse

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils import *
import lightgbm as lgb
import optuna

from train import *

warnings.filterwarnings('ignore')


class LGBM_baseline_AdversarialValidation():
    """
    two methods of adversarial validation
        - public vs private
        - train  vs test
    """
    def __init__(self, CFG, method="public_vs_private", auc_threshold=0.7, logger = None) -> None:
        self.STDOUT = set_STDOUT(logger)
        self.CFG = CFG
        self.features_path = CFG['features_path']
        self.using_features = CFG['using_features']
        self.raw_path = "../../../data/raw"
        self.output_dir = CFG['output_dir']
        self.adval_method = method
        self.drift_feats = []
        self.auc_threshold = auc_threshold
        self.importance_type = "split"
        self.train = self.create_train()
        self.target = self.create_target()
        self.features = self.train.columns
        self.best_params =  {'objective': 'binary', 'metric': 'auc'}
        self.best_params['force_col_wise'] = True
        

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
        self.STDOUT(f"create train for {self.adval_method}")
        if self.adval_method == "public_vs_private":
            for dirname, feature_name in self.using_features.items():
                if feature_name == 'all':
                    feature_name = glob.glob(self.features_path + f'/{dirname}/test/*')
                    feature_name = [os.path.splitext(os.path.basename(F))[0]
                                    for F in feature_name if 'customer_ID' not in F]

                for name in feature_name:
                    filepath = self.features_path + f'/{dirname}/test' + f'/{name}.pickle'
                    one_df = pd.read_pickle(filepath)

                    if 'df' in locals():
                        df = pd.concat([df, one_df], axis=1)
                    else:
                        df = one_df
                    self.STDOUT(f'loading : {name} of {dirname}')
        
        elif self.adval_method == "train_vs_test":
            pass
        elif self.adval_method == "train_vs_public":
            pass
        elif self.adval_method == "train_vs_private":
            pass
        else:
            pass

        self.STDOUT(f'dataframe_info:  {len(df)} rows, {len(df.columns)} features')
        return df


    def create_target(self) -> np.ndarray:
        self.STDOUT(f"create target for {self.adval_method}")
        if self.adval_method == "public_vs_private":
            self.STDOUT("loading test_data.ftr ...")
            target = pd.read_feather(
                os.path.join(self.raw_path, "test_data.ftr"), 
                columns=["customer_ID","S_2"]
                )
            self.STDOUT("done!!")

            target = target.drop_duplicates(subset=["customer_ID"], keep="last").reset_index(drop=True)
            assert self.train.shape[0] == target.shape[0]
            target = pd.concat([self.train, target], axis=1)
            target['S_2'] = pd.to_datetime(target['S_2'])
            target['month'] = (target['S_2'].dt.month).astype('int8')
            target = target.reset_index(drop=True)
            self.train = target.drop(["S_2","month","customer_ID"],axis=1)
            target['private'] = 0
            target.loc[target['month'] == 4,'private'] = 1
            target = target["private"].to_numpy()

        elif self.adval_method == "train_vs_test":
            pass
        elif self.adval_method == "train_vs_public":
            pass
        elif self.adval_method == "train_vs_private":
            pass
        else:
            pass

        return target


    def train_model(self) -> None:
        num_boost_round = self.CFG['num_boost_round']
        eval_interval = self.CFG['eval_interval']
        only_first_fold = self.CFG['ONLY_FIRST_FOLD']
        kf = StratifiedKFold(n_splits=3)
        while True:
            score_list = []
            for fold, (idx_tr, idx_va) in enumerate(kf.split(self.train, self.target)):
                train_x = self.train.iloc[idx_tr][self.features]
                valid_x = self.train.iloc[idx_va][self.features]
                train_y = self.target[idx_tr]
                valid_y = self.target[idx_va]
                dtrain = lgb.Dataset(train_x, label=train_y)
                dvalid = lgb.Dataset(valid_x, label=valid_y)

                gbm = lgb.train(self.best_params, dtrain, valid_sets=[dvalid], 
                                num_boost_round=num_boost_round,
                                callbacks=[lgb.log_evaluation(eval_interval)],
                                )

                preds = gbm.predict(valid_x)
                score = roc_auc_score(valid_y, preds)
                score_list.append(score)

                importance_df = pd.DataFrame(
                    gbm.feature_importance(importance_type=self.importance_type), 
                    index=self.features, 
                    columns=['importance']
                    )
                if only_first_fold: break
            
            self.STDOUT(f"OOF Score: {np.mean(score_list):.5f}")

            if sum(score_list)/len(score_list) > self.auc_threshold:
                max_drift_feat = importance_df['importance'].idxmax()
                self.drift_feats.append(max_drift_feat)
                self.train.drop(max_drift_feat, axis=1, inplace=True)
                self.features = self.train.columns
                self.STDOUT(f"drop drift feature : {max_drift_feat}")
                self.STDOUT(f"next using features : {self.features.tolist()}")
            else:
                break
            

        drift_dict = {
            "auc_threshold":self.auc_threshold,
            "importance_type":self.importance_type,
            "drift_features":self.drift_feats
        }        
        file = os.path.join(self.output_dir, "drift_feats.json")
        with open(file, "w", encoding='utf-8') as f:
            json.dump(drift_dict, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="public_vs_private")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)

    os.makedirs(CFG['output_dir'], exist_ok=True)
    LOGGER = None
    if CFG['log'] == True:
        LOGGER = init_logger(log_file=CFG['output_dir']+'/train.log')
    
    if args.debug:
        add_feats = ["B_29_min","B_29_max","B_29_mean","B_29_median","B_29_std"]
        CFG["using_features"]["Basic_Stat"].extend(add_feats)
        CFG['num_boost_round'] = 100

    baseline = LGBM_baseline_AdversarialValidation(
        CFG=CFG,
        method=args.method,
        auc_threshold=args.threshold,
        logger=LOGGER
        )
    baseline.train_model()
    #baseline.save_features()

if __name__ == '__main__':
    main()