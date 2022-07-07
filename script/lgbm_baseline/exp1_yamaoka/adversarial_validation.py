import numpy as np
import pandas as pd
import warnings
import pickle
import os
import yaml
import json
import argparse

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from utils import *
import lightgbm as lgb

from train import *

warnings.filterwarnings('ignore')


class LGBM_baseline_AdversarialValidation():
    """
    3 methods of adversarial validation
        - public vs private
        - train  vs public
        - train  vs private
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
        
        else:
            """
                train vs public
                train vs private
            """
            for dirname, feature_name in self.using_features.items():
                if feature_name == 'all':
                    feature_name = glob.glob(self.features_path + f'/{dirname}/train/*')
                    feature_name = [os.path.splitext(os.path.basename(F))[0] 
                                    for F in feature_name if 'customer_ID' not in F]

                for name in feature_name:
                    train_filepath = self.features_path + f'/{dirname}/train' + f'/{name}.pickle'
                    test_filepath = self.features_path + f'/{dirname}/test' + f'/{name}.pickle'

                    train_one_df = pd.read_pickle(train_filepath)
                    test_one_df = pd.read_pickle(test_filepath)

                    if 'train_df' in locals():
                        train_df = pd.concat([train_df, train_one_df], axis=1)
                        test_df = pd.concat([test_df, test_one_df], axis=1)
                    else:
                        train_df = train_one_df
                        test_df = test_one_df
                    self.STDOUT(f'loading : {name} of {dirname}')
            
            train_df["target"] = 0
            test_df["target"] = 1
            df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

        self.STDOUT(f'dataframe_info:  {len(df)} rows, {len(df.columns)} features')
        return df

    def load_rawdata(self, file_name):
        #self.STDOUT(f"loading {file_name} ...")
        df = pd.read_feather(
            os.path.join(self.raw_path, file_name), 
            columns=["customer_ID","S_2"]
            )
        #self.STDOUT("done!!")
        df = df.drop_duplicates(subset=["customer_ID"], keep="last").reset_index(drop=True)
        df['S_2'] = pd.to_datetime(df['S_2'])
        df['month'] = (df['S_2'].dt.month).astype('int8')
        return df

    def create_target(self) -> np.ndarray:
        self.STDOUT(f"create target for {self.adval_method}")

        if self.adval_method == "public_vs_private":
            target = self.load_rawdata("test_data.ftr")
            assert self.train.shape[0] == target.shape[0]
            target['private'] = 0
            target.loc[target['month'] == 4,'private'] = 1
            target = target["private"].to_numpy()

        else:
            """
                train vs public
                train vs private
            """
            train = self.load_rawdata("train_data.ftr")
            test = self.load_rawdata("test_data.ftr")
            df = pd.concat([train, test], axis=0).reset_index(drop=True)
            assert self.train.shape[0] == df.shape[0]
            df = pd.concat([self.train,df], axis=1)

            idx = df[df["target"]==0].index.tolist() # extract train idx
            if self.adval_method == "train_vs_public":
                idx.extend(df[(df["target"]==1) & (df["month"]==10)].index.tolist()) # extract public idx and concat train idx
            elif self.adval_method == "train_vs_private":
                idx.extend(df[(df["target"]==1) & (df["month"]==4)].index.tolist()) # extract private idx and concat train idx
            else:
                idx = df.index.tolist()

            # create dataset concat train and (public or private)
            self.train = self.train.drop("target", axis=1).iloc[idx].reset_index(drop=True)
            target = df.iloc[idx]["target"].to_numpy()

        return target


    def train_model(self) -> None:
        num_boost_round = self.CFG['num_boost_round']
        eval_interval = self.CFG['eval_interval']
        only_first_fold = self.CFG['ONLY_FIRST_FOLD']
        skf = StratifiedKFold(n_splits=3)
        while True:
            score_list = []
            for fold, (idx_tr, idx_va) in enumerate(skf.split(self.train, self.target)):
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
            "adval_method":self.adval_method,
            "auc_threshold":self.auc_threshold,
            "importance_type":self.importance_type,
            "drift_features":self.drift_feats
        }        
        file = os.path.join(self.output_dir, f"{self.adval_method}_drift_feats.json")
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
        add_feats = ["D_59_min","D_59_max","D_59_mean"]
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