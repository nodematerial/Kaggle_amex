import numpy as np
import pandas as pd
import warnings
import pickle
import os
import sys
import yaml
import json
import argparse

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from utils import *
import lightgbm as lgb

sys.path.append("../")
from latest.train import *

warnings.filterwarnings('ignore')

methods_list = ["public_vs_private","train_vs_public","train_vs_private"]


class LGBM_baseline_AdversarialValidation(LGBM_baseline):
    """
    3 methods of adversarial validation
        - public vs private
        - train  vs public
        - train  vs private
    """
    def __init__(
        self, 
        CFG, 
        method : str, 
        auc_threshold : float,
        sampling_rate : float,
        num_dropfeats : int,
        logger = None) -> None:
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
        self.sampling_rate = sampling_rate
        self.num_dropfeats = num_dropfeats
        self.drift_feats_df = None
        self.max_drift_feats = []


    def create_train(self) -> pd.DataFrame:
        self.STDOUT(f"create train for {self.adval_method}")
        if self.adval_method == "public_vs_private":
            test_df_dict = {}
            for dirname, feature_name in self.using_features.items():
                if feature_name == 'all':
                    feature_name = glob.glob(self.features_path + f'/{dirname}/test/*')
                    feature_name = [os.path.splitext(os.path.basename(F))[0]
                                    for F in feature_name if 'customer_ID' not in F]

                elif type(feature_name) == str:
                    file = self.feature_groups_path + f'/{dirname}/{feature_name}.txt'
                    feature_name = []
                    with open(file, 'r') as f:
                            for line in f:
                                feature_name.append(line.rstrip("\n"))
                    print(feature_name)

                for name in feature_name:
                    filepath = self.features_path + f'/{dirname}/test' + f'/{name}.pickle'
                    one_df = pd.read_pickle(filepath)
                    test_df_dict[one_df.name] = one_df.values
                    self.STDOUT(f'loading : {name} of {dirname}')

            df = pd.DataFrame(test_df_dict)
        
        else:
            """
                train vs public
                train vs private
            """
            train_df_dict = {}
            test_df_dict = {}
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
                    print(feature_name)

                for name in feature_name:
                    train_filepath = self.features_path + f'/{dirname}/train' + f'/{name}.pickle'
                    test_filepath = self.features_path + f'/{dirname}/test' + f'/{name}.pickle'
                    train_one_df = pd.read_pickle(train_filepath)
                    test_one_df = pd.read_pickle(test_filepath)
                    train_df_dict[train_one_df.name] = train_one_df.values
                    test_df_dict[test_one_df.name] = test_one_df.values
                    self.STDOUT(f'loading : {name} of {dirname}')

            train_df = pd.DataFrame(train_df_dict)
            test_df = pd.DataFrame(test_df_dict)
            
            train_df["target"] = 0
            test_df["target"] = 1
            df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

        self.STDOUT(f'dataframe_info:  {len(df)} rows, {len(df.columns)} features')
        return df

    def load_rawdata(self, file_name : str):
        self.STDOUT(f"loading {file_name} ...")
        df = pd.read_feather(
            os.path.join(self.raw_path, file_name), 
            columns=["customer_ID","S_2"]
            )
        self.STDOUT("done!!")
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

    def sampling_train(self) -> None:
        self.train, _, self.target, _ = train_test_split(
            self.train, 
            self.target, 
            train_size=self.sampling_rate, 
            random_state=42, 
            stratify=self.target
            )


    def train_model(self) -> None:
        num_boost_round = self.CFG['num_boost_round']
        eval_interval = self.CFG['eval_interval']
        only_first_fold = self.CFG['ONLY_FIRST_FOLD']
        skf = StratifiedKFold(n_splits=2)
        if self.sampling_rate < 1.0:
            self.STDOUT(f"before sampling {len(self.train)} rows")
            self.sampling_train()
            self.STDOUT(f"after sampling {len(self.train)} rows")

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

            # 一定のauc値より高ければimportanceが大きいdrift featuresをdropする
            if np.mean(score_list) > self.auc_threshold:
                if self.num_dropfeats > len(self.features):
                    self.num_dropfeats = len(self.features)

                self.max_drift_feats = importance_df.apply(
                    lambda s: s.nlargest(self.num_dropfeats).index.tolist(), 
                    axis=0).to_numpy().flatten().tolist()
                self.drift_feats.extend(self.max_drift_feats)
                self.drift_feats_df = self.train[self.max_drift_feats].copy()
                self.train.drop(self.max_drift_feats, axis=1, inplace=True)
                self.features = self.train.columns
                self.STDOUT(f"drop drift feature : {self.max_drift_feats}")
                self.STDOUT(f"next using features : {self.features.tolist()}")
                self.STDOUT(f"num of drift features : {len(self.drift_feats)}")
                self.STDOUT(f"drift features : {self.drift_feats}")
            else:
                # 余分にdrop featureした場合、戻して1つずつaucを確認しながらdropする
                if len(self.max_drift_feats) > 1:
                    #self.STDOUT(self.max_drift_feats)
                    self.drift_feats = [feat for feat in self.drift_feats if feat not in self.max_drift_feats]
                    #self.STDOUT(self.drift_feats)
                    self.train = pd.concat([self.train, self.drift_feats_df], axis=1)
                    self.features = self.train.columns
                    #self.STDOUT(self.train.columns)
                    self.num_dropfeats = 1
                else:
                    break
            

        drift_dict = {
            "adval_method":self.adval_method,
            "auc_threshold":self.auc_threshold,
            "sampling_rate":self.sampling_rate,
            "importance_type":self.importance_type,
            "drift_features":self.drift_feats
        }        
        file = os.path.join(self.output_dir, f"{self.adval_method}_drift_feats.json")
        with open(file, "w", encoding='utf-8') as f:
            json.dump(drift_dict, f, indent=2)
        
        file = os.path.join(self.output_dir, f"{self.adval_method}_feats.txt")
        with open(file, 'w') as f:
            for feat in self.features:
                f.write("%s\n" % feat)


def my_error(method : str) -> None:
    if method not in methods_list:
        print(f"{method} method dose not exist")
        print("you can use --method option")
        for m in methods_list:
            print(f"    - {m}")
        sys.exit(1)
    else:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="public_vs_private")
    parser.add_argument("--threshold", type=float, default=0.7) # rocaucがthreshold未満になるまでvalidationを行う
    parser.add_argument("--sampling_rate", type=float, default=1.0) # trainのsampling rate（学習の高速化のため）
    parser.add_argument("--num_dropfeats", type=int, default=5) # importanceが上位num_dropfeats個の特徴量をdropする
    parser.add_argument("--debug", action="store_true") # 動作確認用
    args = parser.parse_args()

    my_error(args.method)

    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)

    os.makedirs(CFG['output_dir'], exist_ok=True)
    LOGGER = None
    if CFG['log'] == True:
        LOGGER = init_logger(log_file=CFG['output_dir']+'/train.log')
    
    if args.debug:
        add_feats = ["B_1_last", "B_2_last", "B_3_last", "B_1_max", "B_2_max","B_3_max","D_59_min","D_59_max","D_59_mean"]
        CFG["using_features"]["Basic_Stat"] = add_feats
        CFG['num_boost_round'] = 10
    

    baseline = LGBM_baseline_AdversarialValidation(
        CFG=CFG,
        method=args.method,
        auc_threshold=args.threshold,
        sampling_rate=args.sampling_rate,
        num_dropfeats=args.num_dropfeats,
        logger=LOGGER
        )
    baseline.train_model()
    #baseline.save_features()

if __name__ == '__main__':
    main()