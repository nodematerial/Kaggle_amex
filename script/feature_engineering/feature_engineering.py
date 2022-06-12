import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime as dt
from util_fe import *


class Feature():
    
    def __init__(self):
        self.dir = './data/feature/'
        self.file_dir = self.__class__.__name__ 
        if not os.path.exists(self.dir + self.file_dir):
            os.mkdir(self.dir + self.file_dir)
            os.mkdir(self.dir + self.file_dir + '/train')
            os.mkdir(self.dir + self.file_dir + '/test')
        
        
    def get_dataset(self):
        #データの読み込み
        self.train_df = pd.read_feather('./data/raw/train_data.ftr')
        self.test_df  = pd.read_feather('./data/raw/test_data.ftr')
        return self.train_df, self.test_df
        
        
    def get_features(self ,features = None):
        #作成した特徴量の取得
        if features == None:
            print('features not selected')
            exit(0)
        else:
            dfs = [pd.read_feather(f'.features/{f}.pickle') for f in features]
            dfs = reduce_mem_usage(dfs)
            
            return dfs
        
        
    def save(self ,df, col_name, split_type):
        #作成した特徴量を保存(col名で保存)
        with open(self.dir + self.file_dir + f'/{split_type}/{col_name}.pickle', mode="wb") as f:
            pickle.dump(df[col_name], f)

        
    def create_features(self):
        #作成する特徴量について記述
        pass
    
    def run_train_faetures(self):
        #作成する特徴量について記述
        pass
    
    def run_test_faetures(self):
        #作成する特徴量について記述
        pass