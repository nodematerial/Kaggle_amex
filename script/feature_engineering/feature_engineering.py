import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime as dt
from util_fe import *
import pickle


class Feature():
    
    def __init__(self):
        self.parent = '../../'
        self.feature_pth = 'data/feature/'
        self.raw_pth = 'data/raw/'
        self.file_dir = self.__class__.__name__ 
        if not os.path.exists(self.parent + self.feature_pth + self.file_dir):
            os.mkdir(self.parent + self.feature_pth + self.file_dir)
            os.mkdir(self.parent + self.feature_pth + self.file_dir + '/train')
            os.mkdir(self.parent + self.feature_pth + self.file_dir + '/test')
        
        
    def get_dataset(self):
        #データの読み込み
        self.train_df = pd.read_feather(self.parent + self.raw_pth + 'train_data.ftr')
        self.test_df  = pd.read_feather(self.parent + self.raw_pth + 'test_data.ftr')
        return self.train_df, self.test_df
        
        
    def save(self ,df, split_type):
        #作成した特徴量を保存(col名で保存)
        with open(self.parent + self.feature_pth + self.file_dir + 
                  f'/{split_type}/features.pickle', mode="wb") as f:
            pickle.dump(df, f)


    def create_features(self):
        #作成する特徴量について記述
        pass
    
    
    def run(self):
        df_processed , columns = self.create_features(self.train_df)
        df_processed           = reduce_mem_usage(df = df_processed)
        df_processed.index     = range(len(df_processed))
        self.save(df_processed , 'train')
            
        df_processed , columns = self.create_features(self.test_df)
        df_processed           = reduce_mem_usage(df = df_processed)
        df_processed.index     = range(len(df_processed))
        self.save(df_processed , 'test')
