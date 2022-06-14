import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime as dt
from util_fe import *
import pickle


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
        
        
    def save(self ,df, col_name, split_type):
        #作成した特徴量を保存(col名で保存)
        with open(self.dir + self.file_dir + f'/{split_type}/{col_name}.pickle', mode="wb") as f:
            pickle.dump(df[col_name], f)

        
    def create_features(self):
        #作成する特徴量について記述
        pass
    
    
    def run(self):
        df_processed , columns = self.create_features(self.train_df)
        df_processed           = reduce_mem_usage(df = df_processed)
        df_processed.index     = range(len(df_processed))
        for col in columns:
            self.save(df_processed , col ,'train')
            
        df_processed , columns = self.create_features(self.test_df)
        df_processed           = reduce_mem_usage(df = df_processed)
        df_processed.index     = range(len(df_processed))
        for col in columns:
            self.save(df_processed , col ,'test')
