from feature_engineering import *



#クラス名のdirを作成,train/testごとにカラム名で特徴量を保存
#  class
#  └train
#   └col1.pickle
#   └col2.pickle
#  └test
#   └col1.pickle
#   └col2.pickle
#
#sample名
#

class Div_Min_Max(Feature):

    def create_features(self, df):
        # FEATURE ENGINEERING FROM 
        all_cols = [c for c in list(df.columns) if c not in ['customer_ID','S_2', 'target']]

        cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
        num_features = [col for col in all_cols if col not in cat_features]
        test_num_agg = df.groupby("customer_ID")[num_features].agg(['last', 'max' , 'min'])
        test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]

        for col in num_features:
            test_num_agg[f'{col}_div_min_last'] = test_num_agg[f'{col}_min'] / test_num_agg[f'{col}_last']
            test_num_agg[f'{col}_div_max_last'] = test_num_agg[f'{col}_max'] / test_num_agg[f'{col}_last']
            test_num_agg[f'{col}_div_min_max'] = test_num_agg[f'{col}_min'] / test_num_agg[f'{col}_max']
            test_num_agg = test_num_agg.drop(columns = [f'{col}_last' , f'{col}_min' , f'{col}_max'] )
        #保存したいデータフレーム、カラムを返す
        test_num_agg = test_num_agg.reset_index()
        
        
        
        
        
        return test_num_agg , test_num_agg.columns



def main():            
    sample = Div_Min_Max()
    sample.get_dataset()
    sample.run()

if __name__ == '__main__':
    main()
    
