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

class Div(Feature):

    def create_features(self, df):
        # FEATURE ENGINEERING FROM 
        all_cols = [c for c in list(df.columns) if c not in ['customer_ID','S_2', 'target']]

        cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
        num_features = [col for col in all_cols if col not in cat_features]
        shift_features = [col + '_shift1' for col in num_features]
        df[shift_features] = df.groupby("customer_ID")[num_features].shift(1)
        test_num_agg = df.groupby("customer_ID")[num_features + shift_features].agg(['first', 'last' , 'mean'])
        shift1 = df.groupby("customer_ID")[num_features].shift(1)
        test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]

        for col in num_features:
            test_num_agg[f'{col}_div_first_last'] = test_num_agg[f'{col}_first'] / test_num_agg[f'{col}_last']
            test_num_agg[f'{col}_div_mean_last'] = test_num_agg[f'{col}_mean'] / test_num_agg[f'{col}_last']
            test_num_agg[f'{col}_div_last1_last'] = test_num_agg[f'{col}_shift1_last'] / test_num_agg[f'{col}_last']
            test_num_agg = test_num_agg.drop(columns = [f'{col}_last' , f'{col}_first' , f'{col}_mean'  , f'{col}_shift1_first' , f'{col}_shift1_last' , f'{col}_shift1_mean'] )
        #保存したいデータフレーム、カラムを返す
        
        
        
        
        
        
        return test_num_agg , test_num_agg.columns



def main():            
    sample = Div()
    sample.get_dataset()
    sample.run()

if __name__ == '__main__':
    main()
    
