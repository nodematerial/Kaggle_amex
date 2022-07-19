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

class Shift(Feature):

    def create_features(self, df):
        # FEATURE ENGINEERING FROM 
        all_cols = [c for c in list(df.columns) if c not in ['customer_ID','S_2', 'target']]

        cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
        num_features = [col for col in all_cols if col not in cat_features]
        for col in cat_features:
            num_features.append(col)
        
        shift1_features = [col + '_shift1' for col in num_features]
        df[shift1_features] = df.groupby("customer_ID")[num_features].shift(1)
        shift2_features = [col + '_shift2' for col in num_features]
        df[shift2_features] = df.groupby("customer_ID")[num_features].shift(2)
        test_num_agg = df.groupby("customer_ID")[shift1_features + shift2_features].agg(['last'])
        test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]


        #保存したいデータフレーム、カラムを返す
        return test_num_agg , test_num_agg.columns



def main():            
    sample = Shift()
    sample.get_dataset()
    sample.run()

if __name__ == '__main__':
    main()
    