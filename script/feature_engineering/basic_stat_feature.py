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

class Basic_Stat(Feature):

    def create_features(self, df):
        # FEATURE ENGINEERING FROM 
        # https://www.kaggle.com/code/huseyincot/amex-agg-data-how-it-created
        all_cols = [c for c in list(df.columns) if c not in ['customer_ID','S_2', 'target']]
        
        cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
        num_features = [col for col in all_cols if col not in cat_features]

        test_num_agg = df.groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max', 'last' , 'sum','median'])
        test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]

        test_cat_agg = df.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
        test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]
        df = pd.concat([test_num_agg , test_cat_agg] , axis = 1)
        df = df.reset_index()
        del test_num_agg, test_cat_agg
        
        #保存したいデータフレーム、カラムを返す
        return df , df.columns



def main():            
    sample = Basic_Stat()
    sample.get_dataset()
    sample.run()

if __name__ == '__main__':
    main()
    
