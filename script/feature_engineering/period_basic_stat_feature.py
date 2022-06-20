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

class Period_Basic_Stat(Feature):

    def create_features(self, df):
        # FEATURE ENGINEERING FROM 
        all_cols = [c for c in list(df.columns) if c not in ['customer_ID','S_2']]
        cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
        num_features = [col for col in all_cols if col not in cat_features]

        last_df = df.groupby('customer_ID')['S_2'].last().reset_index()
        last_df.columns = ['customer_ID' , 'last_date']
        df= pd.merge(df , last_df,on = 'customer_ID')
        del last_df
        
        df_list= []
        for day in [100 , 200]:
            test_num_agg = df[df['last_date'] - df['S_2'] < f'{day} days'].groupby("customer_ID")[num_features].agg(['mean', 'std', 'min', 'max' , 'sum' ,'median'])
            test_num_agg.columns = ['_'.join(x) + f'_{day}days' for x in test_num_agg.columns]
            df_list.append(test_num_agg)
        num_df = pd.concat(df_list, axis = 1)
        num_df = num_df.reset_index()
        
        #保存したいデータフレーム、カラムを返す
        return num_df , num_df.columns



def main():            
    sample = Period_Basic_Stat()
    sample.get_dataset()
    sample.run()

if __name__ == '__main__':
    main()
    
