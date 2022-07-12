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

class Date(Feature):

    def create_features(self, df):
        # FEATURE ENGINEERING FROM 
        df['S_2'] = pd.to_datetime(df['S_2'])
        last_df = df.groupby('customer_ID')['S_2'].last().reset_index()
        first_df = df.groupby('customer_ID')['S_2'].first().reset_index()
        first_last_df= pd.merge(first_df , last_df,on = 'customer_ID')
        first_last_df['diff_first_last_day'] = (first_last_df['S_2_y'] - first_last_df['S_2_x']).dt.days

        df['diff_days'] = df.groupby('customer_ID')[['S_2']].diff()
        df['diff_days'] = df['diff_days'].dt.days
        fe_df = df.groupby('customer_ID')['diff_days'].agg(['mean', 'std', 'min', 'max' , 'sum' ,'median'])
        fe_df.columns = ['diff_days_' + x for x in fe_df.columns]

        fe_df = pd.merge(fe_df,first_last_df[['customer_ID' , 'diff_first_last_day']] , on = 'customer_ID' , how = 'inner')
        #保存したいデータフレーム、カラムを返す
        return fe_df , fe_df.columns



def main():            
    sample = Date()
    sample.get_dataset()
    sample.run()

if __name__ == '__main__':
    main()
    
