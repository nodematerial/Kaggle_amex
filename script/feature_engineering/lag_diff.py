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

class Lag_diff(Feature):

    def create_features(self, data):
        # FEATURE ENGINEERING FROM 
        # https://www.kaggle.com/code/ragnar123/amex-lgbm-dart-cv-0-7977
        all_cols = [c for c in list(data.columns) if c not in ['customer_ID','S_2', 'target']]
        
        cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
        num_features = [col for col in all_cols if col not in cat_features]

        df1 = []
        df2 = []
        customer_ids = []
        for customer_id, df in data.groupby(['customer_ID']):
            # Get the differences
            diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
            diff_df2 = df[num_features].diff(2).iloc[[-1]].values.astype(np.float32)
            # Append to lists
            df1.append(diff_df1)
            df2.append(diff_df2)
            customer_ids.append(customer_id)
        # Concatenate
        df1 = np.concatenate(df1, axis = 0)
        df2 = np.concatenate(df2, axis = 0)
        # Transform to dataframe
        df1 = pd.DataFrame(df1, columns = [col + '_diff1' for col in df[num_features].columns])
        df2 = pd.DataFrame(df2, columns = [col + '_diff2' for col in df[num_features].columns])
        # Add customer id
        df1['customer_ID'] = customer_ids
        df = pd.concat([df1, df2] , axis = 1)
        return df, df.columns
def main():            
    sample = Lag_diff()
    sample.get_dataset()
    sample.run()

if __name__ == '__main__':
    main()
    
