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

class Important_cross(Feature):
    def create_features(self, df):
        important_features = ["P_2", "B_1", "B_2", "B_3", "B_4", "B_9", "B_11", "D_39", "D_41", "D_42", "D_43", "D_44", "D_48", "S_3"]        

        test_num_agg = df.groupby("customer_ID")[important_features].agg(['last'])
        df = pd.DataFrame()
        for i in range(len(important_features)):
            for j in range(i + 1, len(important_features)):
                left_name = important_features[i]
                right_name = important_features[j]
                left_feature = test_num_agg[left_name]['last']
                right_feature = test_num_agg[right_name]['last']
                df[f'cross_{left_name}_{right_name}'] = left_feature * right_feature

        df = df.reset_index()
        del test_num_agg
        
        #保存したいデータフレーム、カラムを返す
        return df , df.columns


def main():            
    sample = Important_cross()
    sample.get_dataset()
    sample.run()

if __name__ == '__main__':
    main()