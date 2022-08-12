from feature_engineering import *
import numpy as np


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

class Important_mix42(Feature):
    def create_features(self, df):
        np.random.seed(seed=42)
        important_features = ["P_2", "B_1", "B_2", "B_3", "B_4", "B_9", "B_11", "D_39", "D_41", "D_42", "D_43", "D_44", "D_48", "S_3"]        
        test_num_agg = df.groupby("customer_ID")[important_features].agg(['last'])
        df = pd.DataFrame()
        for i in range(len(important_features)):
            for j in range(i + 1, len(important_features)):
                left_name = important_features[i]
                right_name = important_features[j]
                left_feature = test_num_agg[left_name]['last']
                right_feature = test_num_agg[right_name]['last']
                ratio = np.random.rand()
                df[f'mix_{left_name}_{right_name}_42'] = left_feature * ratio + right_feature * (1 - ratio)

        df = df.reset_index()
        del test_num_agg
        
        #保存したいデータフレーム、カラムを返す
        return df , df.columns


def main():            
    sample = Important_mix42()
    sample.get_dataset()
    sample.run()

if __name__ == '__main__':
    main()