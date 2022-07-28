import numpy as np
import pandas as pd
import warnings
import pickle
import os
import glob


warnings.filterwarnings('ignore')


class LGBM_inference():
    def __init__(self) -> None:
        self.features_path = '../../../data/feature'
        self.samplesub_path = '../../../data/raw/sample_submission.csv'
        self.test1 = self.create_test()[:300000]
        self.test2 = self.create_test()[300000:600000]
        self.test3 = self.create_test()[600000:]


    def create_test(self) -> pd.DataFrame:
        df_dict = {}
        with open('model/features.pkl', 'rb') as f:
            using_features = pickle.load(f)
            
        for dirname, feature_name in using_features.items():
            for name in feature_name:
                filepath = self.features_path + f'/{dirname}/test' + f'/{name}.pickle'
                one_df = pd.read_pickle(filepath)
                df_dict[one_df.name] = one_df.values
                print(f'loading : {name} of {dirname}')
        df = pd.DataFrame(df_dict)
        return df
        
    def infer(self, df) -> list:
        output = []
        models = glob.glob('model/model_fold*.pkl')
        print('[model_list]')
        for i, model_path in enumerate(models):
            print(f'{i+1}: {os.path.splitext(os.path.basename(model_path))[0]}')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            output.append(model.predict_proba(df)[:, 1])
        output = np.array(output).mean(axis=0)
        print('inference succeed')
        return output


    def make_csv(self):
        output1 = self.infer(self.test1)
        output2 = self.infer(self.test2)
        output3 = self.infer(self.test3)
        output = np.concatenate([output1, output2, output3])
        dirname = os.path.basename(os.path.dirname(os.getcwd()))
        submission = pd.read_csv(self.samplesub_path)
        submission['prediction'] = output
        submission.to_csv(f'{dirname}.csv', index=False)


def main():
    baseline = LGBM_inference()
    baseline.make_csv()

if __name__ == '__main__':
    main()