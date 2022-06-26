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
        self.test = self.create_test()


    def create_test(self) -> pd.DataFrame:
        with open('model/features.pkl', 'rb') as f:
            using_features = pickle.load(f)
            
        for dirname, feature_name in using_features.items():
            for name in feature_name:
                filepath = self.features_path + f'/{dirname}/test' + f'/{name}.pickle'
                one_df = pd.read_pickle(filepath)

                if 'df' in locals():
                    df = pd.concat([df, one_df], axis=1)
                else:
                    df = one_df
                print(f'loading : {name} of {dirname}')

        return df
        
    def infer(self) -> list:
        output = []
        models = glob.glob('model/model_fold*.pkl')
        print('[model_list]')
        for i, model_path in enumerate(models):
            print(f'{i+1}: {os.path.splitext(os.path.basename(model_path))[0]}')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            output.append(model.predict(self.test))
        output = np.array(output).mean(axis=0)
        print('inference succeed')
        return output


    def make_csv(self):
        output = self.infer()
        submission = pd.read_csv(self.samplesub_path)
        submission['prediction'] = output
        submission.to_csv('submission.csv', index=False)


def main():
    baseline = LGBM_inference()
    baseline.make_csv()

if __name__ == '__main__':
    main()