import pandas as pd 
import yaml

def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def create_train(feature_path, using_features, logger):
    for dirname, feature_name in using_features.items():
        for name in feature_name:
            filepath = feature_path + f'/{dirname}/train' + f'/{name}.pickle'
            one_df = pd.read_pickle(filepath)

            if 'df' in locals():
                df = pd.concat([df, one_df], axis=1)
            else:
                df = one_df
            print(f'loading : {name} of {dirname}')
    logger.info(f'dataframe_info:  {len(df)} rows, {len(df.columns)} features')
    return df


if __name__ == '__main__':
    with open('config.yml', 'r') as yml:
        CFG = yaml.load(yml, Loader=yaml.SafeLoader)
    create_train(CFG['features_path'], CFG['using_features'])