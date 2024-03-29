## LightGBM Baseline のlatest


### 7/7
using_featuresの引数に特徴量グループを指定できるようにした。

data/feature_groupsフォルダ配下に使用する特徴量を記述したテキストファイル
を配置し、ファイル名をusing_featuresの引数に設定することで、記述した特徴量を
使用することができる。

* 例: Feature_importance でtop100の特徴量を使用する。

```
using_features : {
  Basic_Stat : importance_top100,
}
```

特徴量読み込みの形式を変更し、高速に動作するようになった。(by Occccnさん)

### 7/8
ligthGBM のハイパーパラメータをコンフィグの custom_params から設定できるようにした。

* 例: 公開 Notebook で使用されているパラメータを用いる

```
# Custom param settings ※ use_optuna を On にしていた場合、こちらが優先して使われる。
use_custom_params : True
custom_params : {
  learning_rate : 0.03, 
  reg_lambda : 50,
  min_child_samples : 2400,
  num_leaves : 95,
  colsample_bytree : 0.19,
  max_bins : 511,
} 
```

### 7/11
lightGBM のboosting_typeオプションを指定できるようにした。Config.ymlに
```
boosting_type : gbdt # gbdt(default), rf, dart, goss
```
などと書いて使用する。

### 7/13
config から feature importance の設定が行えるように変更した
```
show_importance : True
```
とすれば出力フォルダにfeature importance の一覧がcsvにて出力される。

### 7/14
Out of fold の予測値を feature 形式で出力できるようにした。
生値に加えて、threshold = 0.9, 0.8, 0.7, 0.6, 0.5 を用いた二値分類を行えるようにしている。
Config.yml より、以下のように指定する。
```
create_oofs : True
```

コードのわかり易さを向上させるため、lightGBMのパラメータの指定方法を変更した。
今までは、config.yml内のパラメータ:use_custom_paramsの指定は任意であったが、
名前をtraining_paramsと改め、指定を必須にしている。
また、use_optunaをtrueにした場合には、Optunaによってチューニングされたパラメータが
training_paramsで指定されていなければ、新たに追加され、既に指定されていた場合は
チューニングパラメータによって上書きされる。

```
#before_optuna        →      #after_optuna
objective : binary           objective : binary　
metric : custom              metric : custom　
n_estimators : 1000          n_estimators : 1000
early_stopping_round : 50    early_stopping_round : 50　　
seed : 42                    seed : 42　　
boosting : gbdt              boosting : gbdt　
learning_rate : 0.05         learning_rate : 0.05
min_child_samples : 2400     min_child_samples : 272
num_leaves : 100             num_leaves : 134　
max_bins : 511               max_bins : 511
force_col_wise : True        force_col_wise : True
                             lambda_l1 : 7.460526006830761e-05
                             lambda_l2 : 0.012092870767817824
                             feature_fraction : 0.11926294640739502
'''