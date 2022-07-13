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