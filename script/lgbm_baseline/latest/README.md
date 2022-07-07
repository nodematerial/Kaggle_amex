## LightGBM Baseline のlatest


### 7/7
using_featuresの引数に特徴量グループを指定できるようになった。

data/feature_groupsフォルダ配下に使用する特徴量を記述したテキストファイル
を配置し、ファイル名をusing_featuresの引数に設定することで、記述した特徴量を
使用することができる。

* 例: Feature_importance でtop100の特徴量を使用する。

```
using_features : {
  Basic_Stat : importance_top100,
}
```