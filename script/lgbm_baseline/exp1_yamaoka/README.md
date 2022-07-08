## adversarial validationの使い方(7/8修正)

現状、以下の3つのvalidation methodが使えます
- public vs private
- train vs public
- train vs private


ROC AUCが一定値（デフォルトは0.7）以上の場合にimportanceが最大のfeatureをdrift featureとみなしtrainからdropします。
これをROC AUCが一定値未満になるまで行い、最終的にmodelディレクトリ配下にdrift feature columnsを記述したjsonが保存されます。
<br>
### 7/8追加
- sampling_rateとnum_dropfeatsオプションを追加しました。sampling_rateは学習時間削減のためtrainの行数をどの割合使用するかを指定でき、num_dropfeatsは一回のadversarial validationでimportance上位何個の特徴量をdropするかを指定できます。（初めは1つずつdropする仕様でしたが、膨大な時間がかかるためです。）
- 実験結果はmodel/*.jsonに保存されますが、adversarial validationの結果、driftしていないと見做せる特徴量をmodel/*.txtに保存されるようにしました。（そのままusing_featuresで引数指定して使えると思います）
<br>

動作確認
```
python adversarial_validation.py --debug --method public_vs_private
```

使用例
```
python adversarial_validation.py --method train_vs_public --sampling_rate 0.2 --num_dropfeats 5 --threshold 0.6
```
