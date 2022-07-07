## adversarial validationの使い方

現状、以下の3つのvalidation methodが使えます
- public vs private
- train vs public
- train vs private


ROC AUCが一定値（デフォルトは0.7）以上の場合にimportanceが最大のfeatureをdrift featureとみなしtrainからdropします。
これをROC AUCが一定値未満になるまで行い、最終的にmodelディレクトリ配下にdrift feature columnsを記述したjsonが保存されます。  
<br>

動作確認
```
python adversarial_validation.py --debug --method public_vs_private
```

使用例
```
python adversarial_validation.py --method train_vs_public --threshold 0.6
```
