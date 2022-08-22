#!/bin/bash

cd feature
mkdir all_features
#mkdir all_features/train
#mkdir all_features/test

rsync */train/* all_features/train/
rsync */test/*  all_features/test/

# mkdir all_features/train/tmp
# mkdir all_features/test/tmp

# cd all_features/train

# while read line
# do
# filename="$line.pickle"
# mv $filename tmp
# done < ../../../feature_groups/all_features/importance_top1500.txt

# cd ../test

# while read line
# do
# filename="$line.pickle"
# mv $filename tmp
# done < ../../../feature_groups/all_features/importance_top1500.txt

# cd ..
# rm train/*
# mv train/tmp/* train/
# rm -r train/tmp

# rm test/*
# mv test/tmp/* test/
# rm -r test/tmp