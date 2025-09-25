#!/bin/bash

data_dir="${1:-.}"
out_zip="${data_dir}/tqa_train_val_test.zip"
[ ! -d $data_dir ] && mkdir -p $data_dir

curl https://s3.amazonaws.com/ai2-vision-textbook-dataset/dataset_releases/tqa/tqa_train_val_test.zip -o $out_zip
unzip $out_zip -d $data_dir && rm $out_zip