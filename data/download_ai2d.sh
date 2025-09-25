#!/bin/bash

data_dir="${1:-.}"
out_zip="${data_dir}/ai2d-all.zip"
[ ! -d $data_dir ] && mkdir -p $data_dir

curl http://ai2-website.s3.amazonaws.com/data/ai2d-all.zip -o $out_zip
unzip $out_zip -d $data_dir && rm $out_zip