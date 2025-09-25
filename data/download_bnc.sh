#!/bin/bash

data_dir="${1:-.}"
out1_zip="${data_dir}/bnc.zip"
out2_zip="${data_dir}/2554.zip"
header="${data_dir}/header2554.xml"
[ ! -d $data_dir ] && mkdir -p $data_dir

curl https://ota.bodleian.ox.ac.uk/repository/xmlui/handle/20.500.12024/2554/allzip -o $out1_zip
unzip $out1_zip -d $data_dir && rm $out1_zip && rm $header
unzip $out2_zip -d $data_dir && rm $out2_zip
mv "${data_dir}/download" "${data_dir}/BNC"