#!/bin/bash

SRC=en
TGT=cmd
INPUT_DIR=../Data/nl2bash-data.json
OUTPUT_DIR=../Data/processed_data
BERT=bert-base-cased

mkdir -p $OUTPUT_DIR
TOKENIZER_SCRIPT=preprocess.py

echo "Splitting source and target files"
python ${TOKENIZER_SCRIPT} --input_file $INPUT_DIR --output_dir $OUTPUT_DIR --src_lang $SRC --tgt_lang $TGT --bert_model $BERT