# Splinter

This repository contains the code, models and datasets discussed in our paper "[Few-Shot Question Answering by Pretraining Span Selection](https://arxiv.org/abs/2101.00438)", to appear at ACL 2021.

Our pretraining code is based on TensorFlow (checked on 1.15), while fine-tuning is based on PyTorch (1.7.1) and 
Transformers (2.9.0). Note each has its own requirement file: [pretraining/requirements.txt](pretraining/requirements.txt) 
and [finetuning/requirements.txt](finetuning/requirements.txt).  

## Data

### Downloading Few-Shot MRQA Splits

```bash
curl -L https://www.dropbox.com/sh/pfg8j6yfpjltwdx/AAC8Oky0w8ZS-S3S5zSSAuQma?dl=1 > mrqa-few-shot.zip
unzip mrqa-few-shot.zip -d mrqa-few-shot
```

## Pretrained Model

##### Command for downloading **Splinter**  

```bash
curl -L https://www.dropbox.com/sh/h63xx2l2fjq8bsz/AAC5_Z_F2zBkJgX87i3IlvGca?dl=1 > splinter.zip
unzip splinter.zip -d splinter 
```

## Pretraining

Create a virtual environment and execute 
```bash
cd pretraining
pip install -r requirements.txt  # or requirements-gpu.txt for a GPU version
```

Then download the raw data (our pretraining was based on Wikipedia and BookCorpus).
We support two data formats:
* For wiki, a ```<doc>``` tag starts a new article and a ```</doc>``` ends it.
* For BookCorpus, we process an already-tokenized file where tokens are separated by whitespaces. 
Newlines stands for a new book. 

##### Command for creating the pretraining data
This command takes as input a set of files (```$INPUT_PATTERN```) and creates a tensorized dataset for pretraining. 
It supports the following masking schemes:
* Masked Language Modeling ([Devlin et. al 2019](https://www.aclweb.org/anthology/N19-1423.pdf))
* Masked Language Modeling with Geometric Masking (SpanBERT; [Joshi et. al 2020](https://www.aclweb.org/anthology/2020.tacl-1.5.pdf)).
See an example for [creating the data](pretraining/scripts/create_data_for_spanbert.sh) for SpanBERT, 
and for [pretraining](pretraining/scripts/run_spanbert.sh) it.
* Recurring Span Selection (our pretraining scheme)

##### Command for creating the data for Splinter (recurring span selection)

```bash
cd pretraining
python create_pretraining_data.py \
    --input_file=$INPUT_PATTERN \
    --output_dir=$OUTPUT_DIR \
    --vocab_file=vocabs/bert-cased-vocab.txt \
    --do_lower_case=False \
    --do_whole_word_mask=False \
    --max_seq_length=512 \
    --num_processes=63 \
    --dupe_factor=5 \
    --max_span_length=10 \
    --recurring_span_selection=True \
    --only_recurring_span_selection=True \
    --max_questions_per_seq=30
``` 

n-gram statistics are written to ```ngrams.txt``` in the output directory.

##### Command for pretraining Splinter

```bash
cd pretraining
python run_pretraining.py \
    --bert_config_file=configs/bert-base-cased-config.json \
    --input_file=$INPUT_FILE \
    --output_dir=$OUTPUT_DIR \
    --max_seq_length=512 \
    --recurring_span_selection=True \
    --only_recurring_span_selection=True \
    --max_questions_per_seq=30 \
    --do_train \
    --train_batch_size=256 \
    --learning_rate=1e-4 \
    --num_train_steps=2400000 \
    --num_warmup_steps=10000 \
    --save_checkpoints_steps=10000 \
    --keep_checkpoint_max=240 \
    --use_tpu \
    --num_tpu_cores=8 \
    --tpu_name=$TPU_NAME

```
This can be trained using GPUs by dropping the ```use_tpu``` flag (although it was tested mainly on TPUs).

#### Convert TensorFlow Model to PyTorch

In order to fine-tune the TF model you pretrained with ```run_pretraining.py```, you will first need to convert it to 
PyTorch. You can do so by
```bash
cd model_conversion
pip install -r requirements.txt
python convert_tf_to_pytorch.py --tf_checkpoint_path $TF_MODEL_PATH --pytorch_dump_path $OUTPUT_PATH
```


## Fine-tuning

Fine-tuning has different requirements than pretraining, as it uses HuggingFace's Transformers library. 
Create a virtual environment and execute 
```bash
cd finetuning
pip install -r requirements.txt
```

**Please Note:**  If you want to reproduce results from the paper or run with a QASS head in genral, questions need to 
be augmented with a ```[QUESTION]``` token. In order to do so, please run

```bash
cd finetuning
python qass_preprocess.py --path "../mrqa-few-shot/*/*.jsonl"
```
This will add a ```[MASK]``` token to each question in the training data, which will later be replaced by a 
```[QUESTION]``` token automatically by the QASS layer implementation.

Then fine-tune Splinter by
```bash
cd finetuning
export MODEL="../splinter"
export OUTPUT_DIR="output"
python run_mrqa.py \
    --model_type=bert \
    --model_name_or_path=$MODEL \
    --qass_head=True \
    --tokenizer_name=$MODEL \
    --output_dir=$OUTPUT_DIR \
    --train_file="../mrqa-few-shot/squad/squad-train-seed-42-num-examples-16_qass.jsonl" \
    --predict_file="../mrqa-few-shot/squad/dev_qass.jsonl" \
    --do_train \
    --do_eval \
    --max_seq_length=384 \
    --doc_stride=128 \
    --threads=4 \
    --save_steps=50000 \
    --per_gpu_train_batch_size=12 \
    --per_gpu_eval_batch_size=16 \
    --learning_rate=3e-5 \
    --max_answer_length=10 \
    --warmup_ratio=0.1 \
    --min_steps=200 \
    --num_train_epochs=10 \
    --seed=42 \
    --use_cache=False \
    --evaluate_every_epoch=False 
```

In order to train with automatic mixed precision, install [apex](https://github.com/NVIDIA/apex/) and add the ```--fp16``` flag.

See an example script for fine-tuning SpanBERT (rather than Splinter) [here](finetuning/scripts/finetune_spanbert.sh).

## Citation

If you find this work helpful, please cite us
```
@inproceedings{ram-etal-2021-shot,
    title = "Few-Shot Question Answering by Pretraining Span Selection",
    author = "Ram, Ori  and
      Kirstain, Yuval  and
      Berant, Jonathan  and
      Globerson, Amir  and
      Levy, Omer",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.239",
    pages = "3066--3079",
}
```

### Acknowledgements

We would like to thank the European Research Council (ERC) for funding the project, and to Google’s TPU 
Research Cloud (TRC) for their support in providing TPUs.