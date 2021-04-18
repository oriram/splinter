export MODEL="SpanBERT/spanbert-base-cased"
export OUTPUT_DIR="output"
python run_mrqa.py \
    --model_type=bert \
    --model_name_or_path=$MODEL \
    --qass_head=False \
    --tokenizer_name=$MODEL \
    --output_dir=$OUTPUT_DIR \
    --train_file="../mrqa-few-shot/squad/squad-train-seed-42-num-examples-16.jsonl" \
    --predict_file="../mrqa-few-shot/squad/dev.jsonl" \
    --do_train \
    --do_eval \
    --cache_dir=.cache \
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