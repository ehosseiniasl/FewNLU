task_name=$1
device=$2
#model_type=$3
model=$3
few_shot_setting=$4
#num_shots=$5

YOUR_DATA_DIR="/export/home/data/FewGLUE_v2/"
YOUR_SAVE_DIR="./output"

#few_shot_setting="dev32_split"
dataset_name="superglue"
method="sequence_classifier"
arch_method="default"
data_dir=$YOUR_DATA_DIR
#save_dir=$YOUR_SAVE_DIR/${few_shot_setting}/${model_type}_${task_name}_${arch_method}_${method}_model
save_dir=$YOUR_SAVE_DIR/${few_shot_setting}/${model}_${task_name}_${arch_method}_${method}_model


#if [ $model_type = "albert" ]; then
#  model_name_or_path="albert-xxlarge-v2"
#
#  TRAIN_BATCH_SIZE_CANDIDATES="8"
#  LR_CANDIDATES="1e-5 2e-5"

if [ $model = "albert_xxlarge" ]; then
  model_type='albert'
  model_name_or_path="albert-xxlarge-v2"

  TRAIN_BATCH_SIZE_CANDIDATES="8"
  LR_CANDIDATES="1e-5 2e-5"

elif [ $model = "albert_base" ]; then
  model_type="albert"
  model_name_or_path="albert-base-v2"

  TRAIN_BATCH_SIZE_CANDIDATES="8"
  LR_CANDIDATES="1e-5 2e-5"

#elif [ $model_type = "deberta" ]; then
#  model_name_or_path="microsoft/deberta-v2-xxlarge"
#
#  TRAIN_BATCH_SIZE_CANDIDATES="2"
#  LR_CANDIDATES="1e-5 5e-6"

elif [ $model = "deberta_xxlarge" ]; then
  model_type='deberta'
  model_name_or_path="microsoft/deberta-v2-xxlarge"

  TRAIN_BATCH_SIZE_CANDIDATES="2"
  LR_CANDIDATES="1e-5 5e-6"

elif [ $model = "bert_base" ]; then
  model_type='bert'
  model_name_or_path="bert-base-uncased"

  TRAIN_BATCH_SIZE_CANDIDATES="8"
  LR_CANDIDATES="1e-5 2e-5"

fi

echo Running with the following parameters:
echo ------------------------------------
echo DATASET_NAME           = "$dataset_name"
echo TASK_NAME              = "$task_name"
echo METHOD                 = "$method"
echo DEVICE                 = "$device"
echo MODEL_TYPE             = "$model_type"
echo MODEL_NAME_OR_PATH     = "$model_name_or_path"
echo DATA_ROOT              = "$data_dir"
echo SAVE_DIR               = "$save_dir"
echo ------------------------------------


SEQ_LENGTH=256
TOTAL_BATCH_SIZE_CANDIDATES="16"
EVAL_BATCH_SIZE=32
DATA_ROOT=$data_dir
TASK=$task_name

if [ $TASK = "wic" ]; then
  DATA_DIR=${DATA_ROOT}WiC

elif [ $TASK = "rte" ]; then
  DATA_DIR=${DATA_ROOT}RTE

elif [ $TASK = "cb" ]; then
  DATA_DIR=${DATA_ROOT}CB

elif [ $TASK = "wsc" ]; then
  DATA_DIR=${DATA_ROOT}WSC
  SEQ_LENGTH=128

elif [ $TASK = "boolq" ]; then
  DATA_DIR=${DATA_ROOT}BoolQ

elif [ $TASK = "copa" ]; then
  DATA_DIR=${DATA_ROOT}COPA
  SEQ_LENGTH=96

elif [ $TASK = "multirc" ]; then
  DATA_DIR=${DATA_ROOT}MultiRC
  SEQ_LENGTH=512
  EVAL_BATCH_SIZE=16
  TRAIN_BATCH_SIZE_CANDIDATES="1"

elif [ $TASK = "record" ]; then
  DATA_DIR=${DATA_ROOT}ReCoRD
  SEQ_LENGTH=512
  EVAL_BATCH_SIZE=16
  TRAIN_BATCH_SIZE_CANDIDATES="1"
else
  echo "Task " $TASK " is not supported by this script."
  exit 1
fi


WARMUP_RATIO="0.0"
SAMPLER_SEED="42"
#SEED="42"
SEEDS="43 44 45 46"
#SEEDS="46"
#SEEDS="44 45 46"
#SEEDS="45 46"
cv_k="4"
every_eval_ratios="1.0"
MAX_STEPS="5000 2500"
split_ratio="0.5"
NUM_SHOTS="8 16 32 64"
#NUM_SHOTS="16 32 64"
#NUM_SHOTS="8"
#NUM_SHOTS="16"
#NUM_SHOTS="32"
#NUM_SHOTS="4"
#NUM_SHOTS="32 64"


for SEED in $SEEDS
do
  for SHOT in $NUM_SHOTS
  do
    GPU=$((device-1))
    for MAX_STEP in $MAX_STEPS
    do
      for TOTAL_TRAIN_BATCH in $TOTAL_BATCH_SIZE_CANDIDATES
      do
        for TRAIN_BATCH_SIZE in $TRAIN_BATCH_SIZE_CANDIDATES
        do
          for LR in $LR_CANDIDATES
          do
            for every_eval_ratio in $every_eval_ratios
            do
            ACCU=$((${TOTAL_TRAIN_BATCH}/${TRAIN_BATCH_SIZE}))
            ((GPU++))
            HYPER_PARAMS=${SEQ_LENGTH}_${MAX_STEP}_${TOTAL_TRAIN_BATCH}_${TRAIN_BATCH_SIZE}_${LR}_${every_eval_ratio}_${cv_k}_${SHOT}_${SEED}
            OUTPUT_DIR=$save_dir/${HYPER_PARAMS}

            CUDA_VISIBLE_DEVICES=$GPU python3 cli.py \
              --method $method \
              --arch_method $arch_method \
              --data_dir $DATA_DIR \
              --pattern_ids 0 \
              --model_type $model_type \
              --model_name_or_path $model_name_or_path \
              --dataset_name $dataset_name \
              --task_name $task_name \
              --output_dir $OUTPUT_DIR \
              --do_eval \
              --do_train \
              --per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
              --per_gpu_train_batch_size $TRAIN_BATCH_SIZE \
              --gradient_accumulation_steps $ACCU \
              --max_seq_length $SEQ_LENGTH \
              --max_steps $MAX_STEP \
              --sampler_seed $SAMPLER_SEED \
              --seed $SEED \
              --warmup_step_ratio $WARMUP_RATIO \
              --learning_rate $LR \
              --repetitions 1 \
              --few_shot_setting $few_shot_setting \
              --every_eval_ratio $every_eval_ratio \
              --cv_k $cv_k \
              --overwrite_output_dir \
              --split_ratio $split_ratio \
              --fix_deberta \
              --num_shots $SHOT & #>myout_${few_shot_setting}_${method}_${task_name}.file 2>&1 &
    #          wait
            done
          done
        done
      done
    done
    wait
    done
    wait
done


# for ALBERT:
# nohup bash search_cls_multisplit.sh boolq 0 albert >myout.file 2>&1 &
# nohup bash search_cls_multisplit.sh rte 1 albert >myout.file 2>&1 &
# nohup bash search_cls_multisplit.sh wic 2 albert >myout.file 2>&1 &
# nohup bash search_cls_multisplit.sh cb 3 albert >myout.file 2>&1 &
# nohup bash search_cls_multisplit.sh multirc 4 albert >myout.file 2>&1 &
# nohup bash search_cls_multisplit.sh copa 5 albert >myout.file 2>&1 &
# nohup bash search_cls_multisplit.sh wsc 6 albert >myout.file 2>&1 &

# for DeBERTa:
# nohup bash search_cls_multisplit.sh boolq 0 deberta >myout.file 2>&1 &
# nohup bash search_cls_multisplit.sh rte 1 deberta >myout.file 2>&1 &
# nohup bash search_cls_multisplit.sh wic 2 deberta >myout.file 2>&1 &
# nohup bash search_cls_multisplit.sh cb 3 deberta >myout.file 2>&1 &
# nohup bash search_cls_multisplit.sh multirc deberta >myout.file 2>&1 &
# nohup bash search_cls_multisplit.sh copa 5 deberta >myout.file 2>&1 &
# nohup bash search_cls_multisplit.sh wsc 6 deberta >myout.file 2>&1 &






