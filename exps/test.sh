#!/bin/bash
# test a model to segment abdominal/cardiac MRI
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

###### Shared configs ######
DATASET='CHAOST2'
NWORKER=0
RUNS=1
ALL_EV=(2) # 5-fold cross validation (0, 1, 2, 3, 4)
TEST_LABEL=(1,4)
NSTEP=100
DECAY=0.98

MAX_ITER=100 # defines the size of an epoch
SNAPSHOT_INTERVAL=100 # interval for saving snapshot
SEED=2024
N_PART=3 # defines the number of chunks for evaluation
ALL_SUPP=(3) # CHAOST2: 0-4, CMR: 0-7

echo ========================================================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
  PREFIX="test_${DATASET}_cv${EVAL_FOLD}"
  echo $PREFIX
  LOGDIR="./results"

  if [ ! -d $LOGDIR ]
  then
    mkdir -p $LOGDIR
  fi
  for SUPP_IDX in "${ALL_SUPP[@]}"
  do
    # RELOAD_PATH='please feed the absolute path to the trained weights here' # path to the reloaded model
    RELOAD_MODEL_PATH="./exps_on_CHAOST2/CATNet_train_CHAOST2_cv${EVAL_FOLD}/1/snapshots/100000.pth"
    
    python test.py with \
    mode="test" \
    dataset=$DATASET \
    num_workers=$NWORKER \
    n_steps=$NSTEP \
    eval_fold=$EVAL_FOLD \
    max_iters_per_load=$MAX_ITER \
    supp_idx=$SUPP_IDX \
    test_label=$TEST_LABEL \
    seed=$SEED \
    n_part=$N_PART \
    reload_model_path=$RELOAD_MODEL_PATH \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR
    done
done

echo done running!