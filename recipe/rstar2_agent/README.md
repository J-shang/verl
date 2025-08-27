## Installation
After install verl, run the following script to install the code judge server.

```bash
bash verl/recipe/rstar2_agent/install.sh
```

## Launch Code Judge Server

```bash
# start redis
redis-server --daemonize yes --protected-mode no --bind 0.0.0.0

# start code judge server, only needed on the master node.
# it receives requests and submits them to redis, after the worker completes the execution, it will get the request execution result in redis and return it.
# some useful environment setting can be found at https://github.com/0xWJ/code-judge/blob/main/app/config.py
# remember modify the $WORKSPACE and $MASTER_ADDR as you need.
tmux new-session -d -s server 'cd $WORKSPACE/code-judge && MAX_EXECUTION_TIME=4 REDIS_URI="redis://$MASTER_ADDR:6379" RUN_WORKERS=0 uvicorn app.main:app --host 0.0.0.0 --port 8088 --workers 16 2>&1 | tee server.log'

# start code judge workers, it can be launch on each node to increase parallelism, modify MAX_WORKERS with a suitable value with your CPU number on each node.
# remember modify the $WORKSPACE and $MASTER_ADDR as you need.
tmux new-session -d -s worker 'cd $WORKSPACE/code-judge && MAX_EXECUTION_TIME=4 REDIS_URI="redis://$MASTER_ADDR:6379" MAX_WORKERS=64 python run_workers.py 2>&1 | tee worker.log'
```

## Prepare Example Data

We use dapo-17k English part as train dataset and aime24 as test dataset as an example to show how to process data.

```bash
python verl/examples/data_preprocess/aime2024_rstar2_agent_loop.py
python verl/examples/data_preprocess/dapo_rstar2_agent_loop.py
```

## Download Model

Directly use the qwen3 14b base model as a runnable model.
In actual experiments, the base model cannot follow the instructions well and needs instruction-following SFT before RL training.

```bash
huggingface-cli download Qwen/Qwen3-14B-Base --local-dir $HOME/models/Qwen3-14B-Base
```

## Run Example

This script can run on 8 A100/H100, please adjust the configuration according to the running environment.

```bash
bash verl/recipe/rstar2_agent/run_qwen3-14b_rstar2_agent_weave.sh
```

The following settings are about resample of correct and reject sampling.

```bash
# global flag to enable the down sampling
augmentation.do_down_sampling=True
# if enable the reject equal reward sampling
augmentation.down_sampling_config.reject_equal_reward=True
# if enable resample of correct by the toolcall error ratio
augmentation.down_sampling_config.roc_error_ratio=True
# if enable resample of correct by the answer format
augmentation.down_sampling_config.roc_answer_format=True
# at least how many negative rollout traces for each data should be retained
augmentation.down_sampling_config.min_zero_reward_trace_num=2
# at least how many positive rollout traces for each data should be retained
augmentation.down_sampling_config.min_non_zero_reward_trace_num=2
# how many rollout traces for each data should be retained after downsample
augmentation.down_sampling_config.down_sample_to_n=16
```
