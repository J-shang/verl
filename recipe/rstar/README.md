## Installation
After install verl, run the following script to install the code judge server.

```bash
bash recipe/rstar/install.sh
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

We use gsm8k as an example to show how to process data.

```bash
python examples/data_preprocess/gsm8k_rstar_tool_agent_loop.py
```

## Download Model

```bash
huggingface-cli download Qwen/Qwen3-8B --local-dir models/Qwen3-8B
```

## Run an Example

```bash
bash recipe/rstar/run_qwen3-8b_gsm8k_jupyter_tool_agent_weave.sh
```
