# Playing Othello(Reversi) By Reinforcement Learning #

## Introduction ##
This is a simple application that learns to play Othello by
*reinforcement learning*.

TD(0) is used to evaluate a policy.

Value approximation function is based on *n-tuple network* introduced
in Wojciech's paper.

## Quick Start ##

Run `python tdl.py` to learn a policy by self-play.

Edit `config/config.ini` to setup players and run `python run.py` to
play Othello in command line.

Or you can try the simple web app:
  * Run `npm install && npm run build` in `web/ui`.
  * Install `gevent` and `flask`: `pip install gevent flask`
  * Run `python run_server.py`
  * Open [http://localhost:44399/othello](http://localhost:44399/othello) and play!

## Reference ##
- Jaśkowski, Wojciech (2014). Systematic n-tuple networks for
  othello position evaluation. ICGA Journal, 37(2), 85–96.

- Sutton, R. S., & Barto, A. G. (1998). Reinforcement learning: an
  introduction. : MIT press Cambridge.

```bash
# 正常训练
PYTHONUNBUFFERED=1 python -u tdl.py \
  --train-depth 1 --train-width 2 \
  --eval-depth 3 --eval-width 6 --eval-games 200 \
  --episodes 200000 --log-every 1000 --eval-every 20000 --ckpt-every 20000 \
  2>&1 | tee train.log

# 采集模式（只落数据，不训练）
PYTHONUNBUFFERED=1 python -u tdl.py \
  --load model/model_ep180000.cpt.npy \
  --episodes 1000000 \
  --train-depth 1 --train-width 2 \
  --collect-out data/v3/train_mc \
  --collect-max-rows 200000 \
  --opening-random-moves 6 \
  --collect-eps 0.05 \
  --mc-reward --mc-mode current \
  --stop-dup-rate 0.6 --stop-min-seen 500000 \
  2>&1 | tee collect_mc.log

# 检查采样数据集

# 检查前 5 个分片、每片抽样 3 条看看长啥样：
python check_dataset.py --data-dir data/v3/train_mc --limit-shards 5 --sample 3

# 全量扫目录
python check_dataset.py --data-dir data/v2/train

# 若你想精确统计“跨分片全局重复”
python check_dataset.py --data-dir data/v2/train --strict-global-unique


# 监督训练

python -u train_resnet_supervised.py \
  --train-dir data/v3/train_mc \
  --val-dir   data/v3/train_mc \
  --ckpt-dir  ckpts_phaseA \
  --amp

python -u train_resnet_supervised.py \
  --train-dir data/v3/train_mc \
  --val-dir   data/v3/train_mc \
  --ckpt-dir  ckpts_phaseB \
  --alpha-policy 1.0 --beta-value 1.5 \
  --arena-opening-plies 8 --arena-nn-eps 0.02 \
  --amp

```
