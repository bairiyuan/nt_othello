# -*- coding: utf-8 -*-
import logging
import time, argparse, os, sys, random
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np

from othello import Board
from util import epsilon_greedy
from value import ModelScorer
from ai import Bot

# --------------------------- 日志 ---------------------------

def setup_logging(logfile: str, verbose: bool = True):
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)

    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    root.addHandler(fh)

    if verbose:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        root.addHandler(sh)

# --------------------------- 工具：坐标/编码/对称 ---------------------------

# 将 (i,j) 映射到 0..63，pass=64
def ij_to_a(i: int, j: int) -> int:
    if i < 0 or j < 0:
        return 64
    return i * 8 + j

def a_to_ij(a: int) -> Tuple[int, int]:
    if a == 64:
        return (-1, -1)
    return a // 8, a % 8

# 把 Board 编码到 (3,8,8) uint8：channel0=当前执手子、channel1=对手子、channel2=全 1（Turn）
def board_to_3ch_uint8(b: Board, p: int) -> np.ndarray:
    # Board 内部约定：黑=1 白=-1（常见做法），也可能是不同常量；用 API 来拿
    # 这里用 score/可落子来侧推不稳妥，因此直接走棋盘格子读值
    # 假设 b._board[i,j] in {Board.BLACK, Board.WHITE, 0}
    # 若你的 Board 没公开 _board，可增加 get_cell(i,j) 等接口；此处按常见实现读取
    arr = np.zeros((3, 8, 8), dtype=np.uint8)
    me, opp = p, Board.opponent(p)
    for i in range(8):
        for j in range(8):
            v = b._board[i, j]  # 若你的 Board 提供别的访问方式，请改这里
            if v == me:
                arr[0, i, j] = 1
            elif v == opp:
                arr[1, i, j] = 1
    arr[2, :, :] = 1  # turn 平面，表示“从当前执手视角”
    return arr

# 生成 8 个 D4 对称（旋转/翻转）
def d4_variants(x: np.ndarray) -> List[np.ndarray]:
    # x: (3,8,8)
    xs = []
    base = x
    for k in range(4):
        rot = np.rot90(base, k, axes=(1, 2))
        xs.append(rot)
        xs.append(np.flip(rot, axis=2))  # 水平翻转
    return xs

def canonicalize_3ch(x: np.ndarray) -> np.ndarray:
    # 返回字典序最小的形态，降低重复
    cand = d4_variants(x)
    keys = [np.packbits(c, axis=None).tobytes() for c in cand]
    idx = int(np.argmin(keys))
    return cand[idx]

def legal_mask_from_board(b: Board, p: int) -> np.ndarray:
    m = np.zeros((64,), dtype=np.uint8)
    options = b.feasible_pos(p)
    for (i, j) in options:
        m[ij_to_a(i, j)] = 1
    return m

def hash64_sa(state_3ch_uint8: np.ndarray, a_idx: int) -> int:
    data = np.packbits(state_3ch_uint8, axis=None).tobytes() + bytes([a_idx & 0xFF])
    h = np.frombuffer(
        memoryview(
            __import__("hashlib").blake2b(data, digest_size=8).digest()
        ),
        dtype=">u8",
    )[0]
    return int(h)

# --------------------------- 评测（vs 随机） ---------------------------

def evaluate_vs_random(model, games: int = 200, seed: int = 123,
                       depth: int = 3, width: int = 6) -> dict:
    rng = random.Random(seed)
    games = games + (games % 2)

    def _play_single_game(black_is_model: bool) -> int:
        b = Board(); b.init_board(); p = Board.BLACK
        black_bot = Bot(model, depth, width, Board.BLACK)
        white_bot = Bot(model, depth, width, Board.WHITE)
        while not b.is_terminal_state():
            options = b.feasible_pos(p)
            if options:
                if (p == Board.BLACK and black_is_model) or (p == Board.WHITE and not black_is_model):
                    if p == Board.BLACK:
                        _, (a0, a1) = black_bot._play(b)
                    else:
                        _, (a0, a1) = white_bot._play(b)
                    b.flip(a0, a1, p)
                else:
                    i, j = rng.choice(options)
                    b.flip(i, j, p)
            p = Board.opponent(p)
        sb, sw = b.score(Board.BLACK), b.score(Board.WHITE)
        result_black = 1 if sb > sw else (-1 if sb < sw else 0)
        return result_black if black_is_model else -result_black

    wins = losses = draws = 0
    for _ in range(0, games, 2):
        r1 = _play_single_game(True)
        r2 = _play_single_game(False)
        for r in (r1, r2):
            if r > 0: wins += 1
            elif r < 0: losses += 1
            else: draws += 1

    n = wins + losses + draws
    wr = wins / n if n else 0.0

    import math
    z = 1.96
    phat = wr
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2*n)) / denom
    margin = (z/denom) * math.sqrt((phat*(1-phat)/n) + (z**2/(4*n**2)))
    return {
        "winrate": wr, "wins": wins, "losses": losses, "draws": draws, "games": n,
        "ci_low": max(0.0, centre - margin), "ci_high": min(1.0, centre + margin)
    }

# --------------------------- 自博弈训练 ---------------------------

def self_play(episodes, model, epsilon, log_every, eval_every, ckpt_every,
              ckpt_dir, save_latest, seed, train_depth, train_width, args ,
              stop_winrate=0.95):
    rng = random.Random(seed)
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_path = Path(save_latest)
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    b = Board()
    black_bot = Bot(model, train_depth, train_width, Board.BLACK)
    white_bot = Bot(model, train_depth, train_width, Board.WHITE)

    t0 = time.time()
    stat_w = stat_l = stat_d = 0

    for t in range(1, episodes + 1):
        b.init_board()
        p = Board.BLACK

        while not b.is_terminal_state():
            options = b.feasible_pos(p)
            vals = []
            if len(options) > 0:
                if p == Board.BLACK:
                    gr, (ga0, ga1) = black_bot._play(b)
                else:
                    gr, (ga0, ga1) = white_bot._play(b)
                    gr = -gr

                for i, j in options:
                    with b.flip2(i, j, p):
                        if b.is_terminal_state():
                            vals.append(b.score(Board.BLACK) - b.score(Board.WHITE))
                        else:
                            vals.append(model(b))

                (a0, a1), v = epsilon_greedy(epsilon, options, vals, (ga0, ga1), gr)
                model.update(b, v)
                b.flip(a0, a1, p)

            p = Board.opponent(p)

        sb = b.score(Board.BLACK)
        sw = b.score(Board.WHITE)
        if sb > sw: stat_w += 1
        elif sb < sw: stat_l += 1
        else: stat_d += 1

                # ========== 新增：胜率停止检查 ==========
        if eval_every > 0 and t % eval_every == 0:
            eval_result = evaluate_vs_random(
                model,
                games=args.eval_games,
                seed=seed + t,  # 使用不同的种子
                depth=args.eval_depth,
                width=args.eval_width
            )
            current_winrate = eval_result["winrate"]
            
            logging.info(f"[stop-check] games={t}, winrate={current_winrate:.3f}, target={stop_winrate}")
            
            # 检查停止条件
            if current_winrate >= stop_winrate:
                logging.info(f"[stop] 🎯 达到目标胜率 {current_winrate:.3f} >= {stop_winrate}, 停止训练!")
                
                # 保存最终模型
                try:
                    model.save(str(save_latest))
                    logging.info(f"[ckpt] 最终模型已保存 -> {save_latest}")
                except Exception as e:
                    logging.exception(f"[ckpt] 最终保存失败: {e}")
                
                return  # 提前结束训练
            # ========== 新增：胜率停止检查 ==========

        if t % log_every == 0:
            elapsed = time.time() - t0
            n_batch = stat_w + stat_l + stat_d
            wr = stat_w / n_batch if n_batch else 0.0
            speed = log_every / max(1e-9, elapsed)
            logging.info(
                f"[train] games={t}/{episodes}  winrate={wr:.3f}  "
                f"w/l/d={stat_w}/{stat_l}/{stat_d}  speed={speed:.1f} games/s  eps={epsilon:.3f}"
            )
            stat_w = stat_l = stat_d = 0
            t0 = time.time()

        if eval_every > 0 and t % eval_every == 0:
            ev = evaluate_vs_random(model,
                                    games=args.eval_games,
                                    seed=seed,
                                    depth=args.eval_depth,
                                    width=args.eval_width)
            logging.info(
                f"[eval] games={ev['games']}  winrate={ev['winrate']:.3f} "
                f"95%CI=[{ev['ci_low']:.3f},{ev['ci_high']:.3f}]  "
                f"w/l/d={ev['wins']}/{ev['losses']}/{ev['draws']} "
                f"(eval_depth={args.eval_depth}, eval_width={args.eval_width})"
            )

        if ckpt_every > 0 and t % ckpt_every == 0:
            ckpt_path = Path(ckpt_dir) / f"model_ep{t}.cpt.npy"
            try:
                model.save(str(ckpt_path))
                model.save(str(latest_path))
                logging.info(f"[ckpt] saved -> {ckpt_path} (and updated {latest_path.name})")
            except Exception as e:
                logging.exception(f"[ckpt] save failed: {e}")

    try:
        model.save(str(latest_path))
        logging.info(f"[ckpt] final save -> {latest_path}")
    except Exception as e:
        logging.exception(f"[ckpt] final save failed: {e}")

# --------------------------- 采集：写分片 ---------------------------

class ShardWriter:
    def __init__(self, out_dir: str, max_rows: int):
        self.dir = Path(out_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.max_rows = int(max_rows)
        self.shard_idx = 0
        self.total = 0
        self.buf: Dict[str, list] = {
            "s": [], "a": [], "s_next": [], "r": [], "d": [],
            "legal_s": [], "legal_s_next": [],
        }

    def add(self, s, a, s_next, r, d, legal_s, legal_s_next):
        self.buf["s"].append(s)
        self.buf["a"].append(np.int16(a))
        self.buf["s_next"].append(s_next)
        self.buf["r"].append(np.float32(r))
        self.buf["d"].append(np.uint8(d))
        self.buf["legal_s"].append(legal_s)
        self.buf["legal_s_next"].append(legal_s_next)

    def flush(self):
        if len(self.buf["a"]) == 0:
            return
        n = len(self.buf["a"])
        out = {}
        for k, vs in self.buf.items():
            v0 = vs[0]
            if isinstance(v0, np.ndarray):
                out[k] = np.stack(vs)
            else:
                out[k] = np.asarray(vs)
        path = self.dir / f"shard_{self.shard_idx:05d}.npz"
        np.savez_compressed(path, **out)
        self.total += n
        logging.info(f"[collect] wrote {n} rows -> {path.name} (total={self.total})")
        self.shard_idx += 1
        for k in self.buf.keys():
            self.buf[k].clear()

# --------------------------- 采集主循环（含 MC 回填） ---------------------------

def collect_selfplay(episodes, model, depth, width, out_dir, max_rows, write_legal=True,
                     opening_random_moves=0, collect_eps=0.0,
                     stop_dup_rate=0.95, stop_min_seen=50_000,
                     log_every_games=50,
                     metrics_file="", metrics_every_games=50,
                     mc_reward=False, mc_mode="current",
                     selection_temperature=0.3):
    """
    只采集，不训练。严格唯一：(canonical(s),a)。每局结束后可选 MC 奖励回填。
    """
    logging.info(f"[mode] collect-only -> {out_dir}  (depth={depth}, width={width})")
    writer = ShardWriter(out_dir, max_rows)

    rng = random.Random(2025)
    b = Board()

    keys_seen = set()       # 全局唯一键
    shard_keys = set()      # 分片内唯一键
    rows_in_shard = 0

    # 指标
    seen_total = 0
    unique_total = 0
    games_done = 0
    t0 = time.time()

    def cur_dup_rate():
        return 0.0 if seen_total == 0 else 1.0 - (unique_total / seen_total)

    # 每局缓冲（用于 MC 回填）
    ep_buf: List[dict] = []

    while games_done < episodes:
        b.init_board()
        p = Board.BLACK
        black_bot = Bot(model, depth, width, Board.BLACK)
        white_bot = Bot(model, depth, width, Board.WHITE)

        # 开局随机若干步（不写数据，只扰动起点）
        rnd_moves = int(max(0, opening_random_moves))
        for _ in range(rnd_moves):
            if b.is_terminal_state():
                break
            opts = b.feasible_pos(p)
            if not opts:
                p = Board.opponent(p)
                continue
            i, j = rng.choice(opts)
            b.flip(i, j, p)
            p = Board.opponent(p)

        # 自对弈直到终局，按“采集 ε”做 ε-greedy（对 Bot 的建议）
        while not b.is_terminal_state():
            options = b.feasible_pos(p)
            if len(options) == 0:
                p = Board.opponent(p)
                continue

            # Bot 的建议（贪心动作）
            if p == Board.BLACK:
                _, (greed_i, greed_j) = black_bot._play(b)
            else:
                _, (greed_i, greed_j) = white_bot._play(b)

            # 计算值列表（用不上 v，仅为了和 util.epsilon_greedy 的接口保持一致）
            vals = []
            for (ii, jj) in options:
                with b.flip2(ii, jj, p):
                    if b.is_terminal_state():
                        vals.append(b.score(Board.BLACK) - b.score(Board.WHITE))
                    else:
                        vals.append(model(b))
            
            # # 采集 ε：使用我们自己的 epsilon（与训练 ε 无关）
            # (a0, a1), _ = epsilon_greedy(collect_eps, options, vals, (greed_i, greed_j), 0.0)
            
            # === 修改开始：替换ε-greedy为价值概率选择 ===
            if len(options) > 0:
                # 将估值转换为概率分布
                vals_array = np.array(vals)
    
            # 处理极端值，防止数值不稳定
            vals_array = np.clip(vals_array, -10, 10)
    
            # 应用softmax（温度参数=selection_temperature）
            temperature = max(0.01,selection_temperature)
            exp_vals = np.exp(vals_array / temperature)
            probabilities = exp_vals / np.sum(exp_vals)
    
            # 根据概率分布采样动作
            chosen_idx = np.random.choice(len(options), p=probabilities)
            a0, a1 = options[chosen_idx]
            selected_val = vals_array[chosen_idx]
            # === 修改结束 ===

            # 记录转移（在 flip 前做编码）
            s_3ch = board_to_3ch_uint8(b, p)
            s_can = canonicalize_3ch(s_3ch)
            legal_s = legal_mask_from_board(b, p)

            # 落子
            b.flip(a0, a1, p)
            p_next = Board.opponent(p)

            # 终局判断 + 奖励（即时奖励：仅终局 ±1，否则 0）
            done_now = b.is_terminal_state()
            if done_now:
                sb, sw = b.score(Board.BLACK), b.score(Board.WHITE)
                # 对“当前这步执手”的即时奖励
                reward = 1.0 if (sb > sw and p == Board.BLACK) or (sw > sb and p == Board.WHITE) else \
                         (-1.0 if (sb < sw and p == Board.BLACK) or (sw < sb and p == Board.WHITE) else 0.0)
            else:
                reward = 0.0

            # 下一个状态编码（从下一步执手视角）
            s_next_3ch = board_to_3ch_uint8(b, p_next)
            s_next_can = canonicalize_3ch(s_next_3ch)
            legal_next = legal_mask_from_board(b, p_next)
            a_idx = ij_to_a(a0, a1)

            # 严格唯一键（全局）
            key = hash64_sa(s_can, a_idx)
            seen_total += 1
            if key not in keys_seen:
                keys_seen.add(key)
                rec = dict(
                    s=s_can, a=a_idx, s_next=s_next_can,
                    r=np.float32(reward), d=np.uint8(1 if done_now else 0),
                    legal_s=legal_s if write_legal else np.zeros((64,), np.uint8),
                    legal_s_next=legal_next if write_legal else np.zeros((64,), np.uint8),
                )
                ep_buf.append(rec)

            # 切换执手方
            p = p_next

        # ======== 一局结束：MC 回填 + 写盘 ========
        # 终局胜负（从黑方得分判断；仅用于 uniform 模式）
        sb, sw = b.score(Board.BLACK), b.score(Board.WHITE)
        res_black = 1 if sb > sw else (-1 if sb < sw else 0)

        if mc_reward and len(ep_buf) > 0:
            if mc_mode == "current":
                # 以“终局那步的即时奖励符号”为尾部标签，向前逐步交替取反
                r_end = float(ep_buf[-1]["r"])
                if   r_end >  0.5: z_tail = 1
                elif r_end < -0.5: z_tail = -1
                else:              z_tail = 0
                for j in range(len(ep_buf)-1, -1, -1):
                    ep_buf[j]["r"] = np.float32(z_tail)
                    z_tail = -z_tail
            elif mc_mode == "uniform":
                z_uni = 1 if res_black > 0 else (-1 if res_black < 0 else 0)
                for rec in ep_buf:
                    rec["r"] = np.float32(z_uni)

        # 把本局缓存写入（分片内再做一次去重）
        for rec in ep_buf:
            k = hash64_sa(rec["s"], rec["a"])
            if k in shard_keys:
                continue
            writer.add(
                s=rec["s"], a=rec["a"], s_next=rec["s_next"],
                r=rec["r"], d=rec["d"],
                legal_s=rec["legal_s"], legal_s_next=rec["legal_s_next"]
            )
            unique_total += 1
            shard_keys.add(k)
            rows_in_shard += 1
            if rows_in_shard >= writer.max_rows:
                writer.flush()
                logging.info(
                    f"[collect] games={games_done}/{episodes}  "
                    f"buffered~{rows_in_shard}  flushed_total={writer.total}  "
                    f"seen_total={seen_total} unique_total={unique_total} dup_rate={cur_dup_rate():.3f}  "
                    f"speed={(log_every_games / max(1e-9, (time.time()-t0))):.2f} g/s"
                )
                shard_keys.clear()
                rows_in_shard = 0
                t0 = time.time()

        ep_buf.clear()
        games_done += 1

        # 日志心跳
        if games_done % log_every_games == 0:
            elapsed = time.time() - t0
            speed = log_every_games / max(1e-9, elapsed)
            logging.info(
                f"[collect] games={games_done}/{episodes}  "
                f"buffered~{rows_in_shard}  flushed_total={writer.total}  "
                f"seen_total={seen_total} unique_total={unique_total} dup_rate={cur_dup_rate():.3f}  "
                f"speed={speed:.2f} g/s"
            )
            t0 = time.time()

        # 早停：全局重复率 / 最小 seen
        if (seen_total >= stop_min_seen) and (cur_dup_rate() >= stop_dup_rate):
            logging.info(f"[collect] stop by dup_rate: seen_total={seen_total} unique_total={unique_total} dup_rate={cur_dup_rate():.3f}")
            break

    # 收尾 flush
    if rows_in_shard > 0:
        writer.flush()

# --------------------------- 入口 ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=700_000, help="总自博弈/采集局数上限")
    parser.add_argument("--epsilon", type=float, default=0.05, help="训练 ε-greedy")
    parser.add_argument("--log-every", type=int, default=100, help="训练每多少局打印一次")
    parser.add_argument("--eval-every", type=int, default=10_000, help="训练每多少局评测（0=关闭）")
    parser.add_argument("--ckpt-every", type=int, default=10_000, help="训练每多少局保存（0=关闭）")
    parser.add_argument("--ckpt-dir", type=str, default="model", help="checkpoint 保存目录")
    parser.add_argument("--latest", type=str, default="model/model.cpt.npy", help="最新快照文件路径")
    parser.add_argument("--load", type=str, default="model/model.cpt.npy.6", help="启动时加载的权重（可留空）")
    parser.add_argument("--lr", type=float, default=0.001, help="ModelScorer 学习率")
    parser.add_argument("--gamma", type=float, default=0.01, help="TD 平滑系数（按你实现语义）")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    parser.add_argument("--logfile", type=str, default="tdl.log", help="日志文件")

    # 训练/评测搜索强度分离
    parser.add_argument("--train-depth", type=int, default=1, help="训练 Bot 深度")
    parser.add_argument("--train-width", type=int, default=2, help="训练 Bot 宽度")
    parser.add_argument("--eval-depth", type=int, default=3, help="评测 Bot 深度")
    parser.add_argument("--eval-width", type=int, default=6, help="评测 Bot 宽度")
    parser.add_argument("--eval-games", type=int, default=200, help="评测对局数")

    # 采集模式开关
    parser.add_argument("--collect-out", type=str, default="", help="非空则进入采集模式，输出目录")
    parser.add_argument("--collect-max-rows", type=int, default=200_000, help="每个分片的行数上限")
    parser.add_argument("--opening-random-moves", type=int, default=0, help="采集时开局随机步数")
    parser.add_argument("--collect-eps", type=float, default=0.0, help="采集时 ε-greedy（与训练 ε 无关）")
    parser.add_argument("--stop-dup-rate", type=float, default=0.95, help="采集早停：全局重复率阈值")
    parser.add_argument("--stop-min-seen", type=int, default=50_000, help="采集早停：最小观测条数")

    # MC 回填
    parser.add_argument("--mc-reward", action="store_true",
                        help="采集时将本局所有步骤的奖励替换为 Monte-Carlo 终局标签")
    parser.add_argument("--mc-mode", type=str, default="current",
                        choices=["current", "uniform"],
                        help="MC 标签模式：current=当前执手交替取符号；uniform=整局统一标签")
    
    # 新增胜率停止参数
    parser.add_argument("--stop-winrate", type=float, default=0.95, 
                       help="当对随机模型的胜率达到此值时停止训练")
    
    # 新增价值概率选择的温度参数（采集模式专用）
    parser.add_argument("--selection-temperature", type=float, default=0.3,
                        help="价值概率选择的温度参数（采集模式专用）")

    args = parser.parse_args()

    os.environ["PYTHONUNBUFFERED"] = "1"
    setup_logging(args.logfile, verbose=True)

    # 模型
    model = ModelScorer(learning_rate=args.lr, gamma=args.gamma)
    if args.load and Path(args.load).exists():
        try:
            model.load(args.load)
            logging.info(f"[init] loaded weights from {args.load}")
        except Exception as e:
            logging.exception(f"[init] load failed: {e}")
    else:
        logging.info("[init] start from fresh weights")

    # 采集模式
    if args.collect_out:
        collect_selfplay(
            episodes=args.episodes,
            model=model,
            depth=args.train_depth,
            width=args.train_width,
            out_dir=args.collect_out,
            max_rows=args.collect_max_rows,
            write_legal=True,
            opening_random_moves=args.opening_random_moves,
            collect_eps=args.collect_eps,
            stop_dup_rate=args.stop_dup_rate,
            stop_min_seen=args.stop_min_seen,
            log_every_games=50,
            metrics_file="",
            metrics_every_games=50,
            mc_reward=args.mc_reward,
            mc_mode=args.mc_mode,
            selection_temperature=args.selection_temperature,
        )
        # 收尾
        try:
            model.save(args.latest)
            logging.info(f"[ckpt] final save -> {args.latest}")
        except Exception as e:
            logging.exception(f"[ckpt] final save failed: {e}")
        return

    # 训练模式
    self_play(episodes=args.episodes,
              model=model,
              epsilon=args.epsilon,
              log_every=args.log_every,
              eval_every=args.eval_every,
              ckpt_every=args.ckpt_every,
              ckpt_dir=args.ckpt_dir,
              save_latest=args.latest,
              seed=args.seed,
              train_depth=args.train_depth,
              train_width=args.train_width,
              args=args,
              stop_winrate=args.stop_winrate)

if __name__ == '__main__':
    main()
