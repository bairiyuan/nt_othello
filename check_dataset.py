# -*- coding: utf-8 -*-
"""
check_dataset.py — 检查自博弈采集的数据分片 (.npz)

功能：
1) 逐分片检查数组形状与 dtype、值域（a∈[0,64]、d∈{0,1}、legal 掩码是 0/1）
2) 统计每片与全局的行数、pass 比例、终局比例、奖励分布、动作直方图
3) 计算严格唯一键 (canonical(s), a) 的 64-bit blake2b 哈希，检测：
   - 分片内是否有重复（理论上应为 0）
   - 全局是否有重复（同一运行应为 0；跨运行会出现）
4) 输出一个汇总 CSV（默认：<数据目录>/dataset_report.csv）
5) 可选随机抽样打印若干条样本信息

用法示例：
python check_dataset.py --data-dir data/v2/train --limit-shards 5 --sample 3
python check_dataset.py --data-dir data/v2/train --report-csv report.csv
"""

import argparse
import os
from pathlib import Path
import numpy as np
import hashlib
import csv
import random
from typing import Dict, Any, Tuple

# --------------------------- 与采集端一致的哈希定义 ---------------------------

def hash64_sa(state_3ch_uint8: np.ndarray, a_idx: int) -> int:
    """
    输入：
      state_3ch_uint8: (3,8,8) uint8
      a_idx: int in [0..64]
    输出：64-bit 无符号整型（Python int）
    """
    assert state_3ch_uint8.shape == (3,8,8) and state_3ch_uint8.dtype == np.uint8
    assert 0 <= a_idx <= 64
    data = np.packbits(state_3ch_uint8, axis=None).tobytes() + bytes([a_idx & 0xFF])
    h = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(h, byteorder="big", signed=False)

# --------------------------- 工具函数 ---------------------------

def int_hist(arr: np.ndarray, nbins: int) -> np.ndarray:
    """对非负整数数组做直方图计数（[0..nbins-1]），超界自动忽略"""
    hist = np.zeros((nbins,), dtype=np.int64)
    mask = (arr >= 0) & (arr < nbins)
    idx = arr[mask]
    if idx.size:
        binc = np.bincount(idx, minlength=nbins)
        hist[:len(binc)] += binc
    return hist

def is_binary_array(x: np.ndarray) -> bool:
    return np.all((x == 0) | (x == 1))

# --------------------------- 主检查逻辑 ---------------------------

def check_shard(npz_path: Path,
                global_hashes: set,
                strict_global_unique: bool = False,
                sample: int = 0,
                rnd: random.Random = None) -> Dict[str, Any]:
    """
    检查单个 npz 分片，返回统计信息。必要时检测全局重复（需要传入 global_hashes）

    strict_global_unique=False 时只做“计数 + 报警”（更快）
    strict_global_unique=True 时会把每条样本的哈希插到 global_hashes（更慢但更准确）
    """
    stat: Dict[str, Any] = {
        "file": npz_path.name,
        "rows": 0,
        "shape_ok": True,
        "dtype_ok": True,
        "values_ok": True,
        "has_legal": False,
        "in_shard_dups": 0,
        "global_dups": 0,
        "pass_ratio": 0.0,
        "done_ratio": 0.0,
        "reward_mean": 0.0,
        "reward_hist_-1/0/+1": (0, 0, 0),
    }

    z = np.load(npz_path, mmap_mode="r", allow_pickle=False)

    # 必需字段
    for key in ["s", "a", "s_next", "r", "d"]:
        if key not in z:
            raise ValueError(f"{npz_path} 缺少关键数组: {key}")

    s = z["s"]           # (N,3,8,8) uint8
    a = z["a"]           # (N,) int16
    sp = z["s_next"]     # (N,3,8,8) uint8
    r = z["r"]           # (N,) float32
    d = z["d"]           # (N,) uint8

    N = s.shape[0]
    stat["rows"] = int(N)

    # 形状检查
    shape_ok = (
        s.ndim == 4 and s.shape[1:] == (3,8,8) and
        sp.ndim == 4 and sp.shape[1:] == (3,8,8) and
        a.ndim == 1 and a.shape[0] == N and
        r.ndim == 1 and r.shape[0] == N and
        d.ndim == 1 and d.shape[0] == N
    )
    stat["shape_ok"] = bool(shape_ok)

    # dtype & 值域检查
    dtype_ok = (s.dtype == np.uint8 and sp.dtype == np.uint8 and
                a.dtype.kind in ("i", "u") and
                r.dtype.kind in ("f", "i", "u") and
                d.dtype == np.uint8)
    stat["dtype_ok"] = bool(dtype_ok)

    values_ok = True
    if N > 0:
        if not (np.all((a >= 0) & (a <= 64))):
            values_ok = False
        if not (np.all((d == 0) | (d == 1))):
            values_ok = False
        # 合法掩码（可选）
        has_legal = ("legal_s" in z) or ("legal_s_next" in z)
        stat["has_legal"] = bool(has_legal)
        if "legal_s" in z:
            ls = z["legal_s"]   # (N,64) uint8
            if not (ls.ndim == 2 and ls.shape == (N,64) and is_binary_array(ls)):
                values_ok = False
        if "legal_s_next" in z:
            lsp = z["legal_s_next"]
            if not (lsp.ndim == 2 and lsp.shape == (N,64) and is_binary_array(lsp)):
                values_ok = False
    stat["values_ok"] = bool(values_ok)

    if N == 0:
        return stat  # 空分片直接返回

    # 基本分布
    pass_ratio = float(np.mean(a == 64))
    done_ratio = float(np.mean(d == 1))
    stat["pass_ratio"] = pass_ratio
    stat["done_ratio"] = done_ratio

    # 奖励分布（-1,0,1 统计）
    # 注意：中间步奖励是 0，终局奖励在 {-1, +1}（按你的采集定义）
    r_arr = np.asarray(r)
    r_neg = int(np.sum(r_arr < -0.5))
    r_zero = int(np.sum(np.abs(r_arr) <= 0.5))
    r_pos = int(np.sum(r_arr > 0.5))
    stat["reward_mean"] = float(np.mean(r_arr))
    stat["reward_hist_-1/0/+1"] = (r_neg, r_zero, r_pos)

    # 动作直方图（仅 0..63，pass=64 单独统计）
    a_arr = np.asarray(a)
    act_hist = int_hist(a_arr[a_arr < 64], 64)
    stat["act_hist_0_63"] = act_hist.tolist()
    stat["act_pass_count"] = int(np.sum(a_arr == 64))

    # 分片内重复（哈希）
    # 为避免大内存占用，这里分块计算
    in_shard_hashes = set()
    chunk = 8192
    in_shard_dups = 0
    global_dups = 0

    for start in range(0, N, chunk):
        end = min(N, start + chunk)
        s_slice = s[start:end]      # (M,3,8,8)
        a_slice = a[start:end]      # (M,)

        # 逐条计算 64-bit 键
        for k in range(s_slice.shape[0]):
            h = hash64_sa(s_slice[k], int(a_slice[k]))
            if h in in_shard_hashes:
                in_shard_dups += 1
            else:
                in_shard_hashes.add(h)
            if strict_global_unique:
                if h in global_hashes:
                    global_dups += 1
                else:
                    global_hashes.add(h)

    stat["in_shard_dups"] = int(in_shard_dups)
    stat["global_dups"] = int(global_dups)

    # 随机抽样输出（可选）
    if sample > 0:
        rnd = rnd or random.Random(2025)
        idxs = rnd.sample(range(N), min(sample, N))
        stat["samples"] = []
        for idx in sorted(idxs):
            rec = {
                "idx": int(idx),
                "a": int(a[idx]),
                "r": float(r[idx]),
                "d": int(d[idx]),
                "s_sum": int(np.sum(s[idx])),
                "s_next_sum": int(np.sum(sp[idx])),
            }
            stat["samples"].append(rec)

    return stat

def scan_dir(data_dir: Path) -> Tuple[list, int]:
    files = sorted(data_dir.glob("**/shard_*.npz"))
    return files, len(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, required=True, help="数据目录（包含 shard_XXXXX.npz）")
    ap.add_argument("--limit-shards", type=int, default=0, help="只检查前 N 个分片（0=不限制）")
    ap.add_argument("--strict-global-unique", action="store_true", help="开启全局重复精查（会更慢、占更多内存）")
    ap.add_argument("--sample", type=int, default=0, help="每个分片随机打印样本条数（0=不打印）")
    ap.add_argument("--report-csv", type=str, default="", help="汇总 CSV 输出路径（默认写到 data-dir/dataset_report.csv）")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"目录不存在：{data_dir}")

    files, nfiles = scan_dir(data_dir)
    if args.limit_shards > 0:
        files = files[:args.limit_shards]

    print(f"[info] will check {len(files)} shard(s) under: {data_dir}")

    global_hashes = set()
    rnd = random.Random(2025)

    # 汇总统计
    tot_rows = 0
    tot_pass = 0
    tot_done = 0
    tot_r_neg = tot_r_zero = tot_r_pos = 0
    global_dups_all = 0
    in_shard_dups_all = 0
    act_hist_global = np.zeros((64,), dtype=np.int64)
    pass_global = 0

    rows_list = []

    for i, f in enumerate(files, 1):
        stat = check_shard(
            f, global_hashes,
            strict_global_unique=args.strict_global_unique,
            sample=args.sample,
            rnd=rnd
        )

        # 打印每片的核心信息
        print(f"[shard {i}/{len(files)}] {stat['file']}: rows={stat['rows']} "
              f"shape_ok={stat['shape_ok']} dtype_ok={stat['dtype_ok']} values_ok={stat['values_ok']} "
              f"in_shard_dups={stat['in_shard_dups']} global_dups={stat['global_dups']} "
              f"pass={stat['pass_ratio']:.3f} done={stat['done_ratio']:.3f} "
              f"reward_mean={stat['reward_mean']:.4f}")

        # 抽样打印
        if args.sample > 0 and stat.get("samples"):
            for rec in stat["samples"]:
                print(f"  sample idx={rec['idx']:>6}  a={rec['a']:>2}  r={rec['r']:+.1f}  d={rec['d']}  "
                      f"s_sum={rec['s_sum']}  s_next_sum={rec['s_next_sum']}")

        # 累加全局
        N = stat["rows"]
        tot_rows += N
        in_shard_dups_all += stat["in_shard_dups"]
        global_dups_all += stat["global_dups"]

        # 奖励分布
        rn, rz, rp = stat["reward_hist_-1/0/+1"]
        tot_r_neg += rn; tot_r_zero += rz; tot_r_pos += rp

        # pass、done
        tot_pass += int(N * stat["pass_ratio"])
        tot_done += int(N * stat["done_ratio"])

        # 动作直方图
        if "act_hist_0_63" in stat:
            act_hist_global += np.asarray(stat["act_hist_0_63"], dtype=np.int64)
        pass_global += stat.get("act_pass_count", 0)

        rows_list.append(stat)

    # 汇总打印
    if tot_rows > 0:
        pass_ratio_glob = tot_pass / tot_rows
        done_ratio_glob = tot_done / tot_rows
        print("\n===== SUMMARY =====")
        print(f"files: {len(files)}")
        print(f"rows_total: {tot_rows}")
        print(f"in_shard_dups_total: {in_shard_dups_all}")
        print(f"global_dups_total: {global_dups_all}  (启用 --strict-global-unique 才会精确统计)")
        print(f"pass_ratio_global: {pass_ratio_glob:.3f}")
        print(f"done_ratio_global: {done_ratio_glob:.3f}")
        print(f"reward_hist_global (-1/0/+1): {tot_r_neg}/{tot_r_zero}/{tot_r_pos}")
        print(f"action_histogram[0..63] (sum={act_hist_global.sum()}, pass={pass_global})")
        # 可选：打印前 10 个最常见落点
        topk = 10
        top_idx = np.argsort(-act_hist_global)[:topk]
        print("top actions:", [(int(idx), int(act_hist_global[idx])) for idx in top_idx])

    # 写 CSV 报告
    report_path = Path(args.report_csv) if args.report_csv else (data_dir / "dataset_report.csv")
    with report_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = [
            "file","rows","shape_ok","dtype_ok","values_ok",
            "in_shard_dups","global_dups",
            "pass_ratio","done_ratio","reward_mean",
            "reward_neg","reward_zero","reward_pos",
            "has_legal"
        ]
        writer.writerow(header)
        for s in rows_list:
            rn, rz, rp = s["reward_hist_-1/0/+1"]
            writer.writerow([
                s["file"], s["rows"], s["shape_ok"], s["dtype_ok"], s["values_ok"],
                s["in_shard_dups"], s["global_dups"],
                f"{s['pass_ratio']:.6f}", f"{s['done_ratio']:.6f}", f"{s['reward_mean']:.6f}",
                rn, rz, rp, s["has_legal"]
            ])
    print(f"\n[ok] report saved -> {report_path}")

if __name__ == "__main__":
    main()
