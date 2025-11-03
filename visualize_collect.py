# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=str, required=True, help="collect_metrics.csv 路径")
    ap.add_argument("--save", type=str, default="", help="保存图片到该目录（留空则直接弹窗显示）")
    args = ap.parse_args()

    path = Path(args.metrics)
    if not path.exists():
        print(f"metrics file not found: {path}")
        return

    df = pd.read_csv(path)
    # 只保留常用指标列
    keep = [c for c in [
        "ts","games_done","dup_rate","unique_total","seen_total",
        "flushed_total","shard_index","speed_gps","event"
    ] if c in df.columns]
    df = df[keep].copy()

    # 基本图 1：dup_rate & speed
    fig1 = plt.figure(figsize=(8,4.5))
    ax1 = plt.gca()
    if "dup_rate" in df.columns:
        df.plot(x="games_done", y="dup_rate", ax=ax1, linewidth=1.5)
    if "speed_gps" in df.columns:
        ax2 = ax1.twinx()
        df.plot(x="games_done", y="speed_gps", ax=ax2, linewidth=1.0, style="--")
        ax2.set_ylabel("speed (games/s)")
    ax1.set_title("Duplicate Rate & Speed")
    ax1.set_xlabel("games_done")
    ax1.set_ylabel("dup_rate")

    # 标注 flush 事件
    if "event" in df.columns and "games_done" in df.columns:
        flush_rows = df[df["event"] == "flush"]
        for _, r in flush_rows.iterrows():
            gd = r["games_done"]
            ax1.axvline(gd, color="gray", linewidth=0.8, alpha=0.5)

    # 基本图 2：unique_total 增长
    fig2 = plt.figure(figsize=(8,4.5))
    ax = plt.gca()
    if "unique_total" in df.columns:
        df.plot(x="games_done", y="unique_total", ax=ax, linewidth=1.5)
        ax.set_title("Unique Samples Accumulation")
        ax.set_xlabel("games_done"); ax.set_ylabel("unique_total")

    # 基本图 3：flushed_total（累计写盘行）
    if "flushed_total" in df.columns:
        fig3 = plt.figure(figsize=(8,4.5))
        ax = plt.gca()
        df.plot(x="games_done", y="flushed_total", ax=ax, linewidth=1.5)
        ax.set_title("Flushed Rows Accumulation")
        ax.set_xlabel("games_done"); ax.set_ylabel("flushed_total")

    if args.save:
        out_dir = Path(args.save); out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fig1.savefig(out_dir / f"dup_rate_speed_{ts}.png", dpi=150, bbox_inches="tight")
        fig2.savefig(out_dir / f"unique_total_{ts}.png", dpi=150, bbox_inches="tight")
        if "flushed_total" in df.columns:
            fig3.savefig(out_dir / f"flushed_total_{ts}.png", dpi=150, bbox_inches="tight")
        print(f"saved figures into {out_dir}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
