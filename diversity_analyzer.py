# diversity_analyzer.py
"""
样本多样性测评工具 - 专注于训练数据的丰富度和新鲜度
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib
import collections
import time
import csv

class DiversityAnalyzer:
    """样本多样性分析器"""
    
    def __init__(self, window_size=5000):
        self.window_size = window_size  # 先设置属性
        self.reset_stats()  # 再调用reset_stats
        
    def reset_stats(self):
        """重置统计信息"""
        self.total_samples = 0
        self.total_games = 0
        self.game_lengths = []
        self.current_game_length = 0
        
        # 状态多样性
        self.unique_states = set()
        self.unique_transitions = set()
        self.recent_states = collections.deque(maxlen=self.window_size)
        
        # 位置多样性
        self.position_visits = np.zeros(64)
        self.position_first_visits = set()  # 首次访问的位置
        
        # 游戏进程多样性
        self.phase_counts = {"opening": 0, "midgame": 0, "endgame": 0}
        self.stone_count_hist = np.zeros(65)  # 0-64个棋子

    def analyze_shard(self, npz_path):
        """分析单个分片文件"""
        print(f"分析: {npz_path.name}")
        
        try:
            data = np.load(npz_path, mmap_mode='r')
        except Exception as e:
            print(f"  错误: {e}")
            return {}
        
        if not all(field in data.files for field in ['s', 'a', 'd']):
            print("  缺少必要字段")
            return {}
        
        N = len(data['a'])
        print(f"  样本数: {N}")
        
        # 按游戏分析（基于终局标志）
        game_boundaries = self._find_game_boundaries(data['d'])
        self.total_games += len(game_boundaries)
        
        for i in range(N):
            self._analyze_sample(data, i)
            self.current_game_length += 1
            
            # 游戏结束
            if data['d'][i] == 1:
                self.game_lengths.append(self.current_game_length)
                self.current_game_length = 0
        
        self.total_samples += N
        return self._generate_diversity_report(npz_path.name)

    def _find_game_boundaries(self, done_flags):
        """找到游戏边界"""
        boundaries = []
        for i, done in enumerate(done_flags):
            if done == 1:
                boundaries.append(i)
        return boundaries

    def _analyze_sample(self, data, idx):
        """分析单个样本的多样性"""
        try:
            s = data['s'][idx]
            a = data['a'][idx]
            s_next = data['s_next'][idx] if 's_next' in data.files else None
            
            self._analyze_state_diversity(s, a, s_next)
            self._analyze_position_diversity(a)
            self._analyze_game_process_diversity(s)
            
        except Exception as e:
            # 静默处理错误，避免过多输出
            pass

    def _analyze_state_diversity(self, state, action, next_state):
        """分析状态多样性"""
        # 状态哈希
        state_hash = self._hash_array(state)
        
        # 状态唯一性
        self.unique_states.add(state_hash)
        self.recent_states.append(state_hash)
        
        # 状态转移唯一性
        if next_state is not None:
            next_hash = self._hash_array(next_state)
            transition_hash = f"{state_hash}-{action}-{next_hash}"
            self.unique_transitions.add(transition_hash)

    def _analyze_position_diversity(self, action):
        """分析位置多样性"""
        if action < 64:  # 非PASS动作
            self.position_visits[action] += 1
            
            # 记录首次访问
            if action not in self.position_first_visits:
                self.position_first_visits.add(action)

    def _analyze_game_process_diversity(self, state):
        """分析游戏进程多样性"""
        # 棋子数量
        stone_count = np.sum(state[0:2])
        if 0 <= stone_count <= 64:  # 确保在有效范围内
            self.stone_count_hist[int(stone_count)] += 1
        
        # 游戏阶段
        if stone_count < 20:
            self.phase_counts["opening"] += 1
        elif stone_count < 50:
            self.phase_counts["midgame"] += 1
        else:
            self.phase_counts["endgame"] += 1

    def _hash_array(self, arr):
        """计算数组哈希"""
        return hashlib.md5(arr.tobytes()).hexdigest()

    def _calculate_entropy(self, distribution):
        """计算分布熵"""
        probs = distribution / (np.sum(distribution) + 1e-8)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    def _generate_diversity_report(self, shard_name):
        """生成多样性报告"""
        # 状态多样性指标
        state_novelty = len(self.unique_states) / max(1, self.total_samples)
        transition_novelty = len(self.unique_transitions) / max(1, self.total_samples)
        recent_diversity = len(set(self.recent_states)) / len(self.recent_states) if self.recent_states else 0
        
        # 位置多样性指标
        position_coverage = len(self.position_first_visits) / 64
        position_entropy = self._calculate_entropy(self.position_visits)
        rare_positions = np.sum(self.position_visits == 1) / 64  # 只被访问一次的位置
        
        # 游戏进程多样性指标
        total_phases = sum(self.phase_counts.values())
        phase_balance = 1 - (abs(self.phase_counts["opening"]/total_phases - 0.3) + 
                           abs(self.phase_counts["endgame"]/total_phases - 0.3)) if total_phases > 0 else 0
        
        stone_entropy = self._calculate_entropy(self.stone_count_hist)
        
        # 游戏长度多样性
        game_length_entropy = self._calculate_entropy(np.bincount(self.game_lengths)) if self.game_lengths else 0
        
        # 综合多样性分数
        diversity_score = self._calculate_diversity_score(
            state_novelty, transition_novelty, recent_diversity,
            position_coverage, position_entropy, rare_positions,
            phase_balance, stone_entropy, game_length_entropy
        )
        
        return {
            "shard": shard_name,
            "samples": self.total_samples,
            "games": self.total_games,
            "diversity_score": diversity_score,
            "diversity_level": self._get_diversity_level(diversity_score),
            
            # 状态多样性
            "state_novelty": state_novelty,
            "transition_novelty": transition_novelty,
            "recent_diversity": recent_diversity,
            "unique_states": len(self.unique_states),
            "unique_transitions": len(self.unique_transitions),
            
            # 位置多样性
            "position_coverage": position_coverage,
            "position_entropy": position_entropy,
            "rare_positions": rare_positions,
            
            # 游戏进程多样性
            "phase_balance": phase_balance,
            "stone_entropy": stone_entropy,
            "game_length_entropy": game_length_entropy,
            "avg_game_length": np.mean(self.game_lengths) if self.game_lengths else 0
        }

    def _calculate_diversity_score(self, state_novelty, transition_novelty, recent_diversity,
                                 position_coverage, position_entropy, rare_positions,
                                 phase_balance, stone_entropy, game_length_entropy):
        """计算综合多样性分数 (0-100)"""
        
        # 状态多样性 (40分)
        state_diversity = (
            state_novelty * 15 +
            transition_novelty * 15 +
            recent_diversity * 10
        )
        
        # 位置多样性 (30分)
        position_diversity = (
            position_coverage * 10 +
            min(1.0, position_entropy / 4.0) * 10 +  # 熵最大约4.0
            rare_positions * 10  # 鼓励探索冷门位置
        )
        
        # 游戏进程多样性 (30分)
        process_diversity = (
            phase_balance * 10 +
            min(1.0, stone_entropy / 4.0) * 10 +
            min(1.0, game_length_entropy / 3.0) * 10  # 游戏长度多样性
        )
        
        return min(100, state_diversity + position_diversity + process_diversity)

    def _get_diversity_level(self, score):
        """获取多样性等级"""
        if score >= 80: return "非常多样"
        elif score >= 65: return "比较多样"
        elif score >= 50: return "一般多样"
        else: return "缺乏多样"


def save_diversity_csv(reports, data_dir):
    """保存多样性报告到CSV"""
    csv_path = data_dir / "sample_diversity_report.csv"
    
    fieldnames = [
        "shard", "samples", "games", "diversity_score", "diversity_level",
        "state_novelty", "transition_novelty", "recent_diversity",
        "position_coverage", "position_entropy", "rare_positions",
        "phase_balance", "stone_entropy", "game_length_entropy", "avg_game_length"
    ]
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for report in reports:
            row = {field: report.get(field, '') for field in fieldnames}
            writer.writerow(row)
    
    print(f"多样性报告已保存: {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="样本多样性测评工具")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--limit-shards", type=int, default=0)
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"错误: 目录不存在 {data_dir}")
        return
    
    shard_files = sorted(data_dir.glob("shard_*.npz"))
    if not shard_files:
        print(f"错误: 在 {data_dir} 中未找到shard_*.npz文件")
        return
    
    if args.limit_shards > 0:
        shard_files = shard_files[:args.limit_shards]
    
    print(f"找到 {len(shard_files)} 个分片文件")
    print("开始多样性分析...")
    start_time = time.time()
    
    analyzer = DiversityAnalyzer()
    reports = []
    
    for i, shard_file in enumerate(shard_files, 1):
        print(f"[{i}/{len(shard_files)}] ", end="")
        report = analyzer.analyze_shard(shard_file)
        if report:
            reports.append(report)
            print(f"多样性: {report['diversity_score']:.1f}分 ({report['diversity_level']})")
    
    if reports:
        csv_path = save_diversity_csv(reports, data_dir)
        
        elapsed_time = time.time() - start_time
        print(f"\n分析完成! 耗时: {elapsed_time:.1f}秒")
        print(f"总样本: {analyzer.total_samples}")
        print(f"总对局: {analyzer.total_games}")
        print(f"报告保存至: {csv_path}")
        
        # 显示关键指标
        avg_diversity = np.mean([r["diversity_score"] for r in reports])
        best_shard = max(reports, key=lambda x: x["diversity_score"])
        
        print(f"\n关键多样性指标:")
        print(f"  平均多样性: {avg_diversity:.1f}分")
        print(f"  最佳分片: {best_shard['shard']} ({best_shard['diversity_score']:.1f}分)")
        print(f"  状态新颖性: {best_shard['state_novelty']:.3f}")
        print(f"  位置覆盖率: {best_shard['position_coverage']:.3f}")
        print(f"  阶段平衡性: {best_shard['phase_balance']:.3f}")
    else:
        print("没有生成有效报告")

if __name__ == "__main__":
    main()