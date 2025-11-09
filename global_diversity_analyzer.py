# global_diversity_analyzer.py
"""
全局样本多样性测评工具 - 跨分片检测重复，计算真正的全局多样性
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib
import collections
import time
import csv

class GlobalDiversityAnalyzer:
    """全局多样性分析器 - 跨分片检测重复"""
    
    def __init__(self, window_size=5000):
        self.window_size = window_size
        self.reset_global_stats()
        
    def reset_global_stats(self):
        """重置全局统计信息"""
        # 全局状态跟踪（跨所有分片）
        self.global_unique_states = set()
        self.global_unique_transitions = set()
        self.global_position_visits = np.zeros(64)
        self.global_position_first_visits = set()
        self.global_phase_counts = {"opening": 0, "midgame": 0, "endgame": 0}
        self.global_stone_count_hist = np.zeros(65)
        self.global_game_lengths = []
        
        # 分片级统计
        self.total_samples = 0
        self.total_games = 0
        self.current_game_length = 0
        self.recent_states = collections.deque(maxlen=self.window_size)
        
        # 跨分片重复检测
        self.cross_shard_duplicates = 0
        self.processed_shards = 0

    def analyze_all_shards(self, shard_files):
        """分析所有分片的全局多样性"""
        print(f"开始全局多样性分析...")
        print(f"将分析 {len(shard_files)} 个分片")
        
        shard_reports = []
        
        for i, shard_file in enumerate(shard_files, 1):
            print(f"[{i}/{len(shard_files)}] 分析: {shard_file.name}")
            report = self._analyze_single_shard(shard_file)
            if report:
                shard_reports.append(report)
                self.processed_shards += 1
                
                # 显示该分片对全局多样性的贡献
                new_states = report['new_states_in_shard']
                total_states = report['samples_in_shard']
                novelty_rate = new_states / total_states if total_states > 0 else 0
                print(f"  新状态: {new_states}/{total_states} ({novelty_rate:.1%})")
        
        # 生成全局报告
        global_report = self._generate_global_report(shard_reports)
        return global_report, shard_reports

    def _analyze_single_shard(self, npz_path):
        """分析单个分片，并更新全局统计"""
        try:
            data = np.load(npz_path, mmap_mode='r')
        except Exception as e:
            print(f"  错误: {e}")
            return None
        
        if not all(field in data.files for field in ['s', 'a', 'd']):
            return None
        
        N = len(data['a'])
        new_states_in_shard = 0
        new_transitions_in_shard = 0
        
        # 临时统计本分片
        shard_samples = N
        
        for i in range(N):
            state_novel, transition_novel = self._analyze_global_sample(data, i)
            if state_novel:
                new_states_in_shard += 1
            if transition_novel:
                new_transitions_in_shard += 1
                
            self.current_game_length += 1
            
            # 游戏结束
            if data['d'][i] == 1:
                self.global_game_lengths.append(self.current_game_length)
                self.current_game_length = 0
        
        self.total_samples += N
        self.total_games += len([d for d in data['d'] if d == 1])
        
        return {
            "shard": npz_path.name,
            "samples_in_shard": shard_samples,
            "new_states_in_shard": new_states_in_shard,
            "new_transitions_in_shard": new_transitions_in_shard,
            "novelty_rate": new_states_in_shard / shard_samples if shard_samples > 0 else 0
        }

    def _analyze_global_sample(self, data, idx):
        """分析单个样本，检测跨分片重复"""
        state_novel = False
        transition_novel = False
        
        try:
            s = data['s'][idx]
            a = data['a'][idx]
            s_next = data['s_next'][idx] if 's_next' in data.files else None
            
            # 状态哈希
            state_hash = self._hash_array(s)
            
            # 检测状态是否新颖（跨分片）
            if state_hash not in self.global_unique_states:
                state_novel = True
                self.global_unique_states.add(state_hash)
            
            self.recent_states.append(state_hash)
            
            # 位置多样性（全局）
            if a < 64:
                self.global_position_visits[a] += 1
                if a not in self.global_position_first_visits:
                    self.global_position_first_visits.add(a)
            
            # 状态转移多样性（全局）
            if s_next is not None:
                next_hash = self._hash_array(s_next)
                transition_hash = f"{state_hash}-{a}-{next_hash}"
                if transition_hash not in self.global_unique_transitions:
                    transition_novel = True
                    self.global_unique_transitions.add(transition_hash)
            
            # 游戏进程多样性（全局）
            stone_count = np.sum(s[0:2])
            if 0 <= stone_count <= 64:
                self.global_stone_count_hist[int(stone_count)] += 1
            
            if stone_count < 20:
                self.global_phase_counts["opening"] += 1
            elif stone_count < 50:
                self.global_phase_counts["midgame"] += 1
            else:
                self.global_phase_counts["endgame"] += 1
                
        except Exception:
            pass
        
        return state_novel, transition_novel

    def _hash_array(self, arr):
        """计算数组哈希"""
        return hashlib.md5(arr.tobytes()).hexdigest()

    def _calculate_entropy(self, distribution):
        """计算分布熵"""
        probs = distribution / (np.sum(distribution) + 1e-8)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    def _generate_global_report(self, shard_reports):
        """生成全局多样性报告"""
        # 全局状态多样性
        global_state_novelty = len(self.global_unique_states) / max(1, self.total_samples)
        global_transition_novelty = len(self.global_unique_transitions) / max(1, self.total_samples)
        global_recent_diversity = len(set(self.recent_states)) / len(self.recent_states) if self.recent_states else 0
        
        # 全局位置多样性
        global_position_coverage = len(self.global_position_first_visits) / 64
        global_position_entropy = self._calculate_entropy(self.global_position_visits)
        global_rare_positions = np.sum(self.global_position_visits == 1) / 64
        
        # 全局游戏进程多样性
        total_global_phases = sum(self.global_phase_counts.values())
        global_phase_balance = 1 - (abs(self.global_phase_counts["opening"]/total_global_phases - 0.3) + 
                                  abs(self.global_phase_counts["endgame"]/total_global_phases - 0.3)) if total_global_phases > 0 else 0
        
        global_stone_entropy = self._calculate_entropy(self.global_stone_count_hist)
        global_game_length_entropy = self._calculate_entropy(np.bincount(self.global_game_lengths)) if self.global_game_lengths else 0
        
        # 跨分片重复统计
        total_samples_all_shards = sum(r["samples_in_shard"] for r in shard_reports)
        cross_shard_duplicate_rate = 1 - (len(self.global_unique_states) / total_samples_all_shards) if total_samples_all_shards > 0 else 0
        
        # 综合全局多样性分数
        global_diversity_score = self._calculate_diversity_score(
            global_state_novelty, global_transition_novelty, global_recent_diversity,
            global_position_coverage, global_position_entropy, global_rare_positions,
            global_phase_balance, global_stone_entropy, global_game_length_entropy
        )
        
        return {
            "global_summary": {
                "total_shards": len(shard_reports),
                "total_samples": self.total_samples,
                "total_games": self.total_games,
                "global_diversity_score": global_diversity_score,
                "global_diversity_level": self._get_diversity_level(global_diversity_score),
                
                # 跨分片重复统计
                "unique_states_global": len(self.global_unique_states),
                "unique_transitions_global": len(self.global_unique_transitions),
                "cross_shard_duplicate_rate": cross_shard_duplicate_rate,
                
                # 全局多样性指标
                "global_state_novelty": global_state_novelty,
                "global_transition_novelty": global_transition_novelty,
                "global_recent_diversity": global_recent_diversity,
                "global_position_coverage": global_position_coverage,
                "global_position_entropy": global_position_entropy,
                "global_rare_positions": global_rare_positions,
                "global_phase_balance": global_phase_balance,
                "global_stone_entropy": global_stone_entropy,
                "global_game_length_entropy": global_game_length_entropy,
                "avg_game_length": np.mean(self.global_game_lengths) if self.global_game_lengths else 0
            },
            "shard_details": shard_reports
        }

    def _calculate_diversity_score(self, state_novelty, transition_novelty, recent_diversity,
                                 position_coverage, position_entropy, rare_positions,
                                 phase_balance, stone_entropy, game_length_entropy):
        """计算综合多样性分数"""
        state_diversity = state_novelty * 15 + transition_novelty * 15 + recent_diversity * 10
        position_diversity = position_coverage * 10 + min(1.0, position_entropy/4.0) * 10 + rare_positions * 10
        process_diversity = phase_balance * 10 + min(1.0, stone_entropy/4.0) * 10 + min(1.0, game_length_entropy/3.0) * 10
        
        return min(100, state_diversity + position_diversity + process_diversity)

    def _get_diversity_level(self, score):
        if score >= 80: return "非常多样"
        elif score >= 65: return "比较多样"
        elif score >= 50: return "一般多样"
        else: return "缺乏多样"


def save_global_diversity_report(global_report, data_dir):
    """保存全局多样性报告"""
    # 保存全局汇总
    summary_path = data_dir / "global_diversity_summary.csv"
    
    summary_data = {
        "统计项目": [
            "分析分片数", "总样本数", "总对局数", "全局多样性分数", "多样性等级",
            "全局唯一状态数", "全局唯一转移数", "跨分片重复率",
            "全局状态新颖性", "全局位置覆盖率", "全局阶段平衡性"
        ],
        "数值": [
            global_report["global_summary"]["total_shards"],
            global_report["global_summary"]["total_samples"],
            global_report["global_summary"]["total_games"],
            f"{global_report['global_summary']['global_diversity_score']:.1f}",
            global_report["global_summary"]["global_diversity_level"],
            global_report["global_summary"]["unique_states_global"],
            global_report["global_summary"]["unique_transitions_global"],
            f"{global_report['global_summary']['cross_shard_duplicate_rate']:.3f}",
            f"{global_report['global_summary']['global_state_novelty']:.3f}",
            f"{global_report['global_summary']['global_position_coverage']:.3f}",
            f"{global_report['global_summary']['global_phase_balance']:.3f}"
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(summary_path, index=False, encoding='utf-8')
    
    # 保存分片详情
    details_path = data_dir / "shard_novelty_details.csv"
    if global_report["shard_details"]:
        df_details = pd.DataFrame(global_report["shard_details"])
        df_details.to_csv(details_path, index=False, encoding='utf-8')
    
    print(f"全局汇总报告: {summary_path}")
    print(f"分片详情报告: {details_path}")
    return summary_path, details_path


def main():
    parser = argparse.ArgumentParser(description="全局样本多样性测评工具")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--limit-shards", type=int, default=0)
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    shard_files = sorted(data_dir.glob("shard_*.npz"))
    
    if args.limit_shards > 0:
        shard_files = shard_files[:args.limit_shards]
    
    print(f"找到 {len(shard_files)} 个分片文件")
    
    analyzer = GlobalDiversityAnalyzer()
    start_time = time.time()
    
    global_report, shard_reports = analyzer.analyze_all_shards(shard_files)
    
    if global_report:
        summary_path, details_path = save_global_diversity_report(global_report, data_dir)
        
        elapsed_time = time.time() - start_time
        summary = global_report["global_summary"]
        
        print(f"\n=== 全局多样性分析完成 ===")
        print(f"耗时: {elapsed_time:.1f}秒")
        print(f"总样本: {summary['total_samples']}")
        print(f"总对局: {summary['total_games']}")
        print(f"全局多样性: {summary['global_diversity_score']:.1f}分 ({summary['global_diversity_level']})")
        print(f"跨分片重复率: {summary['cross_shard_duplicate_rate']:.1%}")
        print(f"全局状态新颖性: {summary['global_state_novelty']:.3f}")
        print(f"全局位置覆盖率: {summary['global_position_coverage']:.3f}")
        print(f"报告保存至: {summary_path}")

if __name__ == "__main__":
    main()