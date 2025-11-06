# print_raw_data.py
import numpy as np
import argparse
from pathlib import Path

def format_board_channel(channel):
    """格式化单个通道显示"""
    lines = []
    for i in range(8):
        row = []
        for j in range(8):
            row.append(str(int(channel[i, j])))
        lines.append(f"         [{','.join(row)}],  # 第{i}行")
    return '\n'.join(lines)

def print_raw_shard_data(npz_path: str, sample_count: int = 5, start_index: int = 0):
    """直接打印数据采集的原始数据"""
    data = np.load(npz_path)
    
    print(f"分片文件: {npz_path}")
    print(f"样本数量: {len(data['a'])}")
    print(f"显示样本: {start_index} 到 {start_index + sample_count - 1}")
    print()
    
    end_index = min(start_index + sample_count, len(data['a']))
    
    for i in range(start_index, end_index):
        print(f"样本 {i}:")
        print("-" * 40)
        
        # 显示当前状态
        s_state = data['s'][i]
        print(f"当前状态 (3,8,8) {s_state.dtype}:")
        print(f"通道0: 当前玩家的棋子位置")
        print(format_board_channel(s_state[0]))
        print(f"通道1: 对手的棋子位置")  
        print(format_board_channel(s_state[1]))
        print(f"通道2: 全1通道")
        print(format_board_channel(s_state[2]))
        print()
        
        # 显示下一状态 - 完整显示
        s_next = data['s_next'][i]
        print(f"下一状态 (3,8,8) {s_next.dtype}:")
        print(f"通道0: 下一状态当前玩家的棋子位置")
        print(format_board_channel(s_next[0]))
        print(f"通道1: 下一状态对手的棋子位置")
        print(format_board_channel(s_next[1]))
        print(f"通道2: 全1通道")
        print(format_board_channel(s_next[2]))
        print()
        
        # 显示动作
        action = data['a'][i]
        if action == 64:
            action_desc = "PASS"
        else:
            row, col = action // 8, action % 8
            action_desc = f"({row},{col})"
        print(f"动作: {action} {action_desc}")
        
        # 显示奖励和终局状态
        reward = data['r'][i]
        terminal = data['d'][i]
        print(f"奖励: {reward}, 终局: {bool(terminal)}")
        
        # 显示合法动作
        if 'legal_s' in data.files:
            legal_s = data['legal_s'][i]
            legal_count = np.sum(legal_s)
            legal_indices = np.where(legal_s == 1)[0]
            legal_positions = [f"({idx//8},{idx%8})" for idx in legal_indices]
            print(f"合法动作: {legal_count}个 {legal_positions}")
        
        # 显示下一状态合法动作
        if 'legal_s_next' in data.files:
            legal_s_next = data['legal_s_next'][i]
            legal_next_count = np.sum(legal_s_next)
            legal_next_indices = np.where(legal_s_next == 1)[0]
            legal_next_positions = [f"({idx//8},{idx%8})" for idx in legal_next_indices]
            print(f"下一状态合法动作: {legal_next_count}个 {legal_next_positions}")
        
        # 显示价值
        if 'vals' in data.files and 'legal_s' in data.files:
            vals = data['vals'][i]
            legal_mask = data['legal_s'][i] == 1
            legal_vals = vals[legal_mask]
            if len(legal_vals) > 0:
                chosen_val = vals[action] if action != 64 else 0
                print(f"选择动作价值: {chosen_val:.3f}")
                print(f"合法动作价值范围: [{np.min(legal_vals):.3f}, {np.max(legal_vals):.3f}]")
        
        print("=" * 60)
        print()

def main():
    parser = argparse.ArgumentParser(description="打印数据采集的原始数据")
    parser.add_argument("shard_path", help="分片文件路径 (.npz)")
    parser.add_argument("--samples", type=int, default=5, help="打印的样本数量")
    parser.add_argument("--start", type=int, default=0, help="起始样本索引")
    
    args = parser.parse_args()
    
    if not Path(args.shard_path).exists():
        print(f"错误: 文件不存在 {args.shard_path}")
        return
    
    print_raw_shard_data(args.shard_path, args.samples, args.start)

if __name__ == "__main__":
    main()