# print_raw_data.py
import numpy as np
import argparse
from pathlib import Path

def format_board_channel(channel):
    """格式化棋盘通道显示"""
    lines = []
    for i in range(8):
        row = []
        for j in range(8):
            row.append(str(int(channel[i, j])))
        lines.append(f"         [{','.join(row)}],  # 第{i}行")
    return '\n'.join(lines)

def format_legal_mask(legal_mask):
    """格式化合法动作掩码显示"""
    lines = []
    for i in range(8):
        row_mask = legal_mask[i*8:(i+1)*8]
        row_str = ','.join(str(int(x)) for x in row_mask)
        legal_positions = []
        for j, val in enumerate(row_mask):
            if val == 1:
                legal_positions.append(f"({i},{j})")
        pos_info = f"  # 第{i}行: {', '.join(legal_positions)}" if legal_positions else ""
        lines.append(f"                      {row_str},{pos_info}")
    return '\n'.join(lines)

def format_vals_with_legal(vals, legal_mask):
    """格式化价值向量显示"""
    legal_indices = np.where(legal_mask == 1)[0]
    if len(legal_indices) == 0:
        return "    # 没有合法动作"
    
    lines = []
    lines.append("    # 只有合法位置有非零值，比如：")
    for idx in legal_indices:
        row, col = idx // 8, idx % 8
        val = vals[idx]
        lines.append(f"    # 索引{idx}({row},{col}): {val:.3f}")
    return '\n'.join(lines)

def print_raw_shard_data(npz_path: str, sample_count: int = 5, start_index: int = 0):
    """直接打印数据采集的原始数据（按照rec格式）"""
    data = np.load(npz_path)
    
    print(f"\n{'='*80}")
    print(f"分片文件: {npz_path}")
    print(f"样本数量: {len(data['a'])}")
    print(f"显示样本: {start_index} 到 {start_index + sample_count - 1}")
    print(f"{'='*80}")
    
    end_index = min(start_index + sample_count, len(data['a']))
    
    for i in range(start_index, end_index):
        print(f"\n📊 样本 {i}:")
        print(f"rec = {{")
        
        # s (状态)
        print(f"    # 当前状态 (3,8,8) {data['s'][i].dtype}")
        print(f"    \"s\": array([")
        print(f"        # 通道0: 当前玩家(X)的棋子位置")
        print(format_board_channel(data['s'][i][0]))
        print(f"        ],")
        print(f"        ")
        print(f"        # 通道1: 对手(O)的棋子位置")
        print(format_board_channel(data['s'][i][1]))
        print(f"        ],")
        print(f"        ")
        print(f"        # 通道2: 全1通道（表示当前玩家视角）")
        print(f"        [[1,1,1,1,1,1,1,1],")
        print(f"         [1,1,1,1,1,1,1,1],")
        print(f"         ...]  # 全部为1")
        print(f"    ], dtype={data['s'][i].dtype}),")
        print(f"    ")
        
        # a (动作)
        action = data['a'][i]
        if action == 64:
            action_desc = "PASS"
        else:
            row, col = action // 8, action % 8
            action_desc = f"({row},{col})"
        print(f"    # 选择的动作索引")
        print(f"    \"a\": {action},  # 对应坐标{action_desc}")
        print(f"    ")
        
        # s_next (下一状态)
        print(f"    # 下一状态 (执行动作后的状态)")
        print(f"    \"s_next\": array([...], dtype={data['s_next'][i].dtype}),  # 形状同样为(3,8,8)")
        print(f"    ")
        
        # r (奖励)
        reward = data['r'][i]
        terminal = data['d'][i]
        reward_desc = "非终局步骤奖励为0" if reward == 0.0 and not terminal else "终局奖励"
        print(f"    # 即时奖励")
        print(f"    \"r\": {reward},  # {reward_desc}")
        print(f"    ")
        
        # d (是否终局)
        terminal_desc = "终局" if terminal else "非终局"
        print(f"    # 是否终局")
        print(f"    \"d\": {int(terminal)},  # {terminal_desc}")
        print(f"    ")
        
        # legal_s (合法动作掩码)
        if 'legal_s' in data.files:
            legal_s = data['legal_s'][i]
            legal_count = np.sum(legal_s)
            print(f"    # 当前状态的合法动作掩码")
            print(f"    \"legal_s\": array([", end="")
            print(f"  # 共{legal_count}个合法动作")
            print(format_legal_mask(legal_s))
            print(f"                      ], dtype={legal_s.dtype}),")
            print(f"    ")
        
        # legal_s_next (下一状态合法动作掩码)
        if 'legal_s_next' in data.files:
            legal_s_next = data['legal_s_next'][i]
            legal_next_count = np.sum(legal_s_next)
            print(f"    # 下一状态的合法动作掩码")
            print(f"    \"legal_s_next\": array([...], dtype={legal_s_next.dtype}),  # 形状(64,), {legal_next_count}个合法动作")
            print(f"    ")
        
        # vals (动作价值)
        if 'vals' in data.files:
            vals = data['vals'][i]
            print(f"    # 所有合法动作的价值估计")
            print(f"    \"vals\": array([...], dtype={vals.dtype})")
            if 'legal_s' in data.files:
                print(format_vals_with_legal(vals, data['legal_s'][i]))
        
        print(f"}}")
        print(f"")

def print_data_structure(npz_path: str):
    """打印数据结构信息"""
    data = np.load(npz_path)
    
    print(f"\n📁 数据结构: {npz_path}")
    print(f"{'='*50}")
    
    for key in data.files:
        arr = data[key]
        print(f"{key:15} : {arr.shape} {arr.dtype}")
        
        # 显示一些统计信息
        if arr.ndim > 0 and len(arr) > 0:
            if key in ['r', 'vals']:
                print(f"{'':15}   范围: [{np.min(arr):.3f}, {np.max(arr):.3f}] 均值: {np.mean(arr):.3f}")
            elif key in ['a', 'd', 'legal_s', 'legal_s_next']:
                unique_vals = np.unique(arr)
                if len(unique_vals) <= 10:  # 只显示较少的唯一值
                    print(f"{'':15}   唯一值: {unique_vals}")
                else:
                    print(f"{'':15}   唯一值数量: {len(unique_vals)}")

def main():
    parser = argparse.ArgumentParser(description="直接打印数据采集的原始数据")
    parser.add_argument("shard_path", help="分片文件路径 (.npz)")
    parser.add_argument("--samples", type=int, default=1, help="打印的样本数量")
    parser.add_argument("--start", type=int, default=0, help="起始样本索引")
    parser.add_argument("--structure", action="store_true", help="只显示数据结构")
    
    args = parser.parse_args()
    
    if not Path(args.shard_path).exists():
        print(f"错误: 文件不存在 {args.shard_path}")
        return
    
    if args.structure:
        print_data_structure(args.shard_path)
    else:
        print_raw_shard_data(args.shard_path, args.samples, args.start)

if __name__ == "__main__":
    main()