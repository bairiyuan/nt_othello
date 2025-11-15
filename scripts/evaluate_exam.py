# -*- coding: utf-8 -*-
"""
evaluate_exam.py

通用评测脚本：在 exam.json 的 Othello 残局上评测不同模型，并与 Edax 标准答案对比。

支持：
- MCTS 模型（使用 MCTS.py + 简单启发式）
- Net 模型（使用 phase3/model_wrap.py + 训练好的 ckpt，例如 ckpts_phaseC/step_34000.pt）
- Net+MCTS 模型（使用训练好的 ckpt 作为 MCTS 的 policy+value）
- Ntuple 模型（使用 value.ModelScorer 训练得到的权重）

用法示例（在项目根目录 ./othello 下）:

    # 评测 MCTS（200 次模拟）
    python scripts/evaluate_exam.py --model mcts --mcts-sims 200

    # 评测网络模型（step_34000.pt）
    python scripts/evaluate_exam.py --model net --ckpt ckpts_phaseC/step_34000.pt

    # 评测 Net+MCTS（200 次模拟）
    python scripts/evaluate_exam.py --model net_mcts --ckpt ckpts_phaseC/step_34000.pt --mcts-sims 200

    # 评测 Ntuple 模型（ntuple 权重在 model/model.cpt.npy）
    python scripts/evaluate_exam.py --model ntuple --ntuple-path model/model.cpt.npy

评测结果会保存到 ./exam 目录下，文件名类似：
    ./exam/MCTS_sims200_results.json
    ./exam/Net_step_34000_results.json
    ./exam/NetMCTS_step_34000_sims200_results.json
    ./exam/Ntuple_model_cpt_results.json
"""

from __future__ import print_function

import os
import sys
import json
import argparse
from types import SimpleNamespace

import numpy as np
import torch

# -------------------------------
# 路径设置：确保能导入 othello / MCTS / model_wrap / value
# -------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))      # .../othello/scripts
ROOT_DIR = os.path.dirname(SCRIPT_DIR)                       # .../othello
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

PHASE3_DIR = os.path.join(ROOT_DIR, "phase3")                # .../othello/phase3
if PHASE3_DIR not in sys.path:
    sys.path.insert(0, PHASE3_DIR)

from othello import Board            # 棋盘实现
# from MCTS import MCTS                # MCTS 实现
# from model_wrap import load_model    # 网络加载函数（返回 OthelloNet 实例）
from value import ModelScorer, ScorerWrapper  # Ntuple 模型和角色包装器

# -------------------------------
# 一些基础工具：坐标 / 落子序列还原
# -------------------------------

def move_str_to_coord(move):
    """
    将 'e6' 这样的落子字符串转成 (i, j) = (row, col)，0-based 下标。
    """
    move = move.strip().lower()
    col_char = move[0]         # 'a'..'h'
    row_char = move[1]         # '1'..'8'

    j = ord(col_char) - ord('a')
    i = int(row_char) - 1
    return i, j


def coord_to_move_str(i, j):
    """
    将 (i, j) 0-based 下标转回 'e6' 形式。
    """
    col_char = chr(ord('a') + j)
    row_char = str(i + 1)
    return "%s%s" % (col_char, row_char)


def side_str_to_player(side):
    """
    exam.json 里的 "side_to_move": "black"/"white" -> Board.BLACK / Board.WHITE
    """
    side = side.lower()
    if side == "black":
        return Board.BLACK
    elif side == "white":
        return Board.WHITE
    else:
        raise ValueError("Unknown side_to_move: %s" % side)


def reconstruct_board_from_history(history, side_to_move_str):
    """
    根据 exam.json 里给出的 history（形如 ["e6","f4",...]）还原棋盘和轮到哪一方走。
    假设对局从标准初始盘面开始，Black 先手，history 中交替落子（未显式包含 pass）。
    """
    board = Board()
    current_player = Board.BLACK

    for mv in history:
        i, j = move_str_to_coord(mv)
        board.flip(i, j, current_player)
        current_player = Board.opponent(current_player)

    declared_side = side_str_to_player(side_to_move_str)
    # 通常 inferred_side == declared_side，这里直接相信 exam.json 声明
    return board, declared_side


# ---------------------------------
# Game 适配器：MCTS 使用的 Game 接口
# ---------------------------------

class OthelloGameAdapter(object):
    """
    将 othello.Board 适配成 MCTS.py 所需要的 Game 接口。

    这里把“状态”定义为一个二元组:
        state = (board, current_player)

    对 MCTS 而言：
    - 动作空间大小：64 个格子 + 1 个 pass 动作
    """

    def __init__(self, board_size=8):
        self.n = board_size

    def getActionSize(self):
        """
        动作空间大小：64 个落子点 + 1 个 pass。
        动作编号约定：
            0 ~ 63 -> 棋盘上的格子 i*8 + j
            64     -> pass
        """
        return self.n * self.n + 1

    def getValidMoves(self, state, player):
        """
        返回一个长度为 ActionSize 的 0/1 向量，表示在该状态下哪些动作合法。
        """
        board, current_player = state
        valids = np.zeros(self.getActionSize(), dtype=np.int32)

        # 终局 -> 只有 pass 合法
        if board.is_terminal_state():
            valids[-1] = 1
            return valids

        moves = board.feasible_pos(current_player)
        if not moves:
            # 没子可下 -> 只能 pass
            valids[-1] = 1
        else:
            for (i, j) in moves:
                idx = i * self.n + j
                valids[idx] = 1

        return valids

    def getGameEnded(self, state, player):
        """
        若未终局 -> 返回 0
        若终局 -> 返回从“当前执子方”的视角的结果:
            胜利  -> +1
            失败  -> -1
            平局  -> 1e-4
        """
        board, current_player = state

        if not board.is_terminal_state():
            return 0

        my_score = board.score(current_player)
        opp_score = board.score(Board.opponent(current_player))

        if my_score > opp_score:
            return 1
        elif my_score < opp_score:
            return -1
        else:
            return 1e-4

    def getNextState(self, state, player, action):
        """
        在当前 state 下执行动作 action，返回 (next_state, next_player)。
        其中 next_state 仍然是 (board, current_player)。
        """
        board, current_player = state

        # 拷贝棋盘
        new_board = Board()
        new_board.set_board(board.board)

        pass_action = self.getActionSize() - 1

        if action != pass_action:
            i = action // self.n
            j = action % self.n
            new_board.flip(i, j, current_player)

        next_player = Board.opponent(current_player)
        next_state = (new_board, next_player)

        return next_state, 1  # next_player 给 getCanonicalForm 用，这里实际忽略

    def getCanonicalForm(self, state, player):
        """
        这里直接把 state 当作 canonical form，忽略 player。
        """
        return state

    def stringRepresentation(self, state):
        """
        用于给 MCTS 做字典 key。
        把棋盘数组和当前执子方编码成一个字符串即可。
        """
        board, current_player = state
        return board.board.tobytes().hex() + "|%d" % current_player


# ------------------------------------------------
# MCTS 用的简单“评估网络”：HeuristicNNet
# ------------------------------------------------

class HeuristicNNet(object):
    """
    给 MCTS 用的一个“假神经网络”，实现:
        - predict(state) -> (policy, value)
        - get_mixed_value(state) -> (value, extra)

    value 使用简单的子力差启发；policy 在合法步上均匀分布。
    """

    def __init__(self, game):
        self.game = game

    def predict(self, state):
        """
        返回 (policy, value)
        policy：对所有动作的概率（非法动作为 0）
        value：当前执子方视角下局面的一个评估值，范围大致在 [-1,1]
        """
        valids = self.game.getValidMoves(state, 1)
        policy = valids.astype(np.float32)

        s = float(np.sum(policy))
        if s > 0:
            policy /= s
        else:
            policy[:] = 1.0 / len(policy)

        value, _ = self.get_mixed_value(state)
        return policy, value

    def get_mixed_value(self, state):
        """
        简单子力差启发：
            v = (my_score - opp_score) / max(1, my_score + opp_score)
        """
        board, current_player = state
        my_score = board.score(current_player)
        opp_score = board.score(Board.opponent(current_player))

        denom = max(1, my_score + opp_score)
        value = (my_score - opp_score) / float(denom)
        return value, None


# ------------------------------------------------
# Net+MCTS 使用的 NNet 包装器
# ------------------------------------------------

class NetMCTSNNet(object):
    """
    给 MCTS 使用的“真实神经网络”包装。
    要实现接口：
        - predict(state) -> (policy, value)
        - get_mixed_value(state) -> (value, extra)

    这里复用 Phase2/Phase3 采样时的编码逻辑：
        board_np: int8 in {-1,0,1}; side in {1,-1}
        me  = (board_np == side)
        opp = (board_np == -side)
        emp = (board_np == 0)
    """

    def __init__(self, game: OthelloGameAdapter, model: torch.nn.Module, device: str = "cpu"):
        self.game = game
        self.model = model
        self.device = device
        self.model.eval()

    def _encode_state(self, state):
        """
        state = (board, current_player)
        Board.board: 0=empty, 1=black, 2=white
        转成 [-1,0,1] 再按“当前执子方视角”编码 3 通道。
        """
        board, current_player = state
        arr = board.board  # (8,8), {0,1,2}

        board_np = np.zeros_like(arr, dtype=np.int8)
        board_np[arr == Board.BLACK] = 1   # 黑子 -> 1
        board_np[arr == Board.WHITE] = -1  # 白子 -> -1
        # 空位保持 0

        side = 1 if current_player == Board.BLACK else -1

        me  = (board_np == side).astype(np.float32)
        opp = (board_np == -side).astype(np.float32)
        emp = (board_np == 0).astype(np.float32)

        x_np = np.stack([me, opp, emp], axis=0)  # [3,8,8]
        x = torch.from_numpy(x_np).unsqueeze(0).to(self.device)  # [1,3,8,8]
        return x

    @torch.no_grad()
    def predict(self, state):
        """
        返回 (policy, value)，其中：
        - policy: 长度为 ActionSize (=65) 的概率向量
        - value : 当前执子方视角的估值，范围约在 [-1,1]
        """
        valids = self.game.getValidMoves(state, 1)  # (65,)
        x = self._encode_state(state)
        policy_logits, v, p, v_logit = self.model(x)  # policy_logits: [1,64], v: [1]

        # 网络的 policy 只有 64 维（棋盘格），MCTS 动作空间有 65（含 pass），这里扩展一下
        policy64 = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()  # (64,)
        action_size = self.game.getActionSize()  # 65

        policy = np.zeros(action_size, dtype=np.float32)
        policy[:64] = policy64  # pass 动作的先验为 0，后续会被 valids mask 掉

        # mask 非法着
        policy = policy * valids.astype(np.float32)
        s = float(policy.sum())
        if s > 0:
            policy /= s
        else:
            # 如果全被 mask 了，则退化到 valids 上均匀
            policy = valids.astype(np.float32)
            s = float(policy.sum())
            if s > 0:
                policy /= s
            else:
                policy[:] = 1.0 / action_size

        value = float(v[0].item())  # 已经是 [-1,1] 的值
        return policy, value

    @torch.no_grad()
    def get_mixed_value(self, state):
        """
        MCTS 里用来取“混合 value”的接口，这里直接复用网络的 value。
        返回 (value, extra)，extra 保持 None。
        """
        x = self._encode_state(state)
        _, v, p, v_logit = self.model(x)
        value = float(v[0].item())
        return value, None


# -----------------------
# Agent 抽象基类
# -----------------------

class BaseAgent(object):
    """
    所有模型共享的接口：给定棋盘和轮到谁走，返回 top-k 动作。
    """
    def get_topk_moves(self, board, side_to_move, k=3):
        raise NotImplementedError


# -----------------------
# MCTS Agent（启发式）
# -----------------------

class MCTSAgent(BaseAgent):
    """
    使用 MCTS + HeuristicNNet 的 Agent，即你的 “MCTS 模型”。
    """

    def __init__(self, num_simulations=200, cpuct=1.0):
        self.game = OthelloGameAdapter(board_size=8)
        self.nnet = HeuristicNNet(self.game)
        self.args = SimpleNamespace(
            numMCTSSims=num_simulations,
            cpuct=cpuct,
            use_root_dirichlet=False,  # 评测时不加 Dirichlet 噪声
        )

    def get_topk_moves(self, board, side_to_move, k=3):
        """
        返回一个按概率从大到小排序的动作字符串列表，例如 ["g2", "a4", "f1"]。
        """
        mcts = MCTS(self.game, self.nnet, self.args)

        state = (board, side_to_move)
        canonical = state

        probs = np.array(mcts.getActionProb(canonical, temp=1.0), dtype=np.float32)
        valids = self.game.getValidMoves(canonical, 1)

        pass_action = self.game.getActionSize() - 1
        candidate_actions = [
            a for a in range(self.game.getActionSize())
            if valids[a] > 0 and a != pass_action
        ]

        if not candidate_actions:
            # 没子可下的情况（题目里基本不会出现）
            return ["pass"]

        candidate_actions.sort(key=lambda a: probs[a], reverse=True)
        top_actions = candidate_actions[:k]

        move_strs = [
            coord_to_move_str(a // 8, a % 8)
            for a in top_actions
        ]
        return move_strs


# -----------------------
# Net Agent（纯神经网络）
# -----------------------

class NetAgent(BaseAgent):
    """
    使用已训练好的 OthelloNet 的 Agent（不做搜索）：
    - 输入：图像式编码的局面
    - 输出：policy logits (64 维)，再配合合法着 mask 取 top-k
    """

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.eval()

    def _encode_board(self, board, side_to_move):
        """
        评测时的编码方式要和采样/训练时的 _encode_planes 完全一致：

            board_np: int8 in {-1,0,1}; side in {1,-1}
            me  = (board_np == side)
            opp = (board_np == -side)
            emp = (board_np == 0)

        这里 Board.board 的取值是:
            0 = empty, 1 = black, 2 = white
        我们先把它映射成 [-1,0,1]，再套用同样逻辑。
        """
        arr = board.board  # (8,8), int {0,1,2}

        board_np = np.zeros_like(arr, dtype=np.int8)
        board_np[arr == Board.BLACK] = 1   # 黑子 -> 1
        board_np[arr == Board.WHITE] = -1  # 白子 -> -1
        # 空位保持 0

        side = 1 if side_to_move == Board.BLACK else -1

        me  = (board_np == side).astype(np.float32)
        opp = (board_np == -side).astype(np.float32)
        emp = (board_np == 0).astype(np.float32)

        x = np.stack([me, opp, emp], axis=0)  # [3,8,8]
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        return x

    def _valid_moves_vector(self, board, side_to_move):
        """
        返回长度为 64 的 0/1 向量，表示每个格子是否是合法落子。
        """
        valids = np.zeros(64, dtype=np.int32)
        moves = board.feasible_pos(side_to_move)
        for (i, j) in moves:
            idx = i * 8 + j
            valids[idx] = 1
        return valids

    @torch.no_grad()
    def get_topk_moves(self, board, side_to_move, k=3):
        """
        直接用网络输出的 policy logits，mask 掉非法着，按概率排序取 top-k。
        """
        x = self._encode_board(board, side_to_move)
        policy_logits, v, p, v_logit = self.model(x)  # policy_logits: (1,64)

        policy = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()  # (64,)
        valids = self._valid_moves_vector(board, side_to_move)        # (64,)

        # 只在合法着上保留概率
        policy = policy * valids.astype(np.float32)

        if policy.sum() <= 0:
            # 理论上不应出现（exam 都是有合法着的残局）
            mask = valids.astype(np.float32)
            if mask.sum() <= 0:
                return ["pass"]
            policy = mask / mask.sum()
        else:
            policy /= policy.sum()

        # 取概率从大到小排序后的前 k 个合法动作
        indices = np.argsort(-policy)
        top_actions = []
        for idx in indices:
            if valids[idx] > 0:
                top_actions.append(idx)
                if len(top_actions) >= k:
                    break

        move_strs = [coord_to_move_str(a // 8, a % 8) for a in top_actions]
        return move_strs


# -----------------------
# Net+MCTS Agent
# -----------------------

class NetMCTSAgent(BaseAgent):
    """
    使用“已训练好的 OthelloNet + MCTS 搜索”的 Agent。
    MCTS 的 game / 搜索逻辑与 MCTSAgent 相同，只是 nnet 换成 NetMCTSNNet。
    """

    def __init__(self, model, device="cuda", num_simulations=200, cpuct=1.0):
        self.game = OthelloGameAdapter(board_size=8)
        self.nnet = NetMCTSNNet(self.game, model=model, device=device)
        self.args = SimpleNamespace(
            numMCTSSims=num_simulations,
            cpuct=cpuct,
            use_root_dirichlet=False,  # 评测时不加 Dirichlet 噪声
        )

    def get_topk_moves(self, board, side_to_move, k=3):
        mcts = MCTS(self.game, self.nnet, self.args)

        state = (board, side_to_move)
        canonical = state

        probs = np.array(mcts.getActionProb(canonical, temp=1.0), dtype=np.float32)
        valids = self.game.getValidMoves(canonical, 1)

        pass_action = self.game.getActionSize() - 1
        candidate_actions = [
            a for a in range(self.game.getActionSize())
            if valids[a] > 0 and a != pass_action
        ]

        if not candidate_actions:
            return ["pass"]

        candidate_actions.sort(key=lambda a: probs[a], reverse=True)
        top_actions = candidate_actions[:k]

        move_strs = [coord_to_move_str(a // 8, a % 8) for a in top_actions]
        return move_strs


# -----------------------
# Ntuple Agent
# -----------------------

class NtupleAgent(BaseAgent):
    """
    使用已有的 Ntuple 价值函数 (value.ModelScorer)：
    - 对每个合法着，临时落子，调用对应一方的 ScorerWrapper(board)
    - 按分数降序排序，取 top-k
    """

    def __init__(self, model_path,
                 learning_rate=0.0,
                 gamma=0.0,
                 optimizer="sgd"):
        # 路径改成绝对路径，方便从脚本目录调用
        if not os.path.isabs(model_path):
            model_path = os.path.join(ROOT_DIR, model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError("Ntuple weight file not found: %s" % model_path)

        # base ntuple 模型：从黑方视角输出值
        base_model = ModelScorer(
            path=model_path,
            learning_rate=learning_rate or 0.0,
            gamma=gamma or 0.0,
            optimizer=optimizer,
        )

        # 包一层 ScorerWrapper，让不同角色得到各自视角的值
        self.scorer_black = ScorerWrapper(Board.BLACK, base_model)
        self.scorer_white = ScorerWrapper(Board.WHITE, base_model)

    def _get_scorer_for_side(self, side_to_move):
        return self.scorer_black if side_to_move == Board.BLACK else self.scorer_white

    def get_topk_moves(self, board, side_to_move, k=3):
        """
        Ntuple 只提供 value，不提供 policy：
        - 穷举所有合法着
        - 对每个合法着假设走一步，调用 ntuple 估值
        - 按估值排序取 top-k
        """
        legal_moves = board.feasible_pos(side_to_move)
        if not legal_moves:
            return ["pass"]

        scorer = self._get_scorer_for_side(side_to_move)

        scored_moves = []
        for (i, j) in legal_moves:
            # flip2: 上下文管理器，内部修改棋盘，退出时自动恢复
            with board.flip2(i, j, side_to_move):
                value = scorer(board)
            scored_moves.append(((i, j), float(value)))

        # 按 value 从大到小排序
        scored_moves.sort(key=lambda x: x[1], reverse=True)

        top_moves = [mv for (mv, v) in scored_moves[:k]]
        move_strs = [coord_to_move_str(i, j) for (i, j) in top_moves]
        return move_strs


# -----------------------
# 评测流程 + 保存结果
# -----------------------

def load_exam_and_answers(exam_path, answer_path):
    with open(exam_path, "r", encoding="utf-8") as f:
        exam_data = json.load(f)

    with open(answer_path, "r", encoding="utf-8") as f:
        answer_data = json.load(f)

    # 把 Edax 答案按 game_id 建索引
    answer_by_game_id = {}
    for entry in answer_data:
        answer_by_game_id[entry["game_id"]] = entry

    return exam_data, answer_by_game_id


def evaluate_agent_on_exam(agent,
                           model_name,
                           exam_path,
                           answer_path,
                           topk=3,
                           max_positions=None,
                           save_dir=None):
    print("[INFO] Using model: %s" % model_name)
    print("[INFO] Loading exam from:   %s" % exam_path)
    print("[INFO] Loading answer from: %s" % answer_path)

    exam_data, answer_by_game_id = load_exam_and_answers(exam_path, answer_path)

    total_items = len(exam_data)
    print("[INFO] Total exam positions (raw): %d" % total_items)

    if max_positions is not None:
        total_items = min(total_items, max_positions)
        print("[INFO] Will evaluate first %d positions only" % total_items)

    total = 0
    top1_correct = 0
    top3_contains_edax_top1 = 0
    top1_in_edax_top3 = 0
    top3_intersection_non_empty = 0

    # 用来保存逐题详细结果
    per_position_results = []

    for idx, item in enumerate(exam_data, start=1):
        if max_positions is not None and idx > total_items:
            break

        game_id = item["game_id"]
        history = item["history"]
        side_to_move_str = item["side_to_move"]

        if game_id not in answer_by_game_id:
            # 没有对应 Edax 结果，跳过
            continue

        edax_entry = answer_by_game_id[game_id]
        edax_top1 = edax_entry["answer"]
        edax_top3 = [x["move"] for x in edax_entry.get("top3", [])]

        board, side_to_move = reconstruct_board_from_history(
            history, side_to_move_str
        )

        model_top_moves = agent.get_topk_moves(board, side_to_move, k=topk)
        if not model_top_moves:
            continue

        model_top1 = model_top_moves[0]

        total += 1

        # 逐题指标
        m_top1_eq_edax_top1 = (model_top1 == edax_top1)
        m_top3_has_edax_top1 = (edax_top1 in model_top_moves)
        m_top1_in_edax_top3 = (bool(edax_top3) and model_top1 in edax_top3)
        m_top3_intersect = bool(edax_top3 and (set(model_top_moves) & set(edax_top3)))

        # 汇总
        if m_top1_eq_edax_top1:
            top1_correct += 1
        if m_top3_has_edax_top1:
            top3_contains_edax_top1 += 1
        if m_top1_in_edax_top3:
            top1_in_edax_top3 += 1
        if m_top3_intersect:
            top3_intersection_non_empty += 1

        # 记录这道题的详细结果
        per_position_results.append({
            "index": idx,
            "game_id": game_id,
            "side_to_move": side_to_move_str,
            "history": history,
            "model_topk": model_top_moves,
            "edax_top1": edax_top1,
            "edax_top3": edax_top3,
            "metrics": {
                "model_top1_eq_edax_top1": m_top1_eq_edax_top1,
                "edax_top1_in_model_topk": m_top3_has_edax_top1,
                "model_top1_in_edax_top3": m_top1_in_edax_top3,
                "model_topk_intersect_edax_top3": m_top3_intersect,
            },
        })

        # 定期打印进度
        if idx % 50 == 0 or idx == total_items:
            print(
                "[PROGRESS] processed %d/%d positions (valid evaluated: %d)"
                % (idx, total_items, total),
                flush=True,
            )

    if total == 0:
        print("[WARN] No matching exam / answer pairs found.")
        return

    # 总体统计
    summary = {
        "model_name": model_name,
        "total_positions_evaluated": total,
        "top1_accuracy": float(top1_correct) / float(total),
        "top3_hit_rate_edax_top1_in_model_topk": float(top3_contains_edax_top1) / float(total),
        "top1_in_edax_top3_rate": float(top1_in_edax_top3) / float(total),
        "top3_intersection_rate": float(top3_intersection_non_empty) / float(total),
    }

    print("========== SUMMARY ==========")
    print("Total positions evaluated: %d" % summary["total_positions_evaluated"])
    print("Top1 accuracy (model top1 == Edax top1): %.4f" %
          summary["top1_accuracy"])
    print("Top3 hit rate (Edax top1 in model top3): %.4f" %
          summary["top3_hit_rate_edax_top1_in_model_topk"])
    print("Top1-in-EdaxTop3 rate (model top1 in Edax top3): %.4f" %
          summary["top1_in_edax_top3_rate"])
    print("Top3 intersection rate (model top3 ∩ Edax top3 != ∅): %.4f" %
          summary["top3_intersection_rate"])

    # --------------------------
    # 把结果保存成 JSON 文件
    # --------------------------
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, "%s_results.json" % model_name)

        out_obj = {
            "summary": summary,
            "per_position": per_position_results,
        }

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2, ensure_ascii=False)

        print("[INFO] Detailed results saved to: %s" % save_path)


# -----------------------
# main：解析命令行参数，选择模型
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate Othello models on exam.json")
    parser.add_argument("--model", type=str, default="mcts",
                        choices=["mcts", "net", "net_mcts", "ntuple"],
                        help="which model to evaluate: mcts / net / net_mcts / ntuple")

    # MCTS 参数（对 mcts 和 net_mcts 都有效）
    parser.add_argument("--mcts-sims", type=int, default=200,
                        help="number of MCTS simulations")
    parser.add_argument("--cpuct", type=float, default=1.0,
                        help="MCTS cpuct parameter")

    # Net 参数
    parser.add_argument("--ckpt", type=str, default="ckpts_phaseC/step_34000.pt",
                        help="checkpoint path for Net model (relative to project root)")

    # Ntuple 参数
    parser.add_argument("--ntuple-path", type=str, default="model/model.cpt.npy",
                        help="checkpoint path for Ntuple ModelScorer weights (relative to project root)")

    # 通用参数
    parser.add_argument("--max-positions", type=int, default=None,
                        help="if set, only evaluate first N positions (debug / speed)")
    parser.add_argument("--topk", type=int, default=3,
                        help="top-k actions to compare with Edax (default 3)")

    args = parser.parse_args()

    exam_dir = os.path.join(ROOT_DIR, "exam")
    exam_path = os.path.join(exam_dir, "exam.json")
    answer_path = os.path.join(exam_dir, "answer.json")

    print("[INFO] exam_path   = %s" % exam_path)
    print("[INFO] answer_path = %s" % answer_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Using device: %s" % device)

    # 选择模型
    if args.model == "mcts":
        print("[INFO] Evaluating MCTS model ...")
        agent = MCTSAgent(num_simulations=args.mcts_sims, cpuct=args.cpuct)
        model_name = "MCTS_sims%d" % args.mcts_sims

    elif args.model == "net":
        print("[INFO] Evaluating Net model ...")
        ckpt_path = args.ckpt
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(ROOT_DIR, ckpt_path)
        print("[INFO] Loading checkpoint: %s" % ckpt_path)

        net = load_model(ckpt_path, device=device)
        agent = NetAgent(net, device=device)

        base_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        model_name = "Net_%s" % base_name

    elif args.model == "net_mcts":
        print("[INFO] Evaluating Net+MCTS model ...")
        ckpt_path = args.ckpt
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(ROOT_DIR, ckpt_path)
        print("[INFO] Loading checkpoint: %s" % ckpt_path)

        net = load_model(ckpt_path, device=device)
        agent = NetMCTSAgent(net, device=device,
                             num_simulations=args.mcts_sims,
                             cpuct=args.cpuct)

        base_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        model_name = "NetMCTS_%s_sims%d" % (base_name, args.mcts_sims)

    elif args.model == "ntuple":
        print("[INFO] Evaluating Ntuple model ...")
        ntuple_path = args.ntuple_path
        agent = NtupleAgent(model_path=ntuple_path)
        base_name = os.path.splitext(os.path.basename(ntuple_path))[0]
        model_name = "Ntuple_%s" % base_name

    else:
        raise ValueError("Unknown model type: %s" % args.model)

    evaluate_agent_on_exam(
        agent=agent,
        model_name=model_name,
        exam_path=exam_path,
        answer_path=answer_path,
        topk=args.topk,
        max_positions=args.max_positions,
        save_dir=exam_dir,
    )


if __name__ == "__main__":
    main()
