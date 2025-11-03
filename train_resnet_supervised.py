import os, sys, math, time, argparse, random, csv
from pathlib import Path
from typing import List, Iterator, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

# ----- AMP compatibility imports -----
try:
    # torch >= 2.0
    from torch.amp import autocast, GradScaler
    _USE_TORCH2 = True
except ModuleNotFoundError:
    from torch.cuda.amp import autocast, GradScaler
    _USE_TORCH2 = False

# --------- project-specific imports ----------
from othello import Board
from value import ModelScorer  # n-tuple model for arena
from ai import Bot             # n-tuple Bot for arena


# ======================== Model: small (scalable) ResNet ========================

class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class OthelloResNet(nn.Module):
    """
    - policy: 输出 64 维落子 logits
    - value : 预测当前执子一方的胜率 p(win|s) ∈ (0,1)
              训练用 BCEWithLogitsLoss（传 logit），推理同时给出 v = 2p-1 以兼容旧评估
    """
    def __init__(self, in_channels=3, channels=64, blocks=8,
                 value_hidden=256, value_logit_scale=0.75, dropout=0.10):
        super().__init__()
        self.value_logit_scale = float(value_logit_scale)

        # stem + trunk
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        # ---------- Policy head ----------
        self.p_head = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.p_fc = nn.Linear(channels * 8 * 8, 64)

        # ---------- Value head (win-prob) ----------
        self.v_head = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.v_gap = nn.AdaptiveAvgPool2d(1)          # [B,C,1,1]
        self.v_ln  = nn.LayerNorm(channels)           # on channel dim
        self.v_fc1 = nn.Linear(channels, value_hidden)
        self.v_act = nn.SiLU(inplace=True)
        self.v_do  = nn.Dropout(p=dropout)
        self.v_fc2 = nn.Linear(value_hidden, 1)

        # init: small last layer, zero bias to避免初期极端 logit
        nn.init.zeros_(self.v_fc2.bias)
        nn.init.normal_(self.v_fc2.weight, mean=0.0, std=0.01)

    def forward(self, x):
        # x: [B,3,8,8]
        h = self.stem(x)
        h = self.body(h)

        # policy
        hp = self.p_head(h).reshape(x.size(0), -1)
        policy_logits = self.p_fc(hp)                 # [B,64]

        # value
        hv = self.v_head(h)                           # [B,C,8,8]
        hv = self.v_gap(hv).squeeze(-1).squeeze(-1)   # [B,C]
        hv = self.v_ln(hv)
        hv = self.v_act(self.v_fc1(hv))
        hv = self.v_do(hv)
        v_logit = self.v_fc2(hv).squeeze(1)           # [B] raw logit
        v_logit = v_logit * self.value_logit_scale    # 温度钳制（防极端）
        p = torch.sigmoid(v_logit)                    # [B] (0,1)
        v = 2.0 * p - 1.0                             # [-1,1] for metrics

        # 返回四个以便训练/评估灵活选择
        return policy_logits, v, p, v_logit


# ======================== Dataset: streaming npz shards ========================

def list_shards(data_dir: Path, limit: int = 0) -> List[Path]:
    files = sorted(data_dir.glob("shard_*.npz"))
    return files if limit <= 0 else files[:limit]

class NPZShardStream(IterableDataset):
    """
    Stream npz shards:
      yields dict{s[3,8,8] float32, a int64, r float32, legal[64] float32}
    """
    def __init__(self, data_dir: str, shuffle_files: bool = True, repeat: bool = True,
                 limit_shards: int = 0, seed: int = 2025, deterministic_stream: bool = False):
        super().__init__()
        self.paths = list_shards(Path(data_dir), limit_shards)
        if not self.paths:
            raise ValueError(f"No shards found under {data_dir}")
        self.shuffle_files = shuffle_files
        self.repeat = repeat
        self.seed = seed
        self.deterministic_stream = deterministic_stream

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        base_seed = self.seed if self.deterministic_stream else (self.seed + int(time.time()))
        rng = random.Random(base_seed)
        paths = self.paths[:]
        while True:
            if self.shuffle_files:
                rng.shuffle(paths)
            for p in paths:
                with np.load(p, allow_pickle=False, mmap_mode="r") as z:
                    s = z["s"]           # (N,3,8,8) uint8
                    a = z["a"]           # (N,) int
                    r = z["r"]           # (N,) float32
                    legal = z["legal_s"] # (N,64) uint8
                    N = s.shape[0]
                    idxs = list(range(N))
                    rng.shuffle(idxs)
                    for i in idxs:
                        yield {
                            "s": torch.from_numpy(s[i].astype(np.float32)),         # [3,8,8]
                            "a": torch.tensor(int(a[i]), dtype=torch.long),         # []
                            "r": torch.tensor(float(r[i]), dtype=torch.float32),    # []
                            "legal": torch.from_numpy(legal[i].astype(np.float32)), # [64]
                        }
            if not self.repeat:
                break

def collate_fn(batch):
    s = torch.stack([b["s"] for b in batch], dim=0)         # [B,3,8,8]
    a = torch.stack([b["a"] for b in batch], dim=0)         # [B]
    r = torch.stack([b["r"] for b in batch], dim=0)         # [B]
    legal = torch.stack([b["legal"] for b in batch], dim=0) # [B,64]
    return {"s": s, "a": a, "r": r, "legal": legal}


# ======================== Symmetry Augmentation (D4) ========================

def _apply_symmetry_8x8(x: torch.Tensor, k_rot: int, flip: bool) -> torch.Tensor:
    x = torch.rot90(x, k=k_rot % 4, dims=(-2, -1))
    if flip:
        x = torch.flip(x, dims=(-1,))
    return x

def _symmetry_transform_sample(s_3x8x8: torch.Tensor, a_idx: torch.Tensor, legal64: torch.Tensor,
                               k_rot: int, flip: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s = s_3x8x8
    s = _apply_symmetry_8x8(s, k_rot, flip)  # channel-wise
    legal = legal64.view(8, 8)
    legal = _apply_symmetry_8x8(legal, k_rot, flip).contiguous().view(64)
    # action
    r, c = int(a_idx.item()) // 8, int(a_idx.item()) % 8
    coord = torch.zeros(8, 8, dtype=torch.float32, device=s.device)
    coord[r, c] = 1.0
    coord = _apply_symmetry_8x8(coord, k_rot, flip)
    nr, nc = torch.nonzero(coord, as_tuple=False)[0].tolist()
    a_new = torch.tensor(nr * 8 + nc, dtype=torch.long, device=a_idx.device)
    return s, a_new, legal

def symmetry_augment_batch(batch: Dict[str, torch.Tensor], prob: float) -> Dict[str, torch.Tensor]:
    if prob <= 0.0:
        return batch
    s, a, legal = batch["s"], batch["a"], batch["legal"]
    B = s.size(0)
    for i in range(B):
        if random.random() < prob:
            k_rot = random.randint(0, 3)
            flip = bool(random.getrandbits(1))
            s_i, a_i, legal_i = _symmetry_transform_sample(s[i], a[i], legal[i], k_rot, flip)
            s[i] = s_i
            a[i] = a_i
            legal[i] = legal_i
    return {"s": s, "a": a, "r": batch["r"], "legal": legal}


# ======================== Losses & evaluation ========================

def masked_policy_loss(logits: torch.Tensor, target_a: torch.Tensor, legal_mask: torch.Tensor):
    """
    logits: [B,64], target_a: [B], legal_mask: [B,64] in {0,1}
    使用真正的 -inf 避免 fp16 下残余概率
    """
    assert logits.shape == legal_mask.shape, f"shape mismatch: {logits.shape} vs {legal_mask.shape}"
    neg_inf = torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype)
    masked_logits = torch.where(legal_mask > 0, logits, neg_inf)
    logp = F.log_softmax(masked_logits, dim=1)
    nll = F.nll_loss(logp, target_a, reduction="mean")
    return nll, logp

def value_loss_fn(pred, target_r, kind="huber", *, bce_use_logits=False, label_smoothing: float = 0.05):
    """
    kind:
      - "bce": pred 为 logit (bce_use_logits=True) 或概率 p (bce_use_logits=False)
               目标 r∈[-1,1] 映射为 y∈[0,1]，draw=0.5；可选 label smoothing toward 0.5
      - "mse": pred 与 r 比较
      - "huber": pred 与 r 比较
    """
    if kind == "bce":
        # r -> y in [0,1]
        y = (target_r + 1.0) * 0.5
        if label_smoothing and label_smoothing > 0.0:
            y = (1.0 - label_smoothing) * y + label_smoothing * 0.5
        if bce_use_logits:
            return F.binary_cross_entropy_with_logits(pred, y, reduction="mean")
        else:
            eps = 1e-6
            p = torch.clamp(pred, eps, 1 - eps)
            return F.binary_cross_entropy(p, y, reduction="mean")

    if kind == "mse":
        return F.mse_loss(pred, target_r, reduction="mean")

    # default huber
    return F.huber_loss(pred, target_r, delta=1.0, reduction="mean")

@torch.no_grad()
def eval_one_epoch(model, loader, device, value_loss_kind="huber",
                   amp_eval=False, amp_dtype=torch.float16,
                   label_smoothing: float = 0.05):
    """
    评估时：
      - 若使用 BCE：用 v_logit 与 y 比较以得到 v_loss；同时用 v=2*sigmoid(v_logit)-1 统计 MSE/MAE/Pearson
      - 若使用 MSE/Huber：直接用 v 与 r 比较
    """
    model.eval()
    total = 0
    p_loss_sum = v_loss_sum = 0.0
    top1 = legal_top1 = 0
    vs = []
    rs = []
    amp_ctx = autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_eval) if _USE_TORCH2 else autocast(enabled=amp_eval)
    with amp_ctx:
        for batch in loader:
            s = batch["s"].to(device, non_blocking=True)
            a = batch["a"].to(device, non_blocking=True)
            r = batch["r"].to(device, non_blocking=True)
            legal = batch["legal"].to(device, non_blocking=True)

            policy_logits, v, p, v_logit = model(s)

            # policy metrics
            p_loss, _ = masked_policy_loss(policy_logits, a, legal)

            # value loss
            if value_loss_kind == "bce":
                v_loss = value_loss_fn(v_logit, r, kind="bce", bce_use_logits=True,
                                       label_smoothing=label_smoothing)
                v_for_metrics = 2.0 * torch.sigmoid(v_logit) - 1.0
            else:
                v_loss = value_loss_fn(v, r, value_loss_kind)
                v_for_metrics = v

            total += s.size(0)
            p_loss_sum += float(p_loss) * s.size(0)
            v_loss_sum += float(v_loss) * s.size(0)

            neg_inf = torch.tensor(float('-inf'), device=policy_logits.device, dtype=policy_logits.dtype)
            masked_logits = torch.where(legal > 0, policy_logits, neg_inf)
            pred_a = masked_logits.argmax(dim=1)
            top1 += int((pred_a == a).sum())
            legal_top1 += int(legal.gather(1, pred_a.view(-1,1)).sum())

            vs.append(v_for_metrics.detach().float().cpu())
            rs.append(r.detach().float().cpu())

    out = {
        "p_loss": p_loss_sum / max(1, total),
        "v_loss": v_loss_sum / max(1, total),
        "top1": top1 / max(1, total),
        "legal_top1": legal_top1 / max(1, total),
        "count": total,
    }
    # value head extra metrics (以 v∈[-1,1] 对 r 统计)
    if len(vs) > 0:
        v_all = torch.cat(vs, dim=0)
        r_all = torch.cat(rs, dim=0)
        mse = torch.mean((v_all - r_all) ** 2).item()
        mae = torch.mean(torch.abs(v_all - r_all)).item()
        vx = v_all - v_all.mean()
        rx = r_all - r_all.mean()
        denom = torch.sqrt((vx**2).sum() * (rx**2).sum()) + 1e-12
        pear = float((vx * rx).sum() / denom)
        out.update({"v_mse": mse, "v_mae": mae, "v_pearson": pear})
    return out


# ======================== Arena helpers ========================

def rc2index(r: int, c: int) -> int:
    return r * 8 + c

def index2rc(a: int) -> Tuple[int,int]:
    return a // 8, a % 8

def encode_current_planes(board: Board, side: int) -> torch.Tensor:
    grid = getattr(board, "_board", None)
    if grid is None:
        raise RuntimeError("Board._board not found; adapt encode_current_planes() to your Board.")
    g = np.array(grid, dtype=np.int8)
    if side == Board.BLACK:
        me_val, opp_val = 1, -1
    else:
        me_val, opp_val = -1, 1
    me    = (g == me_val).astype(np.float32)
    opp   = (g == opp_val).astype(np.float32)
    empty = (g == 0).astype(np.float32)
    x = np.stack([me, opp, empty], axis=0)   # [3,8,8]
    return torch.from_numpy(x)

def legal_mask_from_board(board: Board, side: int) -> torch.Tensor:
    mask = np.zeros(64, dtype=np.float32)
    for (r, c) in board.feasible_pos(side):
        mask[rc2index(r, c)] = 1.0
    return torch.from_numpy(mask)

@torch.no_grad()
def nn_pick(model: nn.Module, board: Board, side: int, device: torch.device, eps: float = 0.0):
    options = board.feasible_pos(side)
    if not options:
        return None
    if eps > 0.0 and random.random() < eps:
        return random.choice(options)

    x = encode_current_planes(board, side).unsqueeze(0).to(device)  # [1,3,8,8]
    policy_logits, _, _, _ = model(x)                               # [1,64], ...
    logits = policy_logits[0]
    legal = legal_mask_from_board(board, side).to(device)           # [64]
    neg_inf = torch.tensor(float('-inf'), device=logits.device, dtype=logits.dtype)
    masked = torch.where(legal > 0, logits, neg_inf)
    a = int(masked.argmax().item())
    r, c = index2rc(a)
    if (r, c) not in options:
        r, c = options[0]
    return (r, c)

def apply_random_opening(b: Board, plies: int, seed: int):
    if plies <= 0:
        return
    rng = random.Random(seed)
    side = Board.BLACK
    for _ in range(plies):
        if b.is_terminal_state():
            break
        opts = b.feasible_pos(side)
        if opts:
            r, c = rng.choice(opts)
            b.flip(r, c, side)
        side = Board.opponent(side)

def play_game_nn_vs_random(model: nn.Module, device: torch.device,
                           nn_side: int = Board.BLACK, seed: int = 0,
                           opening_plies: int = 0, nn_eps: float = 0.0) -> int:
    rng = random.Random(seed)
    b = Board(); b.init_board()
    apply_random_opening(b, opening_plies, seed)
    side = Board.BLACK
    while not b.is_terminal_state():
        options = b.feasible_pos(side)
        if options:
            if side == nn_side:
                mv = nn_pick(model, b, side, device, eps=nn_eps) or options[0]
                b.flip(mv[0], mv[1], side)
            else:
                r, c = rng.choice(options)
                b.flip(r, c, side)
        side = Board.opponent(side)
    sb, sw = b.score(Board.BLACK), b.score(Board.WHITE)
    nn_is_black = (nn_side == Board.BLACK)
    if sb > sw:  return 1 if nn_is_black else -1
    if sb < sw:  return -1 if nn_is_black else 1
    return 0

def play_game_nn_vs_ntuple(model: nn.Module, device: torch.device,
                           ntuple: ModelScorer, nt_depth: int, nt_width: int,
                           nn_side: int = Board.BLACK, seed: int = 0,
                           opening_plies: int = 0, nn_eps: float = 0.0) -> int:
    b = Board(); b.init_board()
    apply_random_opening(b, opening_plies, seed)
    bot_b = Bot(ntuple, nt_depth, nt_width, Board.BLACK)
    bot_w = Bot(ntuple, nt_depth, nt_width, Board.WHITE)
    side = Board.BLACK
    while not b.is_terminal_state():
        options = b.feasible_pos(side)
        if options:
            if side == nn_side:
                mv = nn_pick(model, b, side, device, eps=nn_eps) or options[0]
                b.flip(mv[0], mv[1], side)
            else:
                if side == Board.BLACK:
                    _, (r, c) = bot_b._play(b)
                else:
                    _, (r, c) = bot_w._play(b)
                b.flip(r, c, side)
        side = Board.opponent(side)
    sb, sw = b.score(Board.BLACK), b.score(Board.WHITE)
    nn_is_black = (nn_side == Board.BLACK)
    if sb > sw:  return 1 if nn_is_black else -1
    if sb < sw:  return -1 if nn_is_black else 1
    return 0

@torch.no_grad()
def arena_eval(model: nn.Module, device: torch.device, games: int,
               ntuple_path: Optional[str], nt_depth: int, nt_width: int,
               seed: int,
               opening_plies: int,
               opening_seed_offset: int,
               nn_eps: float) -> Dict[str, float]:
    # NN vs Random
    w=l=d=0
    for k in range(0, games, 2):
        opening_seed = seed + k
        r1 = play_game_nn_vs_random(model, device, nn_side=Board.BLACK,
                                    seed=opening_seed,
                                    opening_plies=opening_plies, nn_eps=nn_eps)
        r2 = play_game_nn_vs_random(model, device, nn_side=Board.WHITE,
                                    seed=opening_seed,
                                    opening_plies=opening_plies, nn_eps=nn_eps)
        for r in (r1, r2):
            if   r>0: w+=1
            elif r<0: l+=1
            else:     d+=1
    wr_rand = w / max(1, w+l+d)

    # NN vs NTuple
    wr_nt = -1.0
    if ntuple_path and Path(ntuple_path).exists():
        nt = ModelScorer(learning_rate=0.0, gamma=0.01)
        nt.load(ntuple_path)
        w=l=d=0
        for k in range(0, games, 2):
            opening_seed = seed + opening_seed_offset + k
            r1 = play_game_nn_vs_ntuple(model, device, nt, nt_depth, nt_width,
                                        nn_side=Board.BLACK,
                                        seed=opening_seed,
                                        opening_plies=opening_plies, nn_eps=nn_eps)
            r2 = play_game_nn_vs_ntuple(model, device, nt, nt_depth, nt_width,
                                        nn_side=Board.WHITE,
                                        seed=opening_seed,
                                        opening_plies=opening_plies, nn_eps=nn_eps)
            for r in (r1, r2):
                if   r>0: w+=1
                elif r<0: l+=1
                else:     d+=1
        wr_nt = w / max(1, w+l+d)

    return {"wr_vs_random": wr_rand, "wr_vs_ntuple": wr_nt}


# ======================== Teacher agreement evaluation ========================

def _board_from_current_planes_tensor(x_3x8x8: torch.Tensor) -> Tuple[Board, int]:
    x = x_3x8x8.detach().cpu().numpy()
    assert x.shape == (3, 8, 8), f"bad shape: {x.shape}"
    me, opp, empty = x[0], x[1], x[2]
    me_i  = (me  > 0.5).astype(np.int32)
    opp_i = (opp > 0.5).astype(np.int32)
    g = me_i * 1 + opp_i * (-1)              # +1 / -1 / 0, int32
    b = Board()
    if hasattr(b, "init_board"):
        b.init_board()
    if hasattr(b, "_board"):
        b._board = g.copy()
    elif hasattr(b, "board"):
        setattr(b, "board", g.copy())
    else:
        raise RuntimeError("Cannot set Board grid; adapt this function to your Board class.")
    for attr in ["current_side", "side_to_move", "player", "turn"]:
        if hasattr(b, attr):
            try:
                setattr(b, attr, Board.BLACK)
            except Exception:
                pass
    return b, Board.BLACK


@torch.no_grad()
def eval_agreement_with_ntuple(model: nn.Module,
                               loader: DataLoader,
                               ntuple_path: str,
                               nt_depth: int,
                               nt_width: int,
                               device: torch.device,
                               max_batches: int = 200,
                               topk: int = 3) -> Dict[str, float]:
    if not (ntuple_path and Path(ntuple_path).exists()):
        return {"match_top1": -1.0, "match_topk": -1.0,
                "teacher_ce": -1.0, "teacher_ppl": -1.0,
                "count": 0, "skipped": 0}

    nt = ModelScorer(learning_rate=0.0, gamma=0.01)
    nt.load(ntuple_path)
    bot_b = Bot(nt, nt_depth, nt_width, Board.BLACK)
    bot_w = Bot(nt, nt_depth, nt_width, Board.WHITE)

    model.eval()
    n = 0
    skipped = 0
    agree1 = 0
    agreek = 0
    ce_sum = 0.0

    iters = 0
    for batch in loader:
        iters += 1
        if iters > max_batches:
            break

        s = batch["s"].to(device, non_blocking=True)         # [B,3,8,8]
        legal = batch["legal"].to(device, non_blocking=True) # [B,64]
        legal_sum = legal.sum(dim=1)                          # [B]

        policy_logits, _, _, _ = model(s)                     # [B,64], ...
        neg_inf = torch.tensor(float('-inf'), device=policy_logits.device, dtype=policy_logits.dtype)
        masked_logits = torch.where(legal > 0, policy_logits, neg_inf)
        logp = F.log_softmax(masked_logits, dim=1)
        topk_k = min(topk, masked_logits.size(1))
        _, topk_idx = torch.topk(masked_logits, k=topk_k, dim=1)

        B = s.size(0)
        for i in range(B):
            if legal_sum[i].item() <= 0:
                skipped += 1
                continue

            try:
                b, _ = _board_from_current_planes_tensor(s[i])
            except Exception:
                skipped += 1
                continue

            res = None
            try:
                res = bot_b._play(b)
            except Exception:
                res = None
            if not res or not isinstance(res, tuple) or len(res) != 2 or res[1] is None:
                try:
                    res = bot_w._play(b)
                except Exception:
                    res = None
            if not res or not isinstance(res, tuple) or len(res) != 2 or res[1] is None:
                skipped += 1
                continue

            value_ret, action = res
            if action is None or not isinstance(action, (tuple, list)) or len(action) != 2:
                skipped += 1
                continue
            tr, tc = int(action[0]), int(action[1])
            t_idx = rc2index(tr, tc)

            if t_idx < 0 or t_idx >= 64 or legal[i, t_idx].item() <= 0.0:
                skipped += 1
                continue

            nn_top1 = int(torch.argmax(masked_logits[i]).item())
            agree1 += int(nn_top1 == t_idx)
            agreek += int(t_idx in topk_idx[i].tolist())
            ce_sum += float(-logp[i, t_idx].item())
            n += 1

    if n == 0:
        return {"match_top1": -1.0, "match_topk": -1.0,
                "teacher_ce": -1.0, "teacher_ppl": -1.0,
                "count": 0, "skipped": skipped}

    ce = ce_sum / n
    return {
        "match_top1": agree1 / n,
        "match_topk": agreek / n,
        "teacher_ce": ce,
        "teacher_ppl": math.exp(min(50.0, ce)),
        "count": n,
        "skipped": skipped
    }


# ======================== Utils ========================

CSV_FIELDS = ["phase","epoch","step","updates","p_loss","v_loss","top1","legal",
              "v_mse","v_mae","v_pearson","match_top1","match_topk","teacher_ce",
              "teacher_ppl","agree_topk","wr_vs_random","wr_vs_ntuple","nt_depth",
              "nt_width","games","open_plies","nn_eps","lr","ips","count","skipped"]

def append_csv(path: Path, row: Dict):
    header_needed = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if header_needed:
            writer.writeheader()
        writer.writerow({k: row.get(k, None) for k in CSV_FIELDS})

def save_checkpoint(path: Path, model, opt, sched, args, step, updates, epoch, extra: Dict):
    ckpt = {
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "sched": sched.state_dict() if sched is not None else None,
        "args": vars(args),
        "step": step, "updates": updates, "epoch": epoch,
        **extra,
    }
    torch.save(ckpt, str(path))


# ======================== Train loop ========================

def main():
    ap = argparse.ArgumentParser()
    # ===== 数据与数据流 =====
    ap.add_argument("--train-dir", type=str, required=True, help="训练数据目录（npz shards）")
    ap.add_argument("--val-dir",   type=str, default="", help="验证数据目录（留空则用训练集子集）")
    ap.add_argument("--limit-train-shards", type=int, default=25, help="训练使用的 shard 数量（0=全部）")
    ap.add_argument("--limit-val-shards",   type=int, default=4,  help="验证使用的 shard 数量（0=全部）")
    ap.add_argument("--train-repeat", action="store_true", default=True, help="训练数据流循环")
    ap.add_argument("--deterministic-stream", action="store_true", default=False,
                    help="固定随机种子（不叠加 time.time），便于复现")

    # ===== 训练时长与批次 =====
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps-per-epoch", type=int, default=30000, help="每 epoch 的优化步数；0=不限")
    ap.add_argument("--max-steps", type=int, default=30000, help="总优化步数上限；0=不限")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--accum-steps", type=int, default=2, help="梯度累积步数（总有效batch=bs*accum）")

    # ===== 优化器与正则 =====
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)

    # ===== 模型规模 =====
    ap.add_argument("--channels", type=int, default=192)
    ap.add_argument("--blocks",   type=int, default=16)
    ap.add_argument("--value-hidden", type=int, default=512)

    # ===== 损失与权重（价值头为 win-prob，默认 BCE）=====
    ap.add_argument("--value-loss", type=str, choices=["huber","mse","bce"], default="bce")
    ap.add_argument("--alpha-policy", type=float, default=1.5, help="策略损失系数")
    ap.add_argument("--beta-value",   type=float, default=1.0, help="价值损失系数")

    # ===== 价值头训练强化与稳定器 =====
    ap.add_argument("--v-lr-mult", type=float, default=2.0, help="价值头学习率倍率（>1 增强价值学习）")
    ap.add_argument("--value-logit-scale", type=float, default=0.5, help="价值 logit 温度（<1 抑制极端）")
    ap.add_argument("--value-dropout", type=float, default=0.10, help="价值头 Dropout 概率")
    ap.add_argument("--value-label-smoothing", type=float, default=0.05,
                    help="BCE 标签向 0.5 平滑，缓解饱和")

    # ===== 数据增广 =====
    ap.add_argument("--symmetry-aug", type=float, default=0.8, help="D4 对称增广概率（0~1）")

    # ===== AMP / 计算资源 =====
    ap.add_argument("--amp", action="store_true", help="启用混合精度（优先 bfloat16，否则 fp16+Scaler）")
    ap.add_argument("--num-workers", type=int, default=4)

    # ===== 日志 / 复现 / 输出 =====
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--ckpt-dir", type=str, default="ckpts_phaseA")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1000)

    # ===== Arena（对战评测，默认开启）=====
    ap.add_argument("--arena-every", type=int, default=2000, help="每隔多少次优化步做一次对战评测；0=关闭")
    ap.add_argument("--arena-games", type=int, default=400, help="每次评测的对局数（含换色）")
    ap.add_argument("--ntuple-path", type=str, default="model/model_ep180000.cpt.npy")
    ap.add_argument("--ntuple-depth", type=int, default=1)
    ap.add_argument("--ntuple-width", type=int, default=1)
    ap.add_argument("--arena-opening-plies", type=int, default=6, help="随机开局步数（对称共享）")
    ap.add_argument("--arena-opening-seed-offset", type=int, default=777, help="与 Random 对局的 seed 偏移")
    ap.add_argument("--arena-nn-eps", type=float, default=0.00, help="对战时 NN epsilon-greedy")

    # ===== Teacher agreement（默认关闭，省时）=====
    ap.add_argument("--agree-every", type=int, default=0, help="0=关闭；>0 表示每隔多少步评 teacher 一致性")
    ap.add_argument("--agree-batches", type=int, default=150, help="一致性评估的验证批次数")
    ap.add_argument("--agree-topk", type=int, default=3, help="top-k 一致性指标")

    args = ap.parse_args()

    # 规范化开关型步频：<=0 视为关闭
    args.log_every  = args.log_every  if args.log_every  and args.log_every  > 0 else 0
    args.eval_every = args.eval_every if args.eval_every and args.eval_every > 0 else 0
    args.agree_every = args.agree_every if args.agree_every and args.agree_every > 0 else 0
    args.arena_every = args.arena_every if args.arena_every and args.arena_every > 0 else 0

    

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    use_cuda = (device.type == "cuda")
    prefer_bf16 = use_cuda and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = (torch.bfloat16 if prefer_bf16 else torch.float16)
    amp_enabled = args.amp and use_cuda
    need_scaler = amp_enabled and (amp_dtype is torch.float16)
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # Datasets / Loaders
    train_ds = NPZShardStream(
        args.train_dir,
        shuffle_files=True,
        repeat=args.train_repeat,
        limit_shards=args.limit_train_shards,
        seed=args.seed,
        deterministic_stream=args.deterministic_stream,
    )
    if args.val_dir:
        val_ds = NPZShardStream(
            args.val_dir,
            shuffle_files=False,   # ← 这里
            repeat=False,
            limit_shards=args.limit_val_shards or 2,
            seed=args.seed+1,
            deterministic_stream=True
        )
    else:
        val_ds = NPZShardStream(
            args.train_dir,
            shuffle_files=False,   # ← 这里
            repeat=False,
            limit_shards=args.limit_val_shards or 1,
            seed=args.seed+1,
            deterministic_stream=True
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=2,
        pin_memory=True, collate_fn=collate_fn
    )

    # Model
    model = OthelloResNet(
        channels=args.channels, blocks=args.blocks, value_hidden=args.value_hidden,
        value_logit_scale=args.value_logit_scale, dropout=args.value_dropout
    ).to(device)

    # Param groups: value head 独立 LR
    v_params = (list(model.v_head.parameters()) +
                list(model.v_ln.parameters()) +
                list(model.v_fc1.parameters()) +
                list(model.v_fc2.parameters()))
    v_param_ids = set(id(p) for p in v_params)
    base_params = [p for p in model.parameters() if id(p) not in v_param_ids]

    opt = torch.optim.AdamW(
        [{"params": base_params, "lr": args.lr},
         {"params": v_params,    "lr": args.lr * args.v_lr_mult}],
        weight_decay=args.weight_decay
    )

    # LR schedule 绑定真实总步数
    total_steps = args.max_steps if args.max_steps > 0 else (args.epochs * (args.steps_per_epoch or 10000))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps))

    # AMP 上下文 & GradScaler
    if _USE_TORCH2:
        train_amp_ctx = lambda: autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled)
    else:
        train_amp_ctx = lambda: autocast(enabled=amp_enabled)
    scaler = GradScaler(enabled=need_scaler)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ckpt_dir / "train_log.csv"

    best_val = float("inf")
    step = 0
    updates = 0
    t0 = time.time()

    print(f"[startup] device={device}, amp={amp_enabled}, bs={args.batch_size}, "
      f"workers={args.num_workers}, steps/epoch={args.steps_per_epoch}, max_steps={args.max_steps}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        steps_this_epoch = 0
        run_n = 0
        run_p = run_v = run_top1 = run_legal = 0.0

        for it, batch in enumerate(train_loader, start=1):
            step += 1
            steps_this_epoch += 1

            s = batch["s"].to(device, non_blocking=True)
            a = batch["a"].to(device, non_blocking=True)
            r = batch["r"].to(device, non_blocking=True)
            legal = batch["legal"].to(device, non_blocking=True)

            if args.symmetry_aug > 0.0:
                aug_batch = {"s": s, "a": a, "r": r, "legal": legal}
                aug_batch = symmetry_augment_batch(aug_batch, prob=args.symmetry_aug)
                s, a, r, legal = aug_batch["s"], aug_batch["a"], aug_batch["r"], aug_batch["legal"]

            with train_amp_ctx():
                policy_logits, v, p, v_logit = model(s)
                p_loss, _ = masked_policy_loss(policy_logits, a, legal)
                if args.value_loss == "bce":
                    v_loss = value_loss_fn(v_logit, r, kind="bce", bce_use_logits=True,
                                           label_smoothing=args.value_label_smoothing)
                else:
                    v_loss = value_loss_fn(v, r, args.value_loss)
                loss = args.alpha_policy * p_loss + args.beta_value * v_loss

            if need_scaler:
                scaler.scale(loss / args.accum_steps).backward()
            else:
                (loss / args.accum_steps).backward()

            with torch.no_grad():
                neg_inf = torch.tensor(float('-inf'), device=policy_logits.device, dtype=policy_logits.dtype)
                masked_logits = torch.where(legal > 0, policy_logits, neg_inf)
                pred_a = masked_logits.argmax(dim=1)
                top1 = (pred_a == a).float().mean()
                legal_top1 = legal.gather(1, pred_a.view(-1,1)).float().mean()
                bs = s.size(0)
                run_n += bs
                run_p += float(p_loss) * bs
                run_v += float(v_loss) * bs
                run_top1 += float(top1) * bs
                run_legal += float(legal_top1) * bs

            if step % args.accum_steps == 0:
                if need_scaler:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                sched.step()
                updates += 1

                # logging
                if args.log_every > 0 and updates % args.log_every == 0 and run_n > 0:
                    dt = time.time() - t0
                    ips = run_n / max(1e-9, dt)
                    msg = (f"[train] ep={epoch} step={step} upd={updates} "
                           f"p_loss={run_p/run_n:.4f} v_loss={run_v/run_n:.4f} "
                           f"top1={run_top1/run_n:.3f} legal={run_legal/run_n:.3f} "
                           f"lr={opt.param_groups[0]['lr']:.2e} ips={ips:.1f}")
                    print(msg)
                    append_csv(csv_path, {
                        "phase":"train","epoch":epoch,"step":step,"updates":updates,
                        "p_loss":run_p/run_n,"v_loss":run_v/run_n,"top1":run_top1/run_n,
                        "legal":run_legal/run_n,"lr":opt.param_groups[0]['lr'],
                        "ips":ips
                    })
                    t0 = time.time()
                    run_n = run_p = run_v = run_top1 = run_legal = 0.0

                # eval
                if args.eval_every > 0 and updates % args.eval_every == 0:
                    stats = eval_one_epoch(
                        model, val_loader, device, args.value_loss,
                        amp_eval=False, amp_dtype=amp_dtype,
                        label_smoothing=args.value_label_smoothing
                    )
                    # 汇总评分：bce 下 v_loss 是 BCE；为了可比性，仍用 alpha*policy + beta*value
                    val_score = args.alpha_policy*stats["p_loss"] + args.beta_value*stats["v_loss"]
                    print(f"[eval]  ep={epoch} step={step} upd={updates} "
                          f"p_loss={stats['p_loss']:.4f} v_loss={stats['v_loss']:.4f} "
                          f"top1={stats['top1']:.3f} legal={stats['legal_top1']:.3f} count={stats['count']}")
                    append_csv(csv_path, {
                        "phase":"eval","epoch":epoch,"step":step,"updates":updates,
                        "p_loss":stats["p_loss"],"v_loss":stats["v_loss"],
                        "top1":stats["top1"],"legal":stats["legal_top1"],
                        "v_mse":stats.get("v_mse",None),
                        "v_mae":stats.get("v_mae",None),
                        "v_pearson":stats.get("v_pearson",None),
                        "lr":opt.param_groups[0]["lr"],
                        "count":stats.get("count", None)
                    })
                    save_checkpoint(ckpt_dir / f"ckpt_step{step}.pt",
                                    model, opt, sched, args, step, updates, epoch,
                                    {"val": stats})
                    if val_score < best_val:
                        best_val = val_score
                        save_checkpoint(ckpt_dir / "best.pt",
                                        model, opt, sched, args, step, updates, epoch,
                                        {"val": stats})
                        print(f"[ckpt] best updated: score={best_val:.4f}")

                # teacher agreement eval
                if args.agree_every > 0 and updates % args.agree_every == 0:
                    agr = eval_agreement_with_ntuple(
                        model, val_loader,
                        args.ntuple_path, args.ntuple_depth, args.ntuple_width,
                        device, max_batches=args.agree_batches, topk=args.agree_topk
                    )
                    print(f"[agree] ep={epoch} step={step} upd={updates} "
                          f"match@1={agr['match_top1']:.3f} match@{args.agree_topk}={agr['match_topk']:.3f} "
                          f"tCE={agr['teacher_ce']:.4f} ppl={agr['teacher_ppl']:.2f} "
                          f"count={agr['count']} skipped={agr['skipped']}")
                    append_csv(csv_path, {
                        "phase":"agree","epoch":epoch,"step":step,"updates":updates,
                        "match_top1":agr["match_top1"],"match_topk":agr["match_topk"],
                        "teacher_ce":agr["teacher_ce"],"teacher_ppl":agr["teacher_ppl"],
                        "agree_topk":args.agree_topk,
                        "nt_depth":args.ntuple_depth,"nt_width":args.ntuple_width,
                        "count":agr["count"], "skipped":agr["skipped"]
                    })

                # arena
                if args.arena_every > 0 and updates % args.arena_every == 0:
                    arena = arena_eval(model, device, args.arena_games,
                                       args.ntuple_path, args.ntuple_depth, args.ntuple_width,
                                       seed=1234,
                                       opening_plies=args.arena_opening_plies,
                                       opening_seed_offset=args.arena_opening_seed_offset,
                                       nn_eps=args.arena_nn_eps)
                    print(f"[arena] ep={epoch} step={step} upd={updates} "
                          f"wr_vs_random={arena['wr_vs_random']:.3f} wr_vs_ntuple={arena['wr_vs_ntuple']:.3f} "
                          f"(games={args.arena_games}, nt={args.ntuple_depth}/{args.ntuple_width}, "
                          f"open={args.arena_opening_plies}, eps={args.arena_nn_eps})")
                    append_csv(csv_path, {
                        "phase":"arena","epoch":epoch,"step":step,"updates":updates,
                        "wr_vs_random":arena["wr_vs_random"],"wr_vs_ntuple":arena["wr_vs_ntuple"],
                        "nt_depth":args.ntuple_depth,"nt_width":args.ntuple_width,
                        "games":args.arena_games,
                        "open_plies":args.arena_opening_plies,
                        "nn_eps":args.arena_nn_eps
                    })

                # stop by max-steps (optimizer steps)
                if args.max_steps > 0 and updates >= args.max_steps:
                    print("[stop] reached max-steps; exiting.")
                    stats = eval_one_epoch(
                        model, val_loader, device, args.value_loss,
                        amp_eval=False, amp_dtype=amp_dtype,
                        label_smoothing=args.value_label_smoothing
                    )
                    save_checkpoint(ckpt_dir / "last.pt",
                                    model, opt, sched, args, step, updates, epoch,
                                    {"val": stats})
                    return

            if args.steps_per_epoch > 0 and steps_this_epoch >= args.steps_per_epoch:
                break

    print("[done] training finished.")


if __name__ == "__main__":
    main()
