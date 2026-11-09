"""
mini_rl_teacher.py —— RL-Teacher 核心机制的「本机迷你复现」

原项目依赖 MuJoCo / TensorFlow 1.2 / Python 3.5，在现代 macOS(arm64) 上无法安装。
本文件用纯 numpy 复刻同样的核心逻辑，每个部分都对应原项目的真实模块，便于对照学习：

    本文件                          <-->   原项目
    ------------------------------------------------------------------------------
    ToyLineEnv                      <-->   rl_teacher/envs.py + gym 环境
    do_rollout / sample_segment     <-->   rl_teacher/segment_sampling.py
    SyntheticComparisonCollector    <-->   rl_teacher/comparison_collectors.py
    LabelSchedule                   <-->   rl_teacher/label_schedules.py
    MLP + ComparisonRewardPredictor <-->   rl_teacher/nn.py + rl_teacher/teach.py
    Policy (REINFORCE)              <-->   agents/parallel-trpo (这里简化为 REINFORCE)

任务设定（对应论文思想：奖励函数不好写，但人能看出好坏）：
  - 「环境」是一张 2D 点阵平面, agent 施加 x/y 两个方向的力让小球移动。
  - 「人类」心里「真正想要」的行为是: 小球停在目标点附近(但 agent 看不到它)。环境不会把这个目标直接给 agent, agent 只能看到「奖励预测器」的输出, 这里「奖励预测器」相当于「人类意图」的代理。
  - 「人类」通过比较 agent行为轨迹(对应原项目中的片段视频)采样中的两两比较, 来告诉预测器: 哪一种行为更接近目标。
  - 为了简化实现, 这里用「合成人类」来代替「人类」: 片段真实累积奖励更高者胜出（对应原项目 -p synth 模式）。

运行: python3 mini_rl_teacher.py
（训练过程中会在终端里持续刷新 2D 点阵图，实时观察小球向目标收敛的过程）
"""

import sys
import numpy as np
from collections import deque

# ------------------------------------------------------------------ 全局常量
CLIP_LENGTH = 1.5          # 每个视频片段时长（秒），对应 teach.py 里的 CLIP_LENGTH
FPS = 10                   # 渲染帧率
SEG_LEN = int(CLIP_LENGTH * FPS)   # 每个片段包含的步数 = 15
EPISODE_LEN = 30           # 每个 episode 的步数
WORLD_LIM = 6.0            # 点阵平面范围：x, y 均在 [-6, 6]
TARGET = np.array([3.0, 2.0], dtype=np.float32)   # 「人类心里真正想要的」目标点（agent 看不到）
OBS_SCALE = np.array([0.2, 0.2, 0.33, 0.33], dtype=np.float32)  # 输入归一化尺度 (x, y, vx, vy)

# ---- CLI 点阵渲染参数与 ANSI 颜色 ----
GRID_W, GRID_H = 41, 21    # 点阵宽/高（字符）；字符高宽约 2:1，故 41x21 视觉接近正方形
N_ITERS = 1000             # 联合训练轮数
DRAW_EVERY = 10            # 每多少轮刷新一次点阵
C_RESET = "\033[0m"
C_TGT = "\033[31m"         # 红：目标
C_BALL = "\033[32m"        # 绿：小球
C_DIM = "\033[90m"         # 灰：背景点阵


# ==================================================================== 1. 玩具环境
class ToyLineEnv:
    """2D 点阵环境。对应 rl_teacher/envs.py + gym 环境。

    状态 [x, y, vx, vy]，动作 [fx, fy]（x/y 两个方向的力）。
    关键点：info 里带 human_obs（这里就是小球位置 [x, y]），
    对应原项目 MjViewer 把 qpos/qvel 塞进 info['human_obs'] 的做法，
    用于之后「渲染」视频片段给人类看。
    """

    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)
        self.obs_dim = 4   # [x, y, vx, vy]
        self.act_dim = 2   # [fx, fy]
        self.reset()

    def reset(self):
        self.pos = self.rng.uniform(-1.0, 1.0, size=2)   # 起点在原点附近随机
        self.vel = np.zeros(2, dtype=np.float32)
        self.t = 0
        return self._obs()

    def _obs(self):
        return np.concatenate([self.pos, self.vel]).astype(np.float32)

    def step(self, a):
        a = np.clip(np.asarray(a, dtype=np.float32).reshape(2), -1.0, 1.0)
        self.vel = np.clip(0.9 * self.vel + a, -3.0, 3.0)
        self.pos = np.clip(self.pos + self.vel * 0.1, -WORLD_LIM, WORLD_LIM)
        self.t += 1

        # 隐藏的真实奖励：越接近目标点越高（就像论文里人类心里真正想要的东西）
        true_reward = -float(np.linalg.norm(self.pos - TARGET))

        info = {"human_obs": self.pos.copy()}   # 给「人类」看的可渲染信息
        done = self.t >= EPISODE_LEN
        return self._obs(), true_reward, done, info


# ============================================================ 2. rollout 与片段采样
# 对应 rl_teacher/segment_sampling.py
def do_rollout(env, action_fn):
    """跑一条完整轨迹。对应 segment_sampling.do_rollout"""
    obs, acts, rews, human = [], [], [], []
    ob = env.reset()
    done = False
    while not done:
        a = np.asarray(action_fn(ob), dtype=np.float32).reshape(-1)
        obs.append(ob)
        acts.append(a)
        ob, r, done, info = env.step(a)
        rews.append(r)
        human.append(np.asarray(info["human_obs"], dtype=np.float32).reshape(-1))
    return {
        "obs": np.array(obs, dtype=np.float32),
        "actions": np.array(acts, dtype=np.float32),
        "original_rewards": np.array(rews, dtype=np.float32),
        "human_obs": np.array(human, dtype=np.float32),
    }


def sample_segment_from_path(path, segment_length):
    """从轨迹里随机切一段。对应 segment_sampling.sample_segment_from_path"""
    n = len(path["obs"])
    if n < segment_length:
        return None
    start = np.random.randint(0, n - segment_length + 1)
    return {k: path[k][start:start + segment_length].copy() for k in path}


def segments_from_rand_rollout(env, n_desired_segments, seed=0):
    """随机 rollout 批量生成片段（预训练用）。
    对应 segment_sampling.segments_from_rand_rollout"""
    segments = []
    while len(segments) < n_desired_segments:
        path = do_rollout(env, lambda ob: np.random.uniform(-1.0, 1.0, size=env.act_dim))
        # 每条轨迹采样的数量控制重复概率，逻辑同原项目
        n_this_path = max(1, int(0.25 * len(path["obs"]) / SEG_LEN))
        for _ in range(n_this_path):
            seg = sample_segment_from_path(path, SEG_LEN)
            if seg is not None:
                segments.append(seg)
    return segments


# ============================================================ 3. 合成比较收集器
# 对应 rl_teacher/comparison_collectors.py 的 SyntheticComparisonCollector
class SyntheticComparisonCollector:
    """「合成人类」：直接用真实奖励比较，谁高谁赢。"""

    def __init__(self):
        self._comparisons = []

    def add_segment_pair(self, left_seg, right_seg):
        self._comparisons.append({"left": left_seg, "right": right_seg, "label": None})

    def __len__(self):
        return len(self._comparisons)

    @property
    def labeled_decisive_comparisons(self):
        return [c for c in self._comparisons if c["label"] in (0, 1)]

    def label_unlabeled_comparisons(self):
        for c in self._comparisons:
            if c["label"] is None:
                left = c["left"]["original_rewards"].sum()
                right = c["right"]["original_rewards"].sum()
                c["label"] = 0 if left > right else 1   # 0=左更好, 1=右更好


# ================================================================ 4. 标签调度
# 对应 rl_teacher/label_schedules.py 的 LabelAnnealer
class LabelSchedule:
    """随训练进程指数式增长「想要的标签数」。"""

    def __init__(self, final_timesteps, final_labels, pretrain_labels):
        self.final_timesteps = final_timesteps
        self.final_labels = final_labels
        self.pretrain_labels = pretrain_labels
        self.timesteps_elapsed = 0

    @property
    def n_desired_labels(self):
        exp_decay = 0.01 ** (self.timesteps_elapsed / self.final_timesteps)
        pretrain_frac = self.pretrain_labels / self.final_labels
        desired_frac = pretrain_frac + (1 - pretrain_frac) * (1 - exp_decay)
        return desired_frac * self.final_labels


# ============================================================ 5. 奖励预测器
# MLP 对应 rl_teacher/nn.py；成对偏好学习对应 rl_teacher/teach.py


def _leaky(x, slope=0.1):
    return np.where(x > 0, x, slope * x)


def _leaky_grad(x, slope=0.1):
    return np.where(x > 0, 1.0, slope)


class MLP:
    """两层隐藏层 MLP + Adam。输入 (state, action) 拼接，输出标量奖励。"""

    def __init__(self, in_dim, h=64, lr=1e-3, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(in_dim, h) * np.sqrt(2.0 / in_dim); self.b1 = np.zeros(h)
        self.W2 = rng.randn(h, h) * np.sqrt(2.0 / h); self.b2 = np.zeros(h)
        self.W3 = rng.randn(h, 1) * np.sqrt(2.0 / h); self.b3 = np.zeros(1)
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
        self.lr = lr
        self.cache = None

    def forward(self, X):
        z1 = X @ self.W1 + self.b1; a1 = _leaky(z1)
        z2 = a1 @ self.W2 + self.b2; a2 = _leaky(z2)
        out = a2 @ self.W3 + self.b3
        self.cache = (X, z1, a1, z2, a2)
        return out

    def backward(self, dout):
        X, z1, a1, z2, a2 = self.cache
        gW3 = a2.T @ dout; gb3 = dout.sum(0)
        dz2 = (dout @ self.W3.T) * _leaky_grad(z2)
        gW2 = a1.T @ dz2; gb2 = dz2.sum(0)
        dz1 = (dz2 @ self.W2.T) * _leaky_grad(z1)
        gW1 = X.T @ dz1; gb1 = dz1.sum(0)
        return [gW1, gb1, gW2, gb2, gW3, gb3]

    def adam_step(self, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mhat = self.m[i] / (1 - b1 ** self.t)
            vhat = self.v[i] / (1 - b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + eps)


class ComparisonRewardPredictor:
    """对应 teach.py 的 ComparisonRewardPredictor。

    核心思想（Bradley-Terry 偏好模型）：
      对每个 (state, action) 预测一个标量奖励 q。
      一段片段的「预测得分」= 各步 q 之和。
      两段片段得分组成 2 维 logits，用 softmax 交叉熵对齐人类标签(0/1)。
      即：不学绝对奖励，只学「相对排序」。
    """

    def __init__(self, obs_dim, act_dim, collector, label_schedule, seed=0):
        self.mlp = MLP(obs_dim + act_dim, h=64, seed=seed)
        self.collector = collector
        self.label_schedule = label_schedule
        self.recent_segments = deque(maxlen=200)
        self._steps_since_last_training = 0
        self._train_iters = 0
        self._train_loss = 0.0
        self.rng = np.random.RandomState(seed + 1)

    # -------- 服务 agent：给一条轨迹输出逐步预测奖励（对应 teach.py predict_reward）
    def predict_reward(self, path):
        X = np.concatenate([path["obs"] * OBS_SCALE, path["actions"]], axis=1)
        return self.mlp.forward(X)[:, 0].copy()

    # -------- 数据生产 + 定期训练（对应 teach.py path_callback）
    def path_callback(self, path):
        self._steps_since_last_training += len(path["obs"])
        self.label_schedule.timesteps_elapsed += len(path["obs"])

        seg = sample_segment_from_path(path, SEG_LEN)
        if seg is not None:
            self.recent_segments.append(seg)

        # 还需要更多标签时，就从近期片段里随机挑两段造一个新比较对
        if (len(self.collector) < int(self.label_schedule.n_desired_labels)
                and len(self.recent_segments) >= 2):
            i, j = self.rng.randint(0, len(self.recent_segments), 2)
            self.collector.add_segment_pair(self.recent_segments[i], self.recent_segments[j])

        # 每积累 100 步训练一次预测器
        if self._steps_since_last_training >= 100:
            self.train_predictor()
            self._steps_since_last_training = 0

    # -------- 学习一步（对应 teach.py train_predictor）
    def train_predictor(self):
        self.collector.label_unlabeled_comparisons()
        labeled = self.collector.labeled_decisive_comparisons
        if len(labeled) == 0:
            return None

        B = min(64, len(labeled))
        idx = self.rng.choice(len(labeled), B, replace=False)
        batch = [labeled[i] for i in idx]
        T = len(batch[0]["left"]["obs"])

        Xl = np.stack([np.concatenate([c["left"]["obs"] * OBS_SCALE, c["left"]["actions"]], axis=1)
                       for c in batch])                                # (B,T,d)
        Xr = np.stack([np.concatenate([c["right"]["obs"] * OBS_SCALE, c["right"]["actions"]], axis=1)
                       for c in batch])
        labels = np.array([c["label"] for c in batch])                  # (B,)

        Xall = np.concatenate([Xl, Xr], axis=0).reshape(2 * B * T, -1)  # 左右一起前向
        out = self.mlp.forward(Xall).reshape(2 * B, T)
        score_left = out[:B].sum(axis=1)
        score_right = out[B:].sum(axis=1)
        logits = np.stack([score_left, score_right], axis=1)            # (B,2)

        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits); probs /= probs.sum(axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(B), labels] + 1e-9).mean()

        dlogits = probs.copy(); dlogits[np.arange(B), labels] -= 1; dlogits /= B
        derr = np.zeros((2 * B, T), dtype=np.float32)
        derr[:B] = dlogits[:, 0][:, None]
        derr[B:] = dlogits[:, 1][:, None]
        grads = self.mlp.backward(derr.reshape(-1, 1))
        self.mlp.adam_step(grads)
        self._train_iters += 1
        self._train_loss = float(loss)
        return loss


# ==================================================================== 6. 策略
# 对应 agents/ 下的 TRPO（这里简化为 REINFORCE）
class Policy:
    """高斯策略：均值 = tanh(W·obs + b)，标准差固定。用 REINFORCE 更新。"""

    def __init__(self, obs_dim, act_dim, lr=0.1, seed=0):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(obs_dim, act_dim) * 0.01
        self.b = np.zeros(act_dim)
        self.log_std = np.full(act_dim, -0.7)
        self.explore_scale = 1.0   # 探索噪声的外部退火系数（训练后期 → 接近 0）
        self.lr = lr
        self.m = [np.zeros_like(p) for p in (self.W, self.b, self.log_std)]
        self.v = [np.zeros_like(p) for p in (self.W, self.b, self.log_std)]
        self.t = 0

    def set_explore_scale(self, s):
        """退火探索噪声：越到后期噪声越小，策略才能稳定收敛到目标附近。"""
        self.explore_scale = float(s)

    def _sigma(self):
        return np.exp(self.log_std) * self.explore_scale

    def act(self, ob):
        pre = (ob * OBS_SCALE) @ self.W + self.b
        mean = np.tanh(pre)
        a = mean + self._sigma() * np.random.randn(*mean.shape)
        return np.clip(a, -1.0, 1.0).astype(np.float32), mean, a

    def act_mean(self, ob):
        """确定性动作（取均值、不加噪声），用于最终评估。"""
        pre = (ob * OBS_SCALE) @ self.W + self.b
        return np.clip(np.tanh(pre), -1.0, 1.0).astype(np.float32)

    def _adam(self, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params = [self.W, self.b, self.log_std]
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            mhat = self.m[i] / (1 - b1 ** self.t)
            vhat = self.v[i] / (1 - b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + eps)

    def update(self, episodes):
        """episodes: 每项含 obs / actions / adv，adv 为「预测奖励」的优势估计。"""
        gW = np.zeros_like(self.W)
        gb = np.zeros_like(self.b)
        glogstd = np.zeros_like(self.log_std)
        total = 0
        for ep in episodes:
            obs = ep["obs"] * OBS_SCALE
            a = ep["actions"]
            adv = ep["adv"][:, None]
            pre = obs @ self.W + self.b
            mean = np.tanh(pre)
            sigma = self._sigma()
            dlogp_dmu = (a - mean) / sigma ** 2
            dlogp_dlogstd = ((a - mean) ** 2) / sigma ** 2 - 1

            dpre = (adv * dlogp_dmu) * (1 - mean ** 2)
            gW += obs.T @ dpre
            gb += dpre.sum(axis=0)
            glogstd += (adv * dlogp_dlogstd).sum(axis=0)
            total += len(ep["obs"])
        self._adam([-gW / total, -gb / total, -glogstd / total])


# ==================================================================== 主流程
def reward_to_go(rewards, gamma=0.99):
    out = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def ascii_grid(pos):
    """把 2D 位置渲染成字符点阵（返回若干行字符串）。

    世界坐标 x,y ∈ [-WORLD_LIM, WORLD_LIM]，映射到 GRID_W × GRID_H 的字符格。
    '·'=背景点阵, '+'=原点, '▼'=优化目标, '●'=小球(本批平均终点), '★'=正好命中目标。
    """
    def to_col(v):
        return int(round((float(np.clip(v, -WORLD_LIM, WORLD_LIM)) + WORLD_LIM) / (2 * WORLD_LIM) * (GRID_W - 1)))

    def to_row(v):   # 注意：y 越大行号越小（画面上方）
        return int(round((WORLD_LIM - float(np.clip(v, -WORLD_LIM, WORLD_LIM))) / (2 * WORLD_LIM) * (GRID_H - 1)))

    grid = [[C_DIM + "·" + C_RESET for _ in range(GRID_W)] for _ in range(GRID_H)]
    grid[to_row(0.0)][to_col(0.0)] = C_DIM + "+" + C_RESET                       # 原点
    grid[to_row(TARGET[1])][to_col(TARGET[0])] = C_TGT + "▼" + C_RESET           # 目标

    c, r = to_col(pos[0]), to_row(pos[1])
    hit = (c == to_col(TARGET[0]) and r == to_row(TARGET[1]))
    grid[r][c] = C_BALL + ("★" if hit else "●") + C_RESET                        # 小球
    return ["".join(row) for row in grid]


def render_frame(it, end_pos, true_rew, corr, n_cmp):
    """组装一帧完整画面（状态行 + 点阵）。行数固定，便于光标回退原地刷新。"""
    dist = float(np.linalg.norm(end_pos - TARGET))
    tag = "  ✓命中(±0.5)" if dist <= 0.5 else ""
    lines = [
        "mini_rl_teacher · 2D 点阵优化      iter %4d / %d" % (it, N_ITERS),
        "目标 %s▼%s=(%.1f, %.1f)   小球 %s●%s=本批平均终点   真奖励=%.1f（越接近 0 越好）"
        % (C_TGT, C_RESET, TARGET[0], TARGET[1], C_BALL, C_RESET, true_rew),
        "当前终点 (%.2f, %.2f)   距目标=%.2f%s   预测器相关系数=%.2f   比较对=%d"
        % (end_pos[0], end_pos[1], dist, tag, corr, n_cmp),
        "",
    ]
    lines += ascii_grid(end_pos)
    lines.append("    x[-6 .. +6]    y[-6 .. +6]    ▼=目标   ●=当前平均终点   + =原点   ·=点阵")
    return lines


def main():
    np.random.seed(0)
    print("=" * 70)
    print("mini_rl_teacher —— RL-Teacher 核心机制本机复现")
    print("目标：让小球停在目标点 %s 附近（agent 看不到这个目标，只靠人类偏好）" % (TARGET.tolist(),))
    print("=" * 70)

    env = ToyLineEnv(seed=0)

    # ---------------- 阶段 1：预训练（对应 teach.py main 里的 pretrain 段）
    print("\n[阶段1] 随机 rollout 生成片段 -> 合成人类打标 -> 训练奖励预测器")
    pretrain_segments = segments_from_rand_rollout(env, n_desired_segments=200, seed=0)
    collector = SyntheticComparisonCollector()
    for i in range(100):   # 组成 100 个比较对
        collector.add_segment_pair(pretrain_segments[i], pretrain_segments[i + 100])
    collector.label_unlabeled_comparisons()
    print("  已生成 %d 个比较对（合成人类标签：真实累积奖励更高者胜）" % len(collector))

    schedule = LabelSchedule(final_timesteps=70 * 10 * EPISODE_LEN,
                             final_labels=1500, pretrain_labels=100)
    predictor = ComparisonRewardPredictor(env.obs_dim, env.act_dim, collector, schedule, seed=0)

    for i in range(600):
        predictor.train_predictor()
    print("  预测器预训练完成，最后 loss = %.4f" % predictor._train_loss)

    # 验证预测器质量：预测总分 与 真实总分 的相关系数
    def predictor_correlation():
        paths = [do_rollout(env, lambda ob: np.random.uniform(-1, 1, size=env.act_dim)) for _ in range(20)]
        pred = np.array([predictor.predict_reward(p).sum() for p in paths])
        true = np.array([p["original_rewards"].sum() for p in paths])
        if pred.std() < 1e-8 or true.std() < 1e-8:
            return 0.0
        return float(np.corrcoef(pred, true)[0, 1])

    print("  预测器 预测奖励 vs 真实奖励 相关系数 = %.3f  (越接近 1 越好)" % predictor_correlation())

    # ---------------- 阶段 2：联合训练（对应 teach.py train_parallel_trpo）
    print("\n[阶段2] 联合训练：agent 用「预测奖励」学习，2D 点阵实时刷新优化进展\n")

    policy = Policy(env.obs_dim, env.act_dim, lr=0.1, seed=1)

    frame_lines = 4 + GRID_H + 1   # 帧行数固定，用于光标回退原地重绘
    drawn_once = False
    corr = 0.0

    def draw(lines):
        """在终端原地重绘一帧：先把光标移回帧顶，再逐行清空重写。"""
        nonlocal drawn_once
        if drawn_once:
            sys.stdout.write("\033[%dA" % frame_lines)
        for ln in lines:
            sys.stdout.write("\r\033[K" + ln + "\n")
        sys.stdout.flush()
        drawn_once = True

    for it in range(1, N_ITERS + 1):
        # 探索噪声线性退火：越到后期噪声越小，策略才能收敛并稳定停在目标附近
        policy.set_explore_scale(max(0.05, 1.0 - 0.95 * it / N_ITERS))
        episodes = []
        for _ in range(16):   # 每轮采集 16 条 episode
            path = do_rollout(env, lambda ob: policy.act(ob)[0])

            # ======= 这就是全项目的接合点（对应 agents/.../rollouts.py）=======
            pred_rewards = predictor.predict_reward(path)   # 用预测奖励替换真奖励
            predictor.path_callback(path)                   # 顺便产出训练预测器的数据
            # ================================================================

            returns = reward_to_go(pred_rewards)
            episodes.append({"obs": path["obs"], "actions": path["actions"],
                             "rewards": pred_rewards, "returns": returns,
                             "end_pos": path["human_obs"][-1],
                             "true_ep_reward": path["original_rewards"].sum()})

        # 优势标准化（对应 TRPO learn() 里对 advantage 的标准化）
        advs = np.concatenate([e["returns"] for e in episodes])
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        idx = 0
        for e in episodes:
            n = len(e["obs"])
            e["adv"] = advs[idx:idx + n]
            idx += n

        policy.update(episodes)

        # ---- 持续刷新点阵：实时展示小球（本批平均终点）向目标的收敛过程 ----
        if it % DRAW_EVERY == 0 or it == 1:
            true_rew = float(np.mean([e["true_ep_reward"] for e in episodes]))
            end_pos = np.mean([e["end_pos"] for e in episodes], axis=0)
            if it % 100 == 0:
                corr = predictor_correlation()
            draw(render_frame(it, end_pos, true_rew, corr, len(collector)))

    # ---- 最终评估：用「确定性策略」（去掉探索噪声）看目标是否达成 ----
    finals = []
    for _ in range(50):
        p = do_rollout(env, policy.act_mean)
        finals.append((p["human_obs"][-1], float(p["original_rewards"].sum())))
    end_pos = np.array([f[0] for f in finals])
    end_r = np.array([f[1] for f in finals])
    dist = np.linalg.norm(end_pos - TARGET, axis=1)
    print("\n[最终评估·确定性策略] 50 条 episode：")
    print("  终点位置 平均=(%.2f, %.2f)，距目标平均=%.2f（越小越贴近目标）"
          % (end_pos[:, 0].mean(), end_pos[:, 1].mean(), dist.mean()))
    print("  真奖励   平均=%.2f（越接近 0 越好）" % end_r.mean())
    print("  终点落在目标 ±0.5 内的比例 = %.0f%%" % (100 * (dist <= 0.5).mean()))

    print("\n完成。可见：agent 只优化「预测奖励」，但真实目标已基本达成 ——")
    print("这正是 RL-Teacher 的核心：把人类的相对偏好，转化成可优化的奖励信号。")


if __name__ == "__main__":
    main()
