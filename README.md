# Mini RL Teacher

**English** | [简体中文](README.zh-CN.md)

A NumPy-only introduction to reinforcement learning from preferences, with a 2D control task, a learned reward model, and terminal visualization.

Start with [mini_rl_teacher.py](mini_rl_teacher.py). This standalone Python file demonstrates the full loop: **compare behavior segments → learn a reward model → optimize a policy using predicted rewards → collect new comparisons**.

The mini version uses a 2D ball environment, synthetic preferences, and REINFORCE. Its neural network, backpropagation, and Adam optimizer are implemented in NumPy. The repository also retains the original RL-Teacher code for MuJoCo, TRPO/PPO, and human feedback collection for comparison. The original project implements Christiano et al.'s 2017 paper, [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741). The mini version explains the core mechanism; it does not reproduce the paper's experimental results.

## Demo

[![Terminal replay of the default training run](docs/demo/preview.en.gif)](https://github.com/InfronAI/mini-rl-teacher/raw/refs/heads/main/docs/demo/mini-rl-teacher.en.mp4)

[Watch / download the MP4](https://github.com/InfronAI/mini-rl-teacher/raw/refs/heads/main/docs/demo/mini-rl-teacher.en.mp4) · [English transcript](docs/demo/terminal.en.txt) · [Original output](docs/demo/terminal.txt) · [Run metadata](docs/demo/run.json)

The English video translates the on-screen text of the original Chinese output; all measurements and grid positions are preserved. [中文版视频](https://github.com/InfronAI/mini-rl-teacher/raw/refs/heads/main/docs/demo/mini-rl-teacher.zh-CN.mp4) replays the same run with Chinese text.

The video comes from an actual run of the unchanged `mini_rl_teacher.py` with its default settings. It includes pretraining, all 101 printed training frames, and final evaluation. Terminal output is reformatted into a roughly 43-second replay, with each training frame held for 0.32 seconds. Playback duration does not represent training time.

Using Python 3.11 and NumPy 2.4.6, this run completed 1,000 training iterations. Deterministic evaluation over 50 episodes reported a mean endpoint of `(6.00, 4.78)`, mean endpoint distance of `4.10`, mean true return of `-91.79`, and a success rate of `0%`. **This default run did not converge.** The video preserves the actual result and illustrates why the script's fixed completion message is not a success check. Other environments and random sequences may produce different outcomes.

See [docs/demo/README.md](docs/demo/README.md) for recording and reproduction instructions.

## Quick start

From the repository root, create a virtual environment, install NumPy, and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy
python mini_rl_teacher.py
```

In Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`; the other commands are the same.

The mini version requires only Python 3 and NumPy. It does not require the root `setup.py`, MuJoCo, TensorFlow, a GPU, a database, or cloud storage. Running `pip install -e .` in the root installs dependencies for the historical implementation instead.

Execution proceeds through reward-model pretraining, joint policy/reward-model training, and final evaluation. The terminal displays the target and the ball's mean endpoint. Use a terminal that supports ANSI cursor control and Unicode. The script currently has no command-line options: training settings are defined in the file, and models or training results are not saved automatically.

## Task: learn to approach a target from preferences

Despite its name, `ToyLineEnv` is a two-dimensional environment:

| Property | Implementation |
| --- | --- |
| State | `[x, y, vx, vy]`: position and velocity, 4 dimensions |
| Action | `[fx, fy]`: force along each axis, clipped to `[-1, 1]` |
| Initial state | Random position in `[-1, 1]²`, zero velocity |
| World boundaries | Both coordinates stay within `[-6, 6]` |
| Episode length | Fixed at 30 steps |
| Target | Fixed position `[3, 2]` |
| True reward | Negative Euclidean distance to the target after taking the action |

Each step updates velocity, then position:

```python
vel = clip(0.9 * vel + action, -3, 3)
pos = clip(pos + 0.1 * vel, -6, 6)
true_reward = -norm(pos - TARGET)
```

The policy observes the ball's position and velocity. Target coordinates are not inputs to either the policy or the reward network; information about the target reaches the learning process indirectly through preference labels.

The example defines a true reward to simulate human judgment: the segment with the higher true cumulative reward wins. **True rewards generate labels and evaluate performance; policy updates use the reward model's predictions.** The mini version does not include an interactive human-labeling workflow. It corresponds to the original project's `synth` mode.

## Training loop

```mermaid
flowchart TD
    A[Collect trajectories with uniform random actions] --> B[Sample 200 segments]
    B --> C[Create 100 synthetic preference pairs]
    C --> D[Pretrain reward model for 600 updates]
    D --> E[Initialize Gaussian policy]
    E --> F[Collect 16 episodes per iteration]
    F --> G[Predict step rewards and compute returns]
    G --> H[Update policy with REINFORCE]
    H --> F
    F --> I[Sample new comparisons from recent trajectories]
    I --> J[Generate labels and update reward model]
    J --> G
```

### 1. Generate initial preference data

`segments_from_rand_rollout()` explores with uniform random actions, without using the state to choose an action. It samples one 15-step segment from each trajectory, collecting 200 segments by default. Pairing the first half with the second half produces 100 comparisons.

`SyntheticComparisonCollector` compares `original_rewards.sum()` on each side. The label is `0` if the left score is strictly higher, and `1` otherwise. There is no tie label.

Each trajectory contains:

| Field | Meaning | Default full-episode shape |
| --- | --- | --- |
| `obs` | Position and velocity before the action | `(30, 4)` |
| `actions` | Actions passed to the environment | `(30, 2)` |
| `original_rewards` | True rewards after the action | `(30,)` |
| `human_obs` | Positions after the action, used for visualization and evaluation | `(30, 2)` |

### 2. Train a reward model from comparisons

`ComparisonRewardPredictor` concatenates scaled states and actions and passes them to an `MLP`:

```text
4 state dimensions + 2 action dimensions
→ Dense(64) + LeakyReLU
→ Dense(64) + LeakyReLU
→ Dense(1): predicted reward for each step
```

The same network scores both segments. Each segment's score is the sum of its predicted step rewards:

$$
S_L = \sum_t \hat r_\theta(s_t^L,a_t^L), \qquad
S_R = \sum_t \hat r_\theta(s_t^R,a_t^R)
$$

The predicted probability of preferring the left segment is:

$$
P(L) = \frac{e^{S_L}}{e^{S_L}+e^{S_R}}
$$

Cross-entropy against the left/right labels drives hand-written backpropagation and Adam updates. Each update samples at most 64 comparisons; pretraining uses 600 updates by default. The objective is to match preferences, not to regress the numerical true rewards. Absolute predicted rewards should not be compared directly with true rewards. The `q` mentioned in code comments denotes an immediate reward prediction, not a Q-function.

### 3. Initialize and train the policy

`Policy` is created only after reward pretraining. It has no hidden layers and loads no pretrained weights:

```python
W = rng.randn(4, 2) * 0.01
b = np.zeros(2)
log_std = np.full(2, -0.7)
```

The action mean and sampling rule are:

```python
mean = tanh((obs * OBS_SCALE) @ W + b)
sigma = exp(log_std) * explore_scale
action = clip(mean + sigma * noise, -1, 1)
```

`noise` is standard Gaussian noise. The initial mean is close to zero, and the standard deviation before the exploration multiplier is approximately `0.497`. All of `W`, `b`, and `log_std` are trained. Although the class docstring describes a fixed standard deviation, the implementation updates it.

Uniform random exploration for the initial dataset and this Gaussian policy are separate stages. The former does not supply weights to the latter.

The integration point in `main()` is:

```python
path = do_rollout(env, lambda ob: policy.act(ob)[0])
pred_rewards = predictor.predict_reward(path)
predictor.path_callback(path)
returns = reward_to_go(pred_rewards)
```

Each iteration collects 16 episodes. Predicted rewards are converted to reward-to-go with a discount factor of `0.99`, then returns are standardized across the batch and passed to `Policy.update()` as `adv`. There is no learned state-value function. This is a simplified REINFORCE update, not TRPO or PPO.

`path_callback()` maintains the latest 200 candidate segments, adds comparisons according to the label schedule, and periodically trains the reward model. Historical comparisons remain in the collector. Updates alternate sequentially; episodes within one policy batch can receive rewards from different versions of the reward model.

### 4. Evaluate the policy

After training, `act_mean()` removes exploration noise for evaluation from 50 random starting positions. The script reports the mean endpoint, mean endpoint distance, mean true return, and the fraction of endpoints within distance `0.5` of the target.

This measures approaching the target within a fixed horizon. The reward does not directly penalize velocity, and endpoint success does not verify that the ball has stopped. To assess stable hovering near the target, also inspect distance and velocity over the final part of each trajectory.

## Reading the terminal output

| Display | Meaning and limitations |
| --- | --- |
| `▼` | Target position |
| `●` | Mean endpoint of the current 16 training episodes, not a live position in one episode |
| `★` | Mean endpoint and target map to the same character cell, not necessarily the same coordinates |
| True reward | Mean true cumulative reward for the batch; closer to zero is better |
| Distance / hit | Distance from the mean endpoint to the target; errors in opposite directions can cancel |
| Predictor correlation | Correlation between predicted and true total rewards on 20 uniform-random-policy trajectories |
| Comparisons | Number of comparisons created, not the number of successful episodes |

The script prints a correlation value after pretraining. In the joint-training display, this value starts at `0` and is recomputed every 100 iterations when the drawing condition is also met; otherwise, it retains the previous value. It does not fully measure reward-model accuracy on states visited by the current policy.

The final statement that the true goal has been achieved is fixed text, **not an automatically verified conclusion**. Use measured final distance and success rate, rather than assuming every run converges.

## Configuration and convergence

There are no `--seed` or `--iters` options. Edit the corresponding constants or settings in `main()`.

| Setting | Default | Location |
| --- | --- | --- |
| Target `TARGET` | `[3, 2]` | Top-level constants |
| Episode length `EPISODE_LEN` | `30` | Top-level constants |
| Segment length `SEG_LEN` | `1.5 × 10 = 15` | Top-level constants |
| Joint iterations `N_ITERS` | `1000` | Top-level constants |
| Drawing interval `DRAW_EVERY` | `10`, plus iteration 1 | Top-level constants |
| Initial segments / comparisons / reward updates | `200 / 100 / 600` | `main()` |
| Episodes per iteration | `16` | `main()` |
| Label target / schedule horizon | `1500 / 21000` steps | `LabelSchedule` construction |
| Reward learning rate / maximum batch size | `0.001 / 64` | `MLP` / `train_predictor()` |
| Policy learning rate | `0.1` | `Policy` construction in `main()` |
| Exploration multiplier | Linearly decreases from near `1` to `0.05` | Joint-training loop |

Default joint training takes `1000 × 16 × 30 = 480000` environment steps, excluding pretraining and evaluation rollouts. The label schedule still uses `70 × 10 × 30 = 21000` steps, so the desired label count approaches its limit relatively early. Actual comparison creation is also limited to at most one new pair per episode. Reducing `N_ITERS` accelerates exploration annealing as well as shortening training. Keep episode length at least as large as segment length; otherwise, initial sampling cannot collect valid segments.

Both the initial policy and the initial dataset affect convergence. The policy determines early exploration, while the dataset determines which behaviors the reward model first learns to distinguish. Subsequent sampling feeds these effects back into training. Approaching a target and slowing down near it require different evidence; initial random trajectories need not cover both.

To study initialization, hold the initial dataset and reward model fixed while varying the policy seed, then hold the policy fixed while varying the data. Compare final distances and success rates across runs, rather than relying only on the mean-endpoint visualization.

The current global NumPy, environment, and reward-model seeds are `0`; the policy initialization seed is `1`. The `seed` argument to `segments_from_rand_rollout()` is unused inside that function, which draws from the global random state. Correlation evaluation also shares the training environment and global random source, so changing the number of evaluation calls changes subsequent random sequences.

## Current limitations

These describe existing behavior and have not been fixed in the implementation:

- **Action clipping and policy gradients are inconsistent.** Actions are sampled from a Gaussian and clipped before execution, but updates insert the clipped actions into the ordinary Gaussian log-probability gradient. This introduces bias when clipping occurs. `act()` already returns the unclipped action, but the main loop stores only the clipped one.
- **Exploration noise need not decrease monotonically.** The external multiplier anneals, while the optimizer also learns `log_std`; both determine the actual standard deviation.
- **Comparisons can repeat a segment.** Online pairing can select the same segment twice, while equal true scores receive a right-wins label. Such comparisons provide no useful directional signal.
- **Reward-update frequency is approximate.** The callback updates after accumulating at least 100 steps, then resets the counter. With 30-step episodes, the actual interval is 120 steps.
- **Generalization and training stability are limited.** The implementation uses one reward network, random pairing, and fixed input scaling. It has no separate preference-validation set, value network, trust-region constraint, or clipped policy objective.
- **No persistence or automated acceptance checks.** Models, optimizer state, and comparisons remain in memory. There are no automated training tests for the mini version or convergence guarantees.

## Reading alongside the original project

```text
mini_rl_teacher.py           # Main entry point: NumPy teaching implementation
rl_teacher/
  teach.py                  # Original training entry point and reward predictor
  nn.py                     # Keras reward network
  envs.py                   # MuJoCo wrappers, reward adjustments, state replay
  segment_sampling.py       # Random rollouts and segment sampling
  comparison_collectors.py  # Synthetic/human labels and video tasks
  label_schedules.py        # Label-budget scheduling
  video.py                  # Video encoding and GCS uploads
  summaries.py              # TensorBoard metrics
agents/
  parallel-trpo/            # Parallel TRPO
  pposgd-mpi/               # PPO with MPI support
  simple-trpo/              # Separate experimental TRPO implementation
human-feedback-api/         # Django comparisons, labeling pages, DB migrations
```

| Mini version | Original module | Main difference |
| --- | --- | --- |
| `ToyLineEnv` | [envs.py](rl_teacher/envs.py) | A 2D ball replaces MuJoCo; position arrays replace replayable physics states |
| `do_rollout`, `sample_segment_from_path` | [segment_sampling.py](rl_teacher/segment_sampling.py) | Single-process NumPy sampling |
| `SyntheticComparisonCollector` | [comparison_collectors.py](rl_teacher/comparison_collectors.py) | Synthetic left/right preferences only; no labeling UI or video uploads |
| `LabelSchedule` | [label_schedules.py](rl_teacher/label_schedules.py) | Retains the step-based label schedule |
| `MLP`, `ComparisonRewardPredictor` | [nn.py](rl_teacher/nn.py), [teach.py](rl_teacher/teach.py) | Hand-written forward/backward passes; no TensorFlow/Keras or Dropout |
| `Policy` | [parallel-trpo](agents/parallel-trpo), [pposgd-mpi](agents/pposgd-mpi) | A linear-plus-tanh Gaussian mean and REINFORCE replace TRPO/PPO |
| `ascii_grid`, `render_frame` | [video.py](rl_teacher/video.py), [summaries.py](rl_teacher/summaries.py) | Terminal mean-endpoint display replaces video and TensorBoard |

Start with `main()` in the mini file, then follow `do_rollout()` → `train_predictor()` → `Policy.update()` to understand the data and gradients. Finally, compare the original project's reward-replacement interface.

### Historical runtime

The original entry point, `rl_teacher/teach.py`, provides three modes: `rl`, `synth`, and `human`. The `rl` baseline uses environment rewards directly; `synth` uses synthetic preferences; `human` uses human feedback. Each starts its own training run. The `rl` mode does not automatically supply a pretrained policy to the other modes.

The original code targets Python 3.5, TensorFlow 1.2, Gym 0.9, and mujoco-py 0.5 and uses legacy interfaces. The mini version's installation steps do not prepare that environment. After setting up a compatible historical environment, example entry points are:

```bash
# True-reward baseline
python rl_teacher/teach.py -p rl -e ShortHopper-v1 -n base-rl

# Synthetic preferences
python rl_teacher/teach.py -p synth -l 1400 -e ShortHopper-v1 -n syn-1400

# Human preferences: requires a shared database, labeling UI, rendering, and GCS
python rl_teacher/teach.py -p human --pretrain_labels 175 -e Reacher-v1 -n human-175
```

Human feedback is read and written directly through Django's `Comparison` model. The training process and web app must access the same database. Videos are uploaded asynchronously to the GCS location specified by `RL_TEACHER_GCS_BUCKET`, and the browser plays their media URLs. The training process does not call a labeling service over HTTP. See [human-feedback-api/README.md](human-feedback-api/README.md) for the web app entry point.

The original environment wrappers may modify rewards or termination conditions. Logged `original_rewards` therefore refer to the wrapped environment's rewards. The original code and tests are retained as historical research code for comparison, not as evidence of a validated training run in the current environment.

## Attribution and license

This repository retains the original RL-Teacher implementation and adds a mini example explaining its preference-learning mechanism. Thanks to Paul Christiano and Dario Amodei for the original system's design and support, Max Harms for parallel TRPO integration and training contributions, Catherine Olsson for improvements to the original documentation and usability, and Kevin Frans for the parallel TRPO implementation.

Related extension: [rl-teacher-atari](https://github.com/machine-intelligence/rl-teacher-atari). See [LICENSE](LICENSE) for licensing.
