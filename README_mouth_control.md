# Microduck 嘴部强化学习（mouth-throw）改动包

这个目录是 **Microduck 嘴部 RL 任务**（用嘴叼起一支笔往天上扔）的全部改动文件，
按它们对应在 `microduck_rl` 仓库里的**目标路径**排好序，方便你：

- 手动把这个目录推到你的 GitHub 仓库（独立小仓库或一个子目录都行）；
- 在另一台机器上把文件**按原路径放回** `microduck_rl` 目录里，直接使用。
- 附带的说明文件（就是本 README）解释每个文件放哪、怎么跑。

改动所依赖的只是 `microduck_rl` 这个仓库本身，不做任何别的改动、不换硬件。

---

## 1. 这个改动是什么

给 `microduck_rl` 增加一个新的训练任务：

| 项 | 内容 |
|---|---|
| 任务 ID | `Mjlab-MouthThrow-Flat-MicroDuck`（+ `-Rough-` 变体） |
| 目标行为 | 一段式：下蹲 > 嘴叼住笔 > 起身蓄力 > 甩头上扬并把笔往天上丢 > 归位站好 |
| 关键新东西 | **第 15 个可驱动关节 `mouth`**（真实机器人有口部电机 ID 34，原 sim 里是并进了 `jaw_soft` 的刚体，无法单独训练） |
| 嘴的建模 | 从 `jaw_soft` 里拆出鸟喙网格 `jaw`/`soft_mouth_top`/`jaw_soft` 和 `mouth_tip` 到新的 `mouth` 铰链 |
| 笔的建模 | 复用仓库已有一套「嘴里叼了个重量」的机制（payload force），不是真的加一个笔的碰撞体 |
| 特别注意 | 本任务 **有意** 改变 obs/action 维度（action 15 维、actor obs 约 63 维），只对这一个独立任务生效，**不影响**其它仍用 14 关节 / 61 维的旧策略 |

更完整的设计说明见：`docs/superpowers/specs/2026-mouth-throw-design.md`。

---

## 2. 文件清单（往 `microduck_rl` 里放）

### 2.1 新增文件（`ADD`）——直接拷贝进去即可

| 文件（相对 `microduck_rl` 的路径） | 作用 |
|---|---|
| `src/mjlab_microduck/robot/microduck/add_mouth.py` | 文本后处理脚本：从一个 Onshape 导出模型生成带 `mouth` 关节的模型。**只跑一次、产物入库**（惯例同 `add_backlash.py`）。 |
| `src/mjlab_microduck/robot/microduck/robot_groundcontact_mouth.xml` | **生成的** 带第 15 个 `mouth` 关节的 ground-contact 模型（15 个执行器）。已用 MuJoCo 校验。 |
| `src/mjlab_microduck/tasks/microduck_mouth_throw_env_cfg.py` | 新训练环境：`make_microduck_mouth_throw_env_cfg()` + `MicroduckMouthThrowRlCfg`。 |
| `tests/test_mouth_throw_cfg.py` | 配置不变式测试（CPU、不需要 GPU）：相位命令、reward 权值、payload、关节索引/网格布局。 |
| `scripts/_validate_mouth_model.py` | 校验辅助：加载 mouth 模型，确认 `mouth` 铰链独立驱动鸟喙、其它头部不动。 |
| `docs/superpowers/specs/2026-mouth-throw-design.md` | 设计说明 / 决策记录。 |

### 2.2 修改文件（`MODIFY`）——**覆盖** `microduck_rl` 里同名文件

| 文件（相对 `microduck_rl` 的路径） | 改了哪里 |
|---|---|
| `src/mjlab_microduck/robot/microduck_constants.py` | 新增 `MICRODUCK_MOUTH_XML`、`get_mouth_spec()`、`HOME_FRAME` 加 `mouth=0`、`MICRODUCK_MOUTH_ROBOT_CFG`。 |
| `src/mjlab_microduck/tasks/mdp.py` | 末尾新增 "Mouth Throw" 段：`mouth_open_during_throw`、`mouth_closed_during_hold`、`mouth_throw_upward_velocity`、`mouth_throw_peak_upward`、`mouth_throw_ground_proximity`、`apply_mouth_payload_release`、`reset_mouth_tip_vel_cache` 及相位门控辅助。 |
| `src/mjlab_microduck/tasks/__init__.py` | 新增 import + 两个 `register_mjlab_task`：`Mjlab-MouthThrow-Flat-` / `-Rough-`。 |

> **注意**：这 3 个是被修改的文件。如果你在别的机器上已经有别的改动，直接覆盖会把那些改动吞掉 —— 请用 git diff 或人工合并这一段的改动进去（改动点都很集中、有注释标记）。

---

## 3. 在另一台机器上怎么用

前置：另一台机器要有 `microduck_rl` 仓库 + `uv` + CUDA（训练用）。

```bash
# 1) 把本目录所有文件按 2 的路径放回 microduck_rl（新增的拷进去，修改的覆盖/合并）

# 2) 确认模型能加载（用系统已装好的 mujoco，不需要 GPU）
python -c "import mujoco; m=mujoco.MjModel.from_xml_path('src/mjlab_microduck/robot/microduck/robot_groundcontact_mouth.xml'); print(m.nu, m.njnt)"   # 期望 15 16
python scripts/_validate_mouth_model.py

# 3) 跑配置测试（CPU）
uv run --with pytest pytest tests/test_mouth_throw_cfg.py

# 4) 必须先跑的 smoke test（AGENTS.md 规定，跑长训练之前必跑）
uv run train Mjlab-MouthThrow-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5

# 5) 没问题再正式训练
uv run train Mjlab-MouthThrow-Flat-MicroDuck --env.scene.num-envs 4096
```

训练后照常 `scripts/export.py` 导出 ONNX 部署。**部署到真机是下一个阶段**（要把 runtime 的 obs/action 契约改成 15 维并重新导出），这个包里只改了训练侧、没动 Rust runtime。

---

## 4. 想重新生成 `mouth` 模型怎么跑

`robot_groundcontact_mouth.xml` 是用 `add_mouth.py` 从 base ground-contact 模型生成的。
一般不需要重新生成（产物已入库），万一要回看/重跑：

```bash
python src/mjlab_microduck/robot/microduck/add_mouth.py \
  src/mjlab_microduck/robot/microduck/robot_groundcontact.xml \
  --out src/mjlab_microduck/robot/microduck/robot_groundcontact_mouth.xml
```

---

## 5. 这份代码目前的验证状态（诚实说明）

- ✅ 所有改到的 Python 文件 `py_compile` 通过；
- ✅ `mouth` 模型在 MuJoCo 3.2.0 实测通过：16 关节 / 15 执行器，驱动 `mouth` 只带动鸟喙、头部其它部分不动；
- ❌ **未在「有 uv + GPU」的机器上跑过 smoke test / pytest**（本机只有 CPU+系统 mujoco，装不了 mjlab/torch）。拿到有 `uv`+GPU 的机器上后，**务必先跑第 3 节的第 3、4 步**。

---

## 6. 主要文件路径速查（在 `microduck_rl` 里）

```
src/mjlab_microduck/robot/microduck/add_mouth.py
src/mjlab_microduck/robot/microduck/robot_groundcontact_mouth.xml
src/mjlab_microduck/robot/microduck_constants.py
src/mjlab_microduck/tasks/mdp.py
src/mjlab_microduck/tasks/microduck_mouth_throw_env_cfg.py
src/mjlab_microduck/tasks/__init__.py
tests/test_mouth_throw_cfg.py
docs/superpowers/specs/2026-mouth-throw-design.md
scripts/_validate_mouth_model.py
```
