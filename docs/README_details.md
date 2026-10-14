# ROS 2 Policy Inference — full reference

> The short guide is [../README.md](../README.md). This page keeps every detail.

Run IL / VLA policies with camera and robot observations.
Publish predicted end-effector actions to `/il/grasp/action`.

## 🏗️ Architecture

![ROS 2 Policy Inference Docker architecture](../architecture.svg)

Each policy's Python dependencies are isolated in its own Docker image, built from a
dedicated Dockerfile (see [Publish a Docker image](#publish-a-docker-image)). Every image
mounts the same source code and checkpoints at container start.
## 🔑 Setup

Get a GitLab token (scope: `read_api`) from your GitLab profile → Access Tokens, then save it once — scripts pick it up automatically:

```bash
printf '%s' 'glpat-...' > ~/.gitlab_token && chmod 600 ~/.gitlab_token
```

## 🚀 Run

> 💡 Start with `dummy` to check inputs and preprocessing without a checkpoint.

### 🔀 Choosing a policy

Every way of running the node takes **one policy name**; nothing else needs editing to switch.

| Policy | Status | Image | Rate | Checkpoint needed |
|:--|:--|:--|:--|:--|
| `dummy` | no-op placeholder | `lerobot_ros2` | 10 Hz | no |
| `act` | LeRobot ACT (fan grasp), working | `lerobot_ros2` | 30 Hz | `checkpoints/act/checkpoint_act_375` |
| `diffusion` | LeRobot Diffusion Policy (fan grasp), working | `lerobot_ros2` | 30 Hz | `checkpoints/diffusion/checkpoint_diffusion_375` |
| `diffusion_rtc` | same model + guided Real-Time Chunking, working | `lerobot_ros2` | 30 Hz | same as `diffusion` |
| `depth_inference` | Diffusion Policy trained on wrist RGB **+ depth**, not yet tried on the robot | `lerobot_ros2` | 30 Hz | `checkpoints/diffusion_depth/checkpoint_diffusion_375` |
| `depth_inference_rtc` | same depth model + guided Real-Time Chunking | `lerobot_ros2` | 30 Hz | same as `depth_inference` |
| `depth_act` | ACT trained on wrist RGB **+ depth**, not yet tried on the robot | `lerobot_ros2` | 30 Hz | `checkpoints/act_depth/checkpoint_act_375` |
| `liteflow` | InternVL3-1B + FLOWER, working | `liteflow_ros2` | 10 Hz | `checkpoints/liteflow/...` (auto-fetched) |
| `pi` | not implemented | `lerobot_ros2` | 10 Hz | — |

Where the name goes:

| How you run it | Where to put the policy name |
|:--|:--|
| Docker, launch directly | `./run_docker.sh --policy-name <policy>` |
| Docker, shell first | `./run_docker.sh --policy-name <policy> bash`, then `ros2 launch ... policy_name:=<policy>` inside |
| Local (no Docker) | `ros2 launch ros2_policy_inference inference.launch.py policy_name:=<policy>` |

What follows from the name automatically:

- **Image**: `run_docker.sh` picks `liteflow_ros2` for `liteflow` and `lerobot_ros2` for everything
  else. Use the same `--policy-name` for the container as in the `ros2 launch` line inside it, or
  `liteflow` will be missing its dependencies.
- **Checkpoint and inference settings**: read from `policies/<policy>/config.yaml`
  (e.g. ACT's temporal ensembling, Diffusion's DDIM steps); `diffusion_rtc` shares
  `policies/diffusion/config.yaml`.
- **Rate**: `act`, `diffusion` and `diffusion_rtc` run at 30 Hz, the rate their training data was recorded at,
  overriding `inference_rate_hz` in `config/params.yaml`. The node logs
  `policy '<policy>' requires 30.0 Hz` when this happens.
- **`depth_inference`**: a second diffusion checkpoint whose input also includes the wrist depth
  image (`/camera_L/depth/aligned_depth_to_color`, already subscribed). `policies/depth_inference/`
  encodes it as in training (0 = no reading, 255 at 0 m ... 1 at 1 m, set in its `config.yaml`) and
  otherwise reuses the `diffusion` backend, so `diffusion` and `diffusion_rtc` are unchanged.
- **`diffusion` vs `diffusion_rtc`**: same checkpoint and settings. `diffusion` is lerobot's
  synchronous chunking (one ~60 ms pause every 8 actions). `diffusion_rtc` adds guided Real-Time
  Chunking at inference: the next chunk is computed in the background and joined smoothly to the
  current one, so the node never pauses. Its startup log says `RTC (guided, ...)` and it logs an
  `RTC: ... stalls 0` summary every 100 chunks; tune it in the `rtc:` block of
  `policies/diffusion/config.yaml`.

Without a name, `ros2 launch` and `config/params.yaml` default to `dummy`.

**Switching**: stop the running node first (Ctrl+C, or `docker stop <container>`; list them with
`docker ps`), then start the new one. All policies publish on the same `/il/grasp/action`, so two
nodes running at once interleave their actions.

### 🐳 Docker

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
./run_docker.sh --policy-name act          # or: depth_act, diffusion, diffusion_rtc, depth_inference, depth_inference_rtc, liteflow, dummy
./run_docker.sh --policy-name diffusion_rtc --checkpoint diffusion_50k   # another checkpoint of that policy
./run_docker.sh --policy-name act bash     # shell in the container instead of launching
```

From that shell, build and launch by hand. The repo is mounted at `/workspace/src/ros2_policy_inference`,
so code edits on the host are picked up by the rebuild:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ros2_policy_inference
source install/setup.bash
ros2 launch ros2_policy_inference inference.launch.py policy_name:=act
```

`run_docker.sh` pulls the image from the GitLab Package Registry if it isn't present locally. Off the
lab network, build it instead (only public sources are needed); `run_docker.sh` then uses the local one:

```bash
docker compose -f docker/docker-compose.yml build policy-inference            # lerobot_ros2
docker compose -f docker/docker-compose.yml build policy-inference-liteflow   # liteflow_ros2
```

### 📦 Checkpoints

Checkpoints live in `checkpoints/` (git-ignored), mounted into the container at `/checkpoints`.
Set `CHECKPOINT_DIR` in the environment or a repo-root `.env` to use another folder.

| Policy | Path | How to get it |
|:--|:--|:--|
| `liteflow` | `checkpoints/liteflow/{step_20000_ema,hf_cache}` | fetched automatically by `run_docker.sh` (needs GitLab access) |
| `diffusion` | `checkpoints/diffusion/checkpoint_diffusion_375` (or `_50`) | `Robot_Friday/checkpoints/checkpoint_diffusion/{375000,050000}/pretrained_model/`; 375 is the former `fan_grasp_diffusion_v1`. Pick one with `checkpoint_dir` in `policies/diffusion/config.yaml` |
| `act` | `checkpoints/act/checkpoint_act_375` | copy `Robot_Friday/checkpoints/checkpoint_act/375000/pretrained_model/` |
| `depth_inference` | `checkpoints/diffusion_depth/checkpoint_diffusion_375` (or `_50`) | `Robot_Friday/checkpoints/checkpoint_diffusion_depth/{375000,050000}/pretrained_model/`, plus `Robot_Friday/data/lerobot_data_depth/meta/depth_encoding.json` (checked at startup) |
| `depth_act` | `checkpoints/act_depth/checkpoint_act_375` | `Robot_Friday/checkpoints/checkpoint_act_depth/375000/pretrained_model/`, plus `Robot_Friday/data/lerobot_data_depth/meta/depth_encoding.json` (checked at startup) |

Each policy's checkpoint path and inference knobs are in `policies/<name>/config.yaml`.

### 💻 Local

> ⚠️ Only for a machine with **ROS 2 Jazzy installed natively** (`/opt/ros/jazzy` exists). Without it
> these commands fail with `/opt/ros/jazzy/setup.bash: No such file or directory` and
> `ros2: command not found` — use [Docker](#-docker) instead: `./run_docker.sh --policy-name <policy>`
> already runs all four steps below inside the container. Activating a conda env (e.g. `lerobot312`)
> does not provide ROS.

Requires ROS 2 Jazzy and the policy's dependencies (for `act`/`diffusion`: torch + lerobot 0.6.2, see
`docker/Dockerfile`).

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ros2_policy_inference
source install/setup.bash
ros2 launch ros2_policy_inference inference.launch.py policy_name:=act   # or: depth_act, diffusion, diffusion_rtc, depth_inference, depth_inference_rtc, liteflow, dummy
ros2 launch ros2_policy_inference inference.launch.py policy_name:=diffusion_rtc checkpoint:=diffusion_50k
```

`checkpoint_dir` in `policies/<policy>/config.yaml` is a container path (`/checkpoints/...`). Outside
Docker, change it to the checkpoint's path on your machine, e.g.
`<repo>/checkpoints/act/checkpoint_act_375`.

### 🎞️ Test with a recorded bag

Feeds recorded robot topics to the node, so a policy can be checked without the robot.
The bag only replays inputs; a policy node must be running to turn them into actions.
All commands from the repo root:

```bash
# 1. the policy node - SKIP if one is already running (see "Docker" above; `docker ps` lists it).
#    Starting a second one makes two nodes publish on /il/grasp/action.
./run_docker.sh --policy-name act

# 2. play the bag on a loop, only the topics the node subscribes to
docker run --rm -it --name bag_player --network host --entrypoint bash \
  -v "$PWD/bags/fan_insertion_2026-09-22:/bag:ro" lerobot_ros2:latest -c \
  'source /opt/ros/jazzy/setup.bash && ros2 bag play /bag --loop --topics \
     /camera_L/color/image_raw/compressed /camera_L/depth/aligned_depth_to_color \
     /left_arm/joint_states /left_arm/tcp_pose_world /left_ft/wrench_compensated'

# 3. watch the actions (or just read the node's terminal, which logs every action)
docker run --rm -it --network host --entrypoint bash lerobot_ros2:latest -c \
  'source /opt/ros/jazzy/setup.bash && ros2 topic echo /il/grasp/action'
```

Bags live in `bags/` (git-ignored). Until the bag starts, the node's terminal repeats
`waiting for all observation topics...` / `no publisher found for ...` — that is expected, not an
error; once it plays, the node logs `action: [...]` at the policy's rate. `fan_insertion_2026-09-22`
(17 s) was recorded during fan *insertion*, so it checks the pipeline (topics → preprocessing →
model → action), not grasp quality; `--loop` also jumps the pose back every 17 s.

### 📡 Check output

With ROS 2 Jazzy installed natively:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo /il/grasp/action
```

> ⚠️ `ros2` needs ROS 2 on the machine. Without it (`ros2: command not found`), run the same command
> in a throwaway container from the policy image — it uses host networking, so it sees the node's
> topic:
>
> ```bash
> docker run --rm -it --network host --entrypoint bash lerobot_ros2:latest -c \
>   'source /opt/ros/jazzy/setup.bash && ros2 topic echo /il/grasp/action'
> ```
>
> Or open a shell in the running policy container (`docker ps` lists it) and run it there:
> `docker exec -it <container> bash`, then `source /opt/ros/jazzy/setup.bash && ros2 topic echo /il/grasp/action`.
> The node's own terminal also logs every action as `action: [...]`.

## 🔌 ROS Interface

### Inputs

| Topic | Message type | Preprocessed observation |
|:--|:--|:--|
| `/camera_L/color/image_raw/compressed` | `CompressedImage` | RGB: `256 × 256 × 3` |
| `/camera_L/depth/aligned_depth_to_color` | `Image` | Depth: `256 × 256`, nearest-neighbor resize |
| `/left_arm/joint_states` | `JointState` | Position: `(6,)`; velocity: `(6,)` |
| `/left_arm/tcp_pose_world` | `PoseStamped` | EE-offset position: `(3,)`; quaternion: `(4,)` |
| `/left_ft/wrench_compensated` | `WrenchStamped` | `[fx, fy, fz, tx, ty, tz]` |

### Output

| Topic | Message type | Format |
|:--|:--|:--|
| `/il/grasp/action` | `Float64MultiArray` | `[dx, dy, dz, drx, dry, drz, gripper]` |

For `act`, `diffusion` and `diffusion_rtc`: position delta in metres and rotation delta as an axis-angle in radians, both in the
world frame (apply the rotation by left multiplication, `R_next = R(drot) · R_now`), one action per
1/30 s; gripper `1` = open, `0` = closed, threshold at `0.5`. Nothing in this repo subscribes to the
topic yet — the consumer that turns actions into arm motion is still to be written.

## 🤖 Real robot

Checklist for running the fan-grasp policies (`act`, `diffusion_rtc`, `diffusion`) on the robot.
They were trained on the RS-M90E7A left arm with its wrist camera (`camera_L`) in the fan-grasp
scene; on a different arm, camera mount or scene they may not transfer even if everything below is
wired correctly. None of them has run on hardware yet.

### 1. Weights

Not in git (see [Checkpoints](#-checkpoints)); copy them in with their checksum file and verify:

```text
checkpoints/act/checkpoint_act_375/                # 197 MB, 7 files
checkpoints/diffusion/checkpoint_diffusion_375/    # 1 GB, 7 files (diffusion and diffusion_rtc)
checkpoints/SHA256SUMS
```

```bash
cd checkpoints && sha256sum -c SHA256SUMS      # all 14 files: OK
```

### 2. Docker image

Rebuild it once. The Dockerfile pins lerobot 0.6.2 and torch 2.11 / CUDA 13, which these checkpoints
need (older lerobot rejects their `config.json`), and `run_docker.sh` never replaces an existing
local `lerobot_ros2` image:

```bash
docker compose -f docker/docker-compose.yml build policy-inference
```

Needs an NVIDIA driver ≥ 580 (`nvidia-smi`). GPU memory: `act` ~0.3 GB, `diffusion`/`diffusion_rtc`
~1.2 GB (measured on an RTX 4070 Ti SUPER).

### 3. Inputs the robot must publish

The node waits until **all five** [input topics](#inputs) have arrived before its first action, so
depth, TCP pose and F/T must be published even though `act`/`diffusion` only use the image, the joints
and the gripper.

- **Camera**: the wrist camera as JPEG `CompressedImage`, at **≥ 30 fps** (subscribed BEST_EFFORT).
- **Joints**: exactly **6 arm positions in radians, in the training order** (`joint_0`…`joint_5`).
  They are read in published order (`msg.position`), not matched by name — a different order or an
  extra gripper joint breaks the state silently or fails the 7-D state check.
- **ROS**: same `ROS_DOMAIN_ID` as the robot (default 0; set it in a repo-root `.env`). The container
  uses host networking and CycloneDDS.

### 4. Start pose

Start inside the training range, ideally where the demonstrations started; far outside it the
policies extrapolate. Joint ranges in the training data (radians):

| `joint_0` | `joint_1` | `joint_2` | `joint_3` | `joint_4` | `joint_5` |
|:--|:--|:--|:--|:--|:--|
| −0.03 … 0.12 | −0.48 … 0.60 | −1.57 … 0.46 | 1.76 … 3.85 | 3.09 … 3.31 | −1.66 … −1.52 |

### 5. Applying the actions

Nothing in this repo moves the arm: the robot-side controller must apply every message on
`/il/grasp/action` (7 values, 30 Hz) the way the training data was recorded:

- `[0:3]` position delta in metres, **world frame**, at the **TCP = Link_6 + 0.2204 m along Link_6's
  z axis** (`ee_pos_offset` in `config/params.yaml`). Applying it at a different point gives
  few-mm errors whenever the tool rotates.
- `[3:6]` rotation delta as a rotation vector, world frame, **left-multiplied**:
  `R_next = R(drot) · R_now`.
- `[6]` gripper: **≥ 0.5 = open, < 0.5 = close**. The node feeds its own last gripper command back
  as state and starts open (`initial_gripper_open`).
- One message = one 1/30 s of motion. Apply every message; applying them at another rate changes
  the arm's speed by the same factor. Typical steps are ~0.5 mm; the largest in training was 5.2 mm
  and 0.0089 rad.

### 6. Which policy

```bash
./run_docker.sh --policy-name act            # first choice
./run_docker.sh --policy-name diffusion_rtc  # diffusion, smooth and a steady 30 Hz
```

Avoid plain `diffusion` on the robot: its ~60 ms pause per chunk makes the node skip timer ticks, so
it publishes at only ~22 Hz and the arm moves ~26% slower than demonstrated. Stop the running node
before switching, or two nodes publish on `/il/grasp/action` at once.

### 7. First run

1. **Without moving the arm**: run the node against the robot's topics (or a
   recorded bag, see *Test with a recorded bag* above) and [check the output](#-check-output): about 30
   messages/s, small values, gripper at 1. The startup log should show `loaded act ...` (or
   `loaded diffusion ...`) and `requires 30.0 Hz`; `diffusion_rtc` also logs `RTC (guided ...)` at
   startup and `RTC: ... stalls 0` every 100 chunks.
2. **On the robot**: reduced speed limits, e-stop within reach, short runs. Watch for the arm
   drifting off the demonstrated path the longer it runs; that is the failure mode of delta actions.

## 🛠️ Development

### Add a policy

1. Subclass `PolicyModel` in `policies/your_policy.py`; see `policies/dummy_policy.py`. A policy
   with its own checkpoint/settings gets a subpackage with a `config.yaml` instead; see
   `policies/diffusion/`.
2. Define observation and action specs in `policies/base_policy.py`.
3. Register the policy in `policies/__init__.py` through `create_policy()`.
4. If its actions are tied to a fixed control period, set `inference_rate_hz` on the class; the
   node then ticks at that rate instead of `config/params.yaml`'s.
