# ROS 2 Policy Inference

Runs a trained policy (ACT, Diffusion, ...) on the robot's camera and joint topics and publishes
one action every 1/30 s on `/il/grasp/action`.

1. [Install on a new PC](#1-install-on-a-new-pc)
2. [Run on the real robot](#2-run-on-the-real-robot)
3. [Use the policy with depth](#3-use-the-policy-with-depth)
4. [See what the policy looks at (Grad-CAM)](#4-see-what-the-policy-looks-at-grad-cam)
5. [Reference](#reference): policies, checkpoints, problems

Everything else (topic formats, start pose, how actions must be applied, adding a policy) is in
[docs/README_details.md](docs/README_details.md).

## What's in this folder

A ROS 2 (Jazzy) Python package that runs inside Docker. Your code and the checkpoints are mounted
into the container, and the package is rebuilt every time the container starts, so you can edit a
file and restart without rebuilding the image.

```text
ros2_policy_inference/
├── run_docker.sh            # start here: ./run_docker.sh --policy-name <policy> [--checkpoint <name>]
├── ros2_policy_inference/   # the Python code
│   ├── inference_node.py    #   the ROS node: reads topics, calls the policy, publishes actions
│   ├── ros_bridge.py        #   subscribers/publisher, ROS messages -> numpy arrays
│   ├── preprocessing.py     #   image resize and end-effector offset, the same as when the data was recorded
│   ├── state_range.py       #   compares the robot state to the training range (log only)
│   └── policies/            #   one folder per policy, each with its own config.yaml (checkpoint path, settings)
│       ├── diffusion/       #     Diffusion Policy (+ rtc.py, smooth chunking)
│       ├── depth_inference/ #     the same + depth image input
│       ├── act/             #     ACT
│       ├── act_depth/       #     ACT + depth image input
│       ├── liteflow/        #     InternVL3 + FLOWER
│       └── __init__.py      #     policy name -> code, and the --checkpoint lookup
├── config/params.yaml       # topic names, image size, TCP offset, gripper threshold
├── launch/                  # ros2 launch file (policy_name:=, checkpoint:=)
├── docker/                  # Dockerfiles, docker-compose.yml, entrypoint (builds the package)
├── docs/                    # README_details.md: the full reference
├── checkpoints/   (not in git, ~4 GB)  # model weights, mounted at /checkpoints in the container
└── bags/          (not in git, ~2 GB)  # recorded robot topics for testing without the robot
```

**Basics**
- **Input:** the wrist camera (RGB and depth), joint states, TCP pose and F/T sensor topics.
  **Output:** `/il/grasp/action` = `[dx, dy, dz, drx, dry, drz, gripper]` at 30 Hz, giving the
  position change in metres and the rotation change in radians (world frame), and the gripper
  command (1 = open, 0 = closed).
- The node itself never moves the arm. The Delta Extension (Fan Grasp Mode IL) applies the actions
  to the robot; it is not part of this repo.
- The models were trained in `Robot_Friday` (dataset `data/lerobot_data`, training scripts in `baselines/`).

---

## 1. Install on a new PC

You need only Docker and an NVIDIA GPU. ROS 2, PyTorch and LeRobot all come inside the Docker image.

**1. NVIDIA driver ≥ 580.** Check with `nvidia-smi`.

**2. Docker, and permission to run it without sudo:**
```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER        # then log out and back in
```

**3. NVIDIA Container Toolkit**, which lets Docker use the GPU
([install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):
```bash
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```

**4. The Docker image.** Choose one:
```bash
# a) on the lab network: save a GitLab token (scope read_api) once; run_docker.sh downloads the image
printf '%s' 'glpat-...' > ~/.gitlab_token && chmod 600 ~/.gitlab_token

# b) anywhere else: build it yourself (uses public sources only)
docker compose -f docker/docker-compose.yml build policy-inference
```

**5. The checkpoints.** They are not in git. Copy the `checkpoints/` folder from the training PC
into this folder, then check that the copy is complete:
```bash
cd checkpoints && sha256sum -c SHA256SUMS && cd ..     # every line must say OK
```
```text
checkpoints/
├── act/checkpoint_act_375/
├── diffusion/checkpoint_diffusion_375/        # RGB only
├── diffusion/checkpoint_diffusion_50/
├── diffusion_depth/checkpoint_diffusion_375/  # RGB + depth
└── diffusion_depth/checkpoint_diffusion_50/
```

**6. Test the install without a robot:**
```bash
./run_docker.sh --policy-name diffusion_rtc --checkpoint diffusion_50k
```
It works if the log shows `loaded diffusion from /checkpoints/diffusion/checkpoint_diffusion_50`
and `warmed up`, and then repeats `waiting for all observation topics...`. Nothing is publishing
yet, so the waiting messages are expected. Stop it with Ctrl+C.

---

## 2. Run on the real robot

The policy node only **reads** topics and **publishes actions**. It never moves the arm by itself.

### 2a. Dry run with fake input (recorded bag, no robot)

This checks the whole chain (topics → model → actions) on your PC. Use two terminals, all
started in this folder.

```bash
# terminal 1: the policy
./run_docker.sh --policy-name diffusion_rtc --checkpoint diffusion_50k

# terminal 2: replay a recording of the robot's topics, on a loop
docker run --rm -it --network host --entrypoint bash \
  -v "$PWD/bags/fan_insertion_2026-09-22:/bag:ro" lerobot_ros2:latest -c \
  'source /opt/ros/jazzy/setup.bash && ros2 bag play /bag --loop --topics \
     /camera_L/color/image_raw/compressed /camera_L/depth/aligned_depth_to_color \
     /left_arm/joint_states /left_arm/tcp_pose_world /left_ft/wrench_compensated'
```
It works if terminal 1 logs `action: [...]` about 30 times per second. The bag was recorded
during fan *insertion*, not grasping, so this only proves the pipeline runs, not that the
grasp is good. The bag is not in git; copy `bags/` over like the checkpoints.

### 2b. With the real robot's observations

1. **Robot side:** start the robot and camera drivers. All five topics must be publishing, or the
   node waits forever:
   `/camera_L/color/image_raw/compressed`, `/camera_L/depth/aligned_depth_to_color`,
   `/left_arm/joint_states`, `/left_arm/tcp_pose_world`, `/left_ft/wrench_compensated`.
2. **Same ROS network:** the PC must use the same `ROS_DOMAIN_ID` as the robot (default 0). To use
   another ID, put `ROS_DOMAIN_ID=<id>` in a file named `.env` in this folder.
3. **Start the policy:**
   ```bash
   ./run_docker.sh --policy-name diffusion_rtc --checkpoint diffusion_50k
   ```
4. **Check before moving anything:** the node should log `action: [...]` about 30 times per second,
   with small values (about 0.0005) and the gripper value at `1`.
5. **Let the arm move:** the actions are applied by the Delta Extension (Fan Grasp Mode
   **IL**, Arm Sync Mode **Per Frame** or **Per Task**, *Publish Policy Observations* **off**).
   Use reduced speed limits, keep the e-stop within reach, and start the arm near where the
   demonstrations started.
6. **Hand-off:** when the extension starts its IL step (after the coarse approach), the node logs
   `IL hand-off #s1: new /il/grasp/action subscriber delta_extension_node` and resets the policy,
   the fed-back gripper state and the gripper debounce. The node detects the IL step's subscription
   itself, so this works with any delta_extension version, no extension change needed. If the line
   doesn't appear when the fine approach starts, stop the run.
   > **Known limit** (Per Frame check in Isaac Sim, 2026-10-14, extension as on origin/main): the
   > extension handles one ROS message per frame across all its topics, so only ~11% of the actions
   > reached the IL step (3.3 of 30 per second). The arm moves much slower than in the demos, and a
   > short gripper close can be missed, so the step may never end by itself.

---

## 3. Use the policy with depth

Two policies were trained on the wrist **RGB image + depth image**:

- `depth_inference_rtc`: the Diffusion Policy.
- `depth_act`: ACT.

The RGB-only `diffusion_rtc` and `act` stay as they were, so you can switch between them to compare.

```bash
./run_docker.sh --policy-name depth_inference_rtc --checkpoint diffusion_50k   # depth diffusion, 50k (best val loss)
./run_docker.sh --policy-name depth_inference_rtc                              # depth diffusion, 375k (default)
./run_docker.sh --policy-name depth_act                                        # depth ACT, 375k (the only one)
```

Everything else is the same as above: the robot ([2](#2-run-on-the-real-robot)) and the bag test
([2a](#2a-dry-run-with-fake-input-recorded-bag-no-robot)) already publish the depth topic `/camera_L/depth/aligned_depth_to_color`, so only the policy name
changes.

**It works if** the startup log shows:
```text
loaded diffusion from /checkpoints/diffusion_depth/checkpoint_diffusion_50 ...
warmed up with 3 unguided + 3 guided inferences ...
depth input observation.images.wrist_depth: 0-1 m -> 255-1, no reading -> 0
```
For `depth_act` the first line is `loaded act from /checkpoints/act_depth/checkpoint_act_375 ...`.

**How depth reaches the model.** The node turns each depth frame into the same image the model was
trained on:

- 0 means no reading;
- otherwise brightness goes from 255 at 0 m down to 1 at 1 m, and anything farther is 1;
- the value is repeated in 3 channels.

The range is set in `policies/depth_inference/config.yaml` and `policies/act_depth/config.yaml`
(`depth_min_m`, `depth_max_m`). It must
match the training data. At startup the node compares it with `depth_encoding.json` in the
checkpoint folder and stops if they differ.

**Checkpoints:** each folder holds the model files plus `depth_encoding.json`. The dataset for both
is Robot_Friday `data/lerobot_data_depth`.

- `checkpoints/diffusion_depth/checkpoint_diffusion_{50,375}/`, from Robot_Friday
  `checkpoints/checkpoint_diffusion_depth/{050000,375000}/pretrained_model/`.
- `checkpoints/act_depth/checkpoint_act_375/`, from Robot_Friday
  `checkpoints/checkpoint_act_depth/375000/pretrained_model/`.

**Limits:**
- About 80% of depth pixels have no reading: the camera fills only a fixed window of the image,
  and the centre loses more readings as the gripper gets close to the fan. The training data had
  the same gaps, so this is not an error.
- Depth diffusion: on the held-out demos it scored about the same as RGB-only at 50k (0.072 vs
  0.073).
- Depth ACT: its action error on recorded demos equals RGB-only ACT's (0.078 for both at 375k).
  Both ACT models trained on all demos, so there is no held-out score.
- Neither depth model has been tried on the robot yet.

---

## 4. See what the policy looks at (Grad-CAM)

Optional, for debugging. Add `--gradcam` to any `act`, `depth_act`, `diffusion*` or
`depth_inference*` run; the overlays are published on `/policy/gradcam/image` (view them with any
ROS image viewer, e.g. `rqt_image_view`):

```bash
./run_docker.sh --policy-name depth_act --gradcam
```

Each camera the policy uses is shown with a heat map on top: red regions drive the predicted
motion most, blue ones not at all. The percentage is that camera's share, so with depth policies
you can see how much they rely on RGB vs depth. The bottom line shows the motion being explained
(sum of the next steps, in mm).

- **What is explained:** how far the predicted actions are from "do nothing", over the motion
  dims by default. `ros2 launch ... gradcam_target:=gripper` (or `position`, `rotation`, `all`)
  explains those dims instead. When the policy predicts almost no motion, there is little to
  explain and the map is mostly noise.
- **Cost:** it runs on its own thread at up to 5 Hz (`gradcam_rate_hz` in `config/params.yaml`)
  and doesn't change the actions. Measured with depth_act: actions stayed at 30 Hz. Per map on an
  RTX 4070 Ti SUPER: ACT ~10–25 ms, diffusion ~100 ms and ~1.2 GB extra GPU memory.
- **Diffusion:** the model only sees a 240×240 center crop, so an 8 px border stays blue. Its
  sampling uses a fixed seed for the map, so successive maps differ only by the camera input.
- Topic: `/policy/gradcam/image` (`sensor_msgs/Image`, rgb8). Code: `ros2_policy_inference/gradcam.py`.

---

## Reference

### Policies

```bash
./run_docker.sh --policy-name <policy> [--checkpoint <name>]
```

| `--policy-name` | What it is | Default checkpoint |
|:--|:--|:--|
| `diffusion_rtc` | Diffusion Policy, RGB camera. **Use this one on the robot** | `diffusion/checkpoint_diffusion_375` |
| `diffusion` | same model without smooth chunking (pauses ~60 ms every 8 actions) | same |
| `depth_inference_rtc` | Diffusion Policy, RGB **+ depth** camera | `diffusion_depth/checkpoint_diffusion_375` |
| `depth_inference` | same without smooth chunking | same |
| `act` | LeRobot ACT, RGB camera | `act/checkpoint_act_375` |
| `depth_act` | LeRobot ACT, RGB **+ depth** camera | `act_depth/checkpoint_act_375` |
| `liteflow` | InternVL3-1B + FLOWER (own image, weights downloaded automatically) | — |
| `dummy` | outputs nothing useful; for testing the topics | — |

`--checkpoint` picks another checkpoint from that policy's folder:

| `--checkpoint` | diffusion / diffusion_rtc | depth_inference / depth_inference_rtc |
|:--|:--|:--|
| *(none)* | `diffusion/checkpoint_diffusion_375` | `diffusion_depth/checkpoint_diffusion_375` |
| `diffusion_50k` | `diffusion/checkpoint_diffusion_50` | `diffusion_depth/checkpoint_diffusion_50` |

The 50k checkpoints had the lowest loss on held-out demonstrations (0.073 RGB, 0.072 RGB+depth,
against 0.141 and 0.123 at 375k). Neither has been tried on the robot yet.

Run only **one** policy at a time. All of them publish on the same topic. Stop one with Ctrl+C
(`docker ps` lists running containers) before starting another.

### Problems

| You see | Meaning / fix |
|:--|:--|
| `waiting for all observation topics...` / `no publisher found for ...` | one of the five input topics is missing. Start the robot or bag, and check `ROS_DOMAIN_ID` |
| no `IL hand-off` line when the IL step starts | the extension's ROS node isn't named `delta_extension_node` (add its name to `handoff_subscriber_nodes` in `config/params.yaml`), or it is on another `ROS_DOMAIN_ID` |
| `checkpoint '...' not found` | the folder isn't under `checkpoints/`; the message lists the ones that are there |
| `could not select device driver "nvidia"` | NVIDIA Container Toolkit missing (install step 3) |
| `permission denied ... docker.sock` | not in the `docker` group (install step 2, then log out and back in) |
| `No GitLab token found` | save a token (install step 4a) or build the image (4b) |
| `obs['depth'] missing` | `depth_inference*` and `depth_act` need `/camera_L/depth/aligned_depth_to_color` |
| `ros2: command not found` on the host | expected: ROS is only inside the container. Use the commands above |
