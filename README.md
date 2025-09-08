# 🚀 OptiChain — AI-Powered Supply Chain Optimization

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-orange.svg)
![SimPy](https://img.shields.io/badge/SimPy-simulation-lightgrey.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)


OptiChain is an **agent-based, reinforcement learning** platform for supply chain optimization. It simulates factories, warehouses and markets using a custom Gymnasium environment built on **SimPy**, trains agents with **Stable-Baselines3 (PPO)**, and provides scripts to test & evaluate learned policies.

---

## Table of Contents

1. [Highlights](#highlights)
2. [Project Layout](#project-layout)
3. [Quickstart — step-by-step (working)](#quickstart)
4. [Detailed Usage & Code Snippets](#snippets)

   * [1) Inspect / run the environment (smoke test)](#env-snippet)
   * [2) Train an agent (PPO)](#train-snippet)
   * [3) Evaluate an agent vs baseline](#eval-snippet)
   * [4) Load a saved model and run a single episode](#load-snippet)
5. [How it works (brief)](#how-it-works)
6. [Tips, troubleshooting & recommendations](#tips)
7. [Roadmap](#roadmap)
8. [Contributing](#contributing)
9. [License & Contact](#license)

---

## Highlights

* Custom Gymnasium environment: `ares_environment/supply_chain_env.py` (factory → warehouse → market flow)
* Agent training script: `train_agent.py` (Stable-Baselines3 PPO)
* Environment test script: `test_env.py` (checks and simple stepping)
* Evaluation script: `evaluate_agent.py` (compare trained agent with a constant baseline)
* Logs & models: `logs/` and `trained_models/`

---

## Project Layout

```
OptiChain/  (local folder name: ARES-Supply-Chain-Optimization-main)
├── .gitignore
├── README.md                # (you are editing this)
├── requirements.txt         # python dependencies
├── train_agent.py           # train RL agent (PPO)
├── test_env.py              # smoke tests and demo stepping through env
├── evaluate_agent.py        # evaluate saved agent vs baseline
├── ares_environment/        # custom environment implementation
│   ├── __init__.py
│   ├── supply_chain_env.py  # main Gym environment class
│   └── simulation_nodes.py  # Factory, Warehouse, Market classes and helpers
├── docs/                    # figures (training_graph.png, etc.)
├── logs/                    # training logs (TensorBoard friendly)
├── trained_models/          # saved model(s), e.g. ppo_ares_agent.zip
└── venv/                    # (this is included in the zip; **recommend removing from repo**)
```

> ⚠️ **Important:** `venv/` is included in the project archive but should not be tracked in GitHub. Remove it before pushing: see [Tips](/#tips) below.

---

## Quickstart — working step-by-step

These exact commands will get the repository running locally (tested for a local CPU-based setup):

1. **Clone (or download & extract) the repo**

```bash
# if you haven't already
git clone https://github.com/meanderinghuman/OptiChain.git
cd OptiChain
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
# mac/linux
source venv/bin/activate
# windows (PowerShell)
venv\Scripts\Activate.ps1
# windows (cmd)
venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> If you want GPU support, install a PyTorch build that matches your CUDA version (official PyTorch install instructions). The `requirements.txt` contains `torch`, but for CUDA you may prefer installing `torch` manually.

4. **(Optional) Run an environment smoke test** — confirms the custom Gym env follows the API

```bash
python test_env.py
```

You should see printed messages showing environment checking and a short 20-step demonstration with observations/rewards.

5. **Train the PPO agent** (default short run)

```bash
python train_agent.py
```

* By default the script trains with `TIMESTEPS_TO_TRAIN = 50000`. Edit `train_agent.py` to increase `TIMESTEPS_TO_TRAIN` (e.g. `1_000_000`) for serious training.
* Training outputs live logs to `logs/<timestamp>/` and saves the final model as `trained_models/ppo_ares_agent.zip`.

6. **Monitor training with TensorBoard** (open a separate terminal)

```bash
tensorboard --logdir=logs/ --port=6006
# then open http://localhost:6006 in your browser
```

7. **Evaluate the trained agent**

```bash
python evaluate_agent.py
```

This script loads `trained_models/ppo_ares_agent.zip` and compares average reward against a simple constant-order baseline. It prints a summary; if the PPO agent wins, you'll see a success message.

---

## Detailed Usage & Code Snippets

Below are short, copyable snippets that match how the repo's scripts use the environment and models.

### 1) Inspect / run the environment (smoke test)

This is the same idea implemented by `test_env.py`.

```python
from ares_environment.supply_chain_env import SupplyChainEnv

env = SupplyChainEnv()
# Gymnasium-style reset returns (obs, info)
obs, info = env.reset()
print("Initial observation:", obs)

for i in range(20):
    # random action in the normalized action space [-1, 1]
    action = env.action_space.sample()

    # step returns: observation, reward, terminated, truncated, info
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {i+1}: action={action} reward={reward:.2f} obs={obs}")

    if terminated or truncated:
        print("Episode ended")
        break

env.close()
```

**Observation space:** 3-dimensional `numpy` array: `[factory_inventory, warehouse_inventory, market_demand]`
**Action space:** single value `[-1, 1]` which the environment rescales to an integer order quantity via:

```text
order_quantity = int(((action[0] + 1) / 2) * env.max_order_quantity)
```

This keeps the agent's policy normalized and stable while allowing discrete order quantities internally.

---

### 2) Train an agent (PPO)

`train_agent.py` creates the environment, sets up a Stable-Baselines3 PPO model, trains and saves it. The core flow is:

```python
from stable_baselines3 import PPO
from ares_environment.supply_chain_env import SupplyChainEnv

env = SupplyChainEnv()
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/"
)
model.learn(total_timesteps=50000, tb_log_name="PPO_ARES_v1")
model.save("trained_models/ppo_ares_agent.zip")
env.close()
```

Change `total_timesteps` to a larger value for production training (e.g. `1_000_000+`).

---

### 3) Evaluate an agent vs baseline

`evaluate_agent.py` demonstrates a comparison run: it loads the trained PPO model and runs multiple episodes for both the learned policy and a simple baseline.

The sketch:

```python
import numpy as np
from stable_baselines3 import PPO
from ares_environment.supply_chain_env import SupplyChainEnv

env = SupplyChainEnv()
model = PPO.load("trained_models/ppo_ares_agent.zip", env=env)

# baseline: always order 30 units (rescaled to action space inside file)
# run several episodes and compare average rewards

# final printout: average reward PPO vs baseline
```

Run the file directly:

```bash
python evaluate_agent.py
```

---

### 4) Load a saved model and run one deterministic episode

```python
from stable_baselines3 import PPO
from ares_environment.supply_chain_env import SupplyChainEnv

env = SupplyChainEnv()
model = PPO.load("trained_models/ppo_ares_agent.zip", env=env)

obs, info = env.reset()
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

This is useful for producing policy rollouts for demo videos or plotting.

---

## How it works (brief)

* `ares_environment/supply_chain_env.py` implements a Gymnasium-compatible environment using **SimPy** to simulate time-based shipping and demand processes.
* The environment maintains three core components: `Factory`, `Warehouse`, and `Market` (see `ares_environment/simulation_nodes.py`).
* The RL agent controls ordering decisions (action → order quantity). Rewards are shaped to encourage revenue while penalizing holding costs and unmet demand.
* The repo uses stable-baselines3's PPO as the default learning algorithm for reliability and reproducibility.

---

## Tips, Troubleshooting & Recommendations

* **Remove `venv/` before pushing** to GitHub: it bloats the repo. Locally run:

```bash
# remove the folder from disk and stop tracking it
rm -rf venv
git rm -r --cached venv
# add venv to .gitignore
echo "venv/" >> .gitignore
```

* **Large requirements file**: If `pip install -r requirements.txt` fails because of binary wheels (e.g., `torch`), install PyTorch from its official installer for your OS/CUDA combination, then re-run `pip install -r requirements.txt` with `--no-deps` if needed.

* **TensorBoard logs not visible?** Make sure you started `tensorboard` pointing at the `logs/` parent directory that contains timestamped subfolders created during training.

* **Stable-Baselines3 version**: If you have an older SB3 or Gym/Gymnasium mismatch, you may see API errors. The repository uses `gymnasium` and SB3; upgrade/downgrade packages accordingly.

* **If training is slow**: reduce `total_timesteps` when experimenting, or run on a machine with GPU and install a CUDA-enabled PyTorch.


## Roadmap (short)

* Add a Flask-based dashboard to visualize live rollouts & KPIs
* Add configurable scenario files (demand profiles, shipping latency distributions)
* Integrate real datasets (CSV ingestion, external API connectors)
* Add automated unit tests for the environment dynamics
* Add hyperparameter tuning (Optuna / Ray Tune)


## Contributing

Contributions, issues and feature requests are welcome!

1. Fork this repository
2. Create a branch (`git checkout -b feature/awesome`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature/awesome`)
5. Open a Pull Request


## License & Contact

This project is released under the **MIT License**.
If you'd like to reach out: **Siddharth Pal** — meanderinghuman

---

