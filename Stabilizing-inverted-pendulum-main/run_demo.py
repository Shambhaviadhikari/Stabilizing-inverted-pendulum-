"""
run_demo.py  –  Inverted Pendulum PPO Demo  (NO TensorFlow required)
=====================================================================
Compatible with: Python 3.12 | pure NumPy inference

Put these files in the SAME folder as this script:
    actor_weights (1).h5
    critic_weights (1).h5
    pendulum.py

Then run:
    python run_demo.py

Outputs (saved in same folder):
    angle_vs_time.png
    reward_per_step.png
    state_overview.png
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(20)

# ── file paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ACTOR_FILE  = os.path.join(SCRIPT_DIR, "actor_weights (1).h5")

# ── settings ──────────────────────────────────────────────────────────────────
UPPER_BOUND   = 2.0
TEST_EPISODES = 5
MAX_STEPS     = 500


# ─────────────────────────────────────────────────────────────────────────────
# Pure NumPy Actor  (mirrors: Dense(128,relu) → Dense(64,relu) → Dense(64,relu)
#                             → Dense(1,tanh) * upper_bound)
# ─────────────────────────────────────────────────────────────────────────────
def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

class NumpyActor:
    def __init__(self, weights):
        """
        weights: list of (W, b) tuples for each Dense layer, in order.
        """
        self.weights = weights

    def predict(self, state):
        x = np.array(state, dtype=np.float32).reshape(1, -1)
        # Dense(128, relu)
        x = relu(x @ self.weights[0][0] + self.weights[0][1])
        # Dense(64, relu)
        x = relu(x @ self.weights[1][0] + self.weights[1][1])
        # Dense(64, relu)
        x = relu(x @ self.weights[2][0] + self.weights[2][1])
        # Dense(1, tanh) * upper_bound
        x = tanh(x @ self.weights[3][0] + self.weights[3][1]) * UPPER_BOUND
        return float(x[0, 0])


def load_actor(path):
    """Extract Dense layer weights from the .h5 file."""
    weights = []
    with h5py.File(path, "r") as f:
        # Print available keys to help debug if needed
        def find_dense_layers(name, obj):
            if isinstance(obj, h5py.Group):
                keys = list(obj.keys())
                if any("kernel" in k for k in keys) or any("bias" in k for k in keys):
                    return name
        
        # Try common Keras h5 weight storage formats
        try:
            # Keras 2 format: /model_weights/dense/dense/kernel:0
            model_weights = f["model_weights"]
            layer_names = [k for k in model_weights.keys() if "dense" in k.lower()]
            layer_names = sorted(layer_names)
            for ln in layer_names:
                grp = model_weights[ln]
                # navigate one level deeper if needed
                if ln in grp:
                    grp = grp[ln]
                kernel = grp["kernel:0"][:]
                bias   = grp["bias:0"][:]
                weights.append((kernel, bias))
        except KeyError:
            # Keras 3 / alternative format
            try:
                layer_names = sorted([k for k in f.keys() if "dense" in k.lower()])
                for ln in layer_names:
                    grp = f[ln]
                    # try nested
                    if ln in grp:
                        grp = grp[ln]
                    kernel = grp["kernel:0"][:]
                    bias   = grp["bias:0"][:]
                    weights.append((kernel, bias))
            except Exception:
                # Last resort: walk the file and grab all kernel/bias pairs in order
                kernels, biases = [], []
                def collect(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        if "kernel" in name:
                            kernels.append((name, obj[:]))
                        elif "bias" in name:
                            biases.append((name, obj[:]))
                f.visititems(collect)
                kernels.sort(key=lambda x: x[0])
                biases.sort(key=lambda x: x[0])
                weights = [(k[1], b[1]) for k, b in zip(kernels, biases)]

    if len(weights) < 4:
        raise ValueError(
            f"Expected 4 Dense layers in {path}, found {len(weights)}. "
            "The .h5 file structure may differ from expected."
        )

    print(f"  ✓ Actor weights loaded: {path}")
    print(f"    Layer shapes: {[w[0].shape for w in weights]}")
    return NumpyActor(weights)


def get_action(actor, state):
    act = actor.predict(state)
    return np.clip(np.atleast_1d(act), -UPPER_BOUND, UPPER_BOUND)


# ─────────────────────────────────────────────────────────────────────────────
# Load custom pendulum environment
# ─────────────────────────────────────────────────────────────────────────────
def load_pendulum_env():
    import importlib.util
    env_path = os.path.join(SCRIPT_DIR, "pendulum.py")
    spec = importlib.util.spec_from_file_location("pendulum_env", env_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.PendulumEnv()


# ─────────────────────────────────────────────────────────────────────────────
# Run episodes
# ─────────────────────────────────────────────────────────────────────────────
def run_episodes(env, actor, n_episodes=5):
    results = []
    for ep in range(n_episodes):
        raw   = env.reset()
        state = raw[0] if isinstance(raw, tuple) else raw

        angles, rewards, cos_th, sin_th, omega = [], [], [], [], []

        for t in range(MAX_STEPS):
            action = get_action(actor, state)

            step = env.step(action)
            if len(step) == 5:
                next_state, reward, terminated, truncated, _ = step
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step

            theta_deg = float(np.degrees(np.arctan2(state[1], state[0])))
            angles.append(theta_deg)
            rewards.append(float(reward))
            cos_th.append(float(state[0]))
            sin_th.append(float(state[1]))
            omega.append(float(state[2]))

            state = next_state
            if done:
                break

        total = sum(rewards)
        print(f"  Episode {ep+1:2d} | steps: {len(angles):4d} | "
              f"total reward: {total:8.2f}")
        results.append(dict(angles=angles, rewards=rewards,
                            cos_th=cos_th, sin_th=sin_th, omega=omega))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 – Angle vs Time
# ─────────────────────────────────────────────────────────────────────────────
def plot_angle_vs_time(episodes, save_path):
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    colors = plt.cm.cool(np.linspace(0.2, 0.9, len(episodes)))
    offset = 0.0

    for i, ep in enumerate(episodes):
        t = np.arange(len(ep["angles"])) * 0.036
        ax.plot(t + offset, ep["angles"],
                color=colors[i], linewidth=1.1, alpha=0.9,
                label=f"Ep {i+1}")
        offset += t[-1] + 0.5

    ax.axhline( 10, color="#00ffcc", lw=1.3, ls="--", alpha=0.8, label="±10° target")
    ax.axhline(-10, color="#00ffcc", lw=1.3, ls="--", alpha=0.8)
    ax.axhline(  0, color="white",   lw=0.5, ls=":",  alpha=0.3)

    ax.set_xlabel("Time (s)",    color="white", fontsize=12)
    ax.set_ylabel("Angle (deg)", color="white", fontsize=12)
    ax.set_title("Angle vs Time — Inverted Pendulum (Pre-trained PPO)",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.legend(facecolor="#1a1a2e", edgecolor="#555",
              labelcolor="white", fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 – Reward per step
# ─────────────────────────────────────────────────────────────────────────────
def plot_reward(episodes, save_path):
    ep  = episodes[0]
    t   = np.arange(len(ep["rewards"])) * 0.036
    r   = np.array(ep["rewards"])

    win    = 20
    pad    = np.pad(r, (win//2, win//2), mode="edge")
    smooth = np.convolve(pad, np.ones(win)/win, mode="valid")[:len(r)]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    ax.fill_between(t, r, alpha=0.2, color="#4fc3f7")
    ax.plot(t, r,      color="#4fc3f7", lw=0.8, alpha=0.6, label="Raw reward")
    ax.plot(t, smooth, color="#ffffff", lw=2.0,             label=f"Rolling avg ({win} steps)")

    ax.set_xlabel("Time (s)", color="white", fontsize=12)
    ax.set_ylabel("Reward",   color="white", fontsize=12)
    ax.set_title("Reward per Timestep — Episode 1",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.legend(facecolor="#1a1a2e", edgecolor="#555",
              labelcolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 – State overview 2x2
# ─────────────────────────────────────────────────────────────────────────────
def plot_state_overview(episodes, save_path):
    ep = episodes[0]
    t  = np.arange(len(ep["cos_th"])) * 0.036

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(2, 2, hspace=0.5, wspace=0.35)

    panels = [
        ("cos θ",         ep["cos_th"],  "#4fc3f7"),
        ("sin θ",         ep["sin_th"],  "#ffa726"),
        ("Reward",        ep["rewards"], "#66bb6a"),
        ("Angular vel ω", ep["omega"],   "#ef5350"),
    ]

    for idx, (title, data, color) in enumerate(panels):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        ax.set_facecolor("#0d1117")
        ax.plot(t, data, color=color, lw=1.1)
        ax.fill_between(t, data, alpha=0.15, color=color)
        ax.set_title(title,       color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel("time (s)", color="#aaa",  fontsize=9)
        ax.tick_params(colors="#aaa")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    fig.suptitle("State Overview — Episode 1 (Virtual ARX Model)",
                 color="white", fontsize=14, fontweight="bold")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Inverted Pendulum PPO Demo (NumPy-only) ===\n")

    # 1. Load weights
    print("[1/4] Loading actor weights...")
    if not os.path.exists(ACTOR_FILE):
        raise FileNotFoundError(
            f"\nCannot find: {ACTOR_FILE}\n"
            "Make sure 'actor_weights (1).h5' is in the same folder as this script.")
    actor = load_actor(ACTOR_FILE)

    # 2. Load environment
    print("\n[2/4] Loading environment...")
    try:
        env = load_pendulum_env()
        print("  Custom ARX pendulum.py loaded")
    except Exception as e:
        print(f"  pendulum.py failed ({e})")
        print("  Falling back to gymnasium Pendulum-v1")
        import gymnasium as gym
        env = gym.make("Pendulum-v1")

    # 3. Run episodes
    print(f"\n[3/4] Running {TEST_EPISODES} episodes...")
    data = run_episodes(env, actor, n_episodes=TEST_EPISODES)
    env.close()

    avg = np.mean([sum(ep["rewards"]) for ep in data])
    print(f"\n  Average total reward: {avg:.2f}")

    # 4. Save plots
    print("\n[4/4] Saving plots...")
    plot_angle_vs_time(data,  os.path.join(SCRIPT_DIR, "angle_vs_time.png"))
    plot_reward(data,         os.path.join(SCRIPT_DIR, "reward_per_step.png"))
    plot_state_overview(data, os.path.join(SCRIPT_DIR, "state_overview.png"))

    print(f"\nDone! Plots saved to: {SCRIPT_DIR}")