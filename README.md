# PPO for Inverted Pendulum (Pendulum-v1)

This project implements the **Proximal Policy Optimization (PPO)** algorithm using **TensorFlow 2.x** and **TensorFlow Probability** to solve the **Pendulum-v1** OpenAI Gym environment. The agent learns a continuous control policy to stabilize the pendulum by maximizing cumulative reward. The implementation supports both **PPO-Clip** and **PPO with KL Penalty** methods and demonstrates stable convergence.

---

## Repository Structure
- `inv_pendulum.py` – Main PPO implementation (actor, critic, buffer, training, and testing)
- `actor_weights.h5` – Trained actor network weights
- `critic_weights.h5` – Trained critic network weights
- `result_clip_1.txt` – PPO-Clip training results
- `inv_pendulum.txt` – Inverted pendulum experiment logs
- `pendulum.txt` – Environment performance logs
- `README.md` – Project documentation

---

## Requirements
- Python 3.x  
- TensorFlow 2.x  
- TensorFlow Probability  
- OpenAI Gym  
- NumPy  
- SciPy  

Install dependencies:
```bash
pip install tensorflow tensorflow-probability gym numpy scipy
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
