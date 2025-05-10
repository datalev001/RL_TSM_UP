# =========================================================
# 0. LIBRARIES
# =========================================================
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------------------------------------
#  DATA  ►  simulate → save CSV → reload
# ---------------------------------------------------------
DATA_PATH  = "RL_sales_data.csv"

# -------- reload so downstream code reads from file ------
data = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date")

# ---------------------------------------------------------
#  FROZEN BASELINE   (train on first 300 days)
# ---------------------------------------------------------
TRAIN_END = 300
train_y = data["sales"][:TRAIN_END]
arma = ARIMA(train_y, order=(2,0,1)).fit()

baseline_pred, history = [], list(train_y)
for t in range(TRAIN_END, N):
    y_hat = ARIMA(history, order=(2,0,1)).fit().forecast()[0]
    baseline_pred.append(y_hat)
    history.append(data["sales"].iloc[t])

data.loc[data.index[TRAIN_END:], "baseline"] = baseline_pred

# ---------------------------------------------------------
#  GYM ENV  (fixed step() to avoid out-of-range)
# ---------------------------------------------------------
WINDOW, GAMMA = 7, 0.95
class SalesCorrectEnv(gym.Env):
    def __init__(self, df, start):
        super().__init__()
        self.df, self.start = df.reset_index(drop=True), start
        obs_dim = WINDOW + 4 + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space      = spaces.Box(-1., 1., (1,), np.float32)
        self.reset()
    def _get_obs(self):
        res = self.residuals[-WINDOW:]
        if len(res) < WINDOW:
            res = np.concatenate([np.zeros(WINDOW-len(res)), res])
        row = self.df.loc[self.t]
        doy = 2*np.pi*(self.t % 365)/365
        return np.concatenate([res,
                               [row.price, row.comp_price, row.promo, row.mkt],
                               [np.sin(doy), np.cos(doy)]]).astype(np.float32)
    def step(self, action):
        w = np.clip(0.2*action[0]+1.0, 0.8, 1.2)
        row   = self.df.loc[self.t]
        y_hat = w * row.baseline
        r     = (abs(row.sales-row.baseline)-abs(row.sales-y_hat)) / (abs(row.sales-row.baseline)+1e-6)
        self.residuals.append(row.sales-y_hat)
        self.t += 1
        done   = self.t >= len(self.df)
        next_obs = np.zeros(self.observation_space.shape, np.float32) if done else self._get_obs()
        return next_obs, r, done, False, {}
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t, self.residuals = self.start, []
        return self._get_obs(), {}

env = DummyVecEnv([lambda: SalesCorrectEnv(data, TRAIN_END)])

# ---------------------------------------------------------
#  TRAIN PPO
# ---------------------------------------------------------
model = PPO("MlpPolicy", env,
            policy_kwargs=dict(net_arch=[64,64]),
            learning_rate=3e-4, batch_size=128, gamma=GAMMA,
            verbose=0).learn(30_000)

# ---------------------------------------------------------
# GENERATE RL-CORRECTED FORECASTS
# ---------------------------------------------------------
env_eval = SalesCorrectEnv(data, TRAIN_END); obs,_ = env_eval.reset()
corrected = []
for _ in range(TRAIN_END, N):
    act,_ = model.predict(obs, deterministic=True)
    obs, _, _, _, _ = env_eval.step(act)
    corrected.append(env_eval.residuals[-1] + data["baseline"].iloc[len(corrected)+TRAIN_END])
data.loc[data.index[TRAIN_END:], "corrected"] = corrected

# ---------------------------------------------------------
# METRICS & VISUAL
# ---------------------------------------------------------
test_y = data["sales"].iloc[TRAIN_END:]
base_p = data["baseline"].iloc[TRAIN_END:]
corr_p = data["corrected"].iloc[TRAIN_END:]

print("\n=== Hold-out results ===")
print(f"Baseline RMSE {np.sqrt(mean_squared_error(test_y, base_p)):.2f}  "
      f"MAE {mean_absolute_error(test_y, base_p):.2f}")
print(f"PPO-patch RMSE {np.sqrt(mean_squared_error(test_y, corr_p)):.2f}  "
      f"MAE {mean_absolute_error(test_y, corr_p):.2f}\n")

plt.figure(figsize=(11,4))
plt.plot(test_y.index, test_y, label="Actual")
plt.plot(base_p.index, base_p, label="Baseline")
plt.plot(corr_p.index, corr_p, label="RL-Corrected")
plt.legend(); plt.tight_layout(); plt.show()
