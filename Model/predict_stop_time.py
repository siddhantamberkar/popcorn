import torch
import torch.nn as nn
import numpy as np
import sys

# === MODEL DEFINITION ===
class StopPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# === LOADING THE MODEL ===
model = StopPolicyNet()
model.load_state_dict(torch.load("stop_policy_model.pth"))
model.eval()

# === CDF MODEL ===
import joblib
spline_model = joblib.load("./spline_model.pkl")

# === GET USER INPUT ===
if len(sys.argv) < 2:
    print("Usage: python predict_stop_time.py <first_pop_time>")
    sys.exit(1)

first_pop_time = float(sys.argv[1])
time = np.arange(0, 180.01, 0.01)

# === SIMULATE POP ARRIVALS BASED ON FIRST POP ===
# Simple heuristic: simulate popping curve from first pop time
pop_times = np.array([first_pop_time + i for i in range(0, 60, 5)])  # Fake kernel times

def compute_max_age_curve(pop_times, time):
    max_age_curve = np.zeros_like(time)
    for i, t in enumerate(time):
        ages = np.clip(t - pop_times[pop_times <= t], 0, None)
        max_age_curve[i] = np.max(ages) if len(ages) > 0 else 0
    return max_age_curve

cdf = np.array([spline_model(ti) for ti in time])
max_age = compute_max_age_curve(pop_times, time)
features = np.stack([time, cdf, max_age], axis=1)
features_tensor = torch.tensor(features, dtype=torch.float32)

# === PREDICT STOP TIME ===
with torch.no_grad():
    preds = model(features_tensor).squeeze().numpy()

best_idx = np.argmin(np.abs(preds - time))
stop_time = time[best_idx]
remaining_time = stop_time - first_pop_time
print(f"Predicted Stop Time: {stop_time:.2f} seconds")
print(f"Expected Remaining Time: {remaining_time:.2f} seconds")
print(f"Predicted Max Kernel Age: {max_age[best_idx]:.2f} seconds")
