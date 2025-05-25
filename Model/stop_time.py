import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# --- Load your pre-trained spline model ---
spline_model = joblib.load("./spline_model.pkl")

# --- Helper function to compute max age curve ---
def compute_max_age_curve(pop_times, time):
    """
    Calculate the maximum age of all popped kernels over time.
    """
    max_age_curve = np.zeros_like(time)
    for i, t in enumerate(time):
        ages = np.clip(t - pop_times[pop_times <= t], 0, None)
        if len(ages) > 0:
            max_age_curve[i] = np.max(ages)
        else:
            max_age_curve[i] = 0
    return max_age_curve

# --- Optimization function to find best stop time ---
def optimize_stop_time(spline_model, time, pop_times, alpha=0.5):
    """
    Find optimal stop time by minimizing the objective:
    L(t) = alpha * (1 - CDF(t)) + (1 - alpha) * MaxAge(t)
    """
    cdf_pred = np.array([spline_model(ti) for ti in time])
    max_age_curve = compute_max_age_curve(pop_times, time)
    age_norm = (max_age_curve - max_age_curve.min()) / (max_age_curve.max() - max_age_curve.min() + 1e-8)
    cdf_loss = 1 - cdf_pred
    total_loss = alpha * cdf_loss + (1 - alpha) * age_norm
    best_idx = np.argmin(total_loss)
    return time[best_idx], cdf_pred[best_idx], max_age_curve[best_idx]

# --- Load and prepare data ---
df = pd.read_csv('../Data/popcorn_pops_custom.csv') 

df['pop_time_seconds'] = df['pop_time_seconds'].apply(lambda x: np.random.normal(150,20) if x < 20 else x)
df['trial'] = df['trial'] + 1

import random
random.seed(42)

sample = random.sample(range(1, 40), 30)
train = pd.DataFrame()
for trial in sample:
    train = pd.concat([train, df[df['trial'] == trial]])

test = pd.DataFrame()
for i in range(1, max(df['trial'])+1):
    if i not in sample:
        test = pd.concat([test, df[df['trial'] == i]])

time = np.arange(0, 180.01, 0.01)

# --- Prepare training data ---
X = []
y = []

for trial in train['trial'].unique():
    pop_times = np.sort(train[train['trial'] == trial]['pop_time_seconds'].values)
    cdf = np.array([spline_model(ti) for ti in time])
    max_age = compute_max_age_curve(pop_times, time)
    features = np.stack([time, cdf, max_age], axis=1)
    t_star, _, _ = optimize_stop_time(spline_model, time, pop_times, alpha=0.5)
    X.append(features)
    y.append(t_star * np.ones(len(time)))

X = np.vstack(X)
y = np.hstack(y)

# --- Define PyTorch model ---
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

# --- Prepare data loader ---
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# --- Train model ---
model = StopPolicyNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(30):
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, loss: {loss.item():.4f}")

# --- Interactive prediction function ---
def predict_stop_given_first_pop(model, spline_model, first_pop_time, all_pop_times, full_time):
    # Filter time array from first pop onwards
    time_after_first_pop = full_time[full_time >= first_pop_time]
    # Filter pop_times to those after first_pop_time for max age calc
    filtered_pop_times = all_pop_times[all_pop_times >= first_pop_time]

    # Calculate features
    cdf = np.array([spline_model(ti) for ti in time_after_first_pop])
    max_age = compute_max_age_curve(filtered_pop_times, time_after_first_pop)
    features = np.stack([time_after_first_pop, cdf, max_age], axis=1)

    # Predict stop times from model
    features_tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        preds = model(features_tensor).numpy().flatten()

    predicted_stop_time = np.mean(preds)
    expected_remaining = predicted_stop_time - first_pop_time

    # Get max age at predicted stop time (approximate)
    idx_closest = (np.abs(time_after_first_pop - predicted_stop_time)).argmin()
    max_age_at_stop = max_age[idx_closest]

    return predicted_stop_time, expected_remaining, max_age_at_stop

# # Save the trained model
# torch.save(model.state_dict(), "stop_policy_model.pth")

# --- Main interactive loop ---
print("\n=== Popcorn Microwave Stop Time Predictor ===")
print("After the first kernel pops, enter the pop time in seconds (e.g., 42.42)")
print("Type 'exit' to quit.")

# For demo, get all pop times from a random test trial
test_trial = test['trial'].unique()[0]
all_pop_times_test_trial = np.sort(test[test['trial'] == test_trial]['pop_time_seconds'].values)

while True:
    user_input = input("\nEnter first pop time in seconds (or 'exit'): ").strip()
    if user_input.lower() == 'exit':
        print("Goodbye!")
        break
    try:
        first_pop = float(user_input)
        if first_pop < 0 or first_pop > 180:
            print("Please enter a value between 0 and 180 seconds.")
            continue

        stop_time, remaining, max_age = predict_stop_given_first_pop(
            model, spline_model, first_pop, all_pop_times_test_trial, time
        )
        print(f"Predicted stop time: {stop_time:.2f} sec")
        print(f"Expected remaining time after first pop: {remaining:.2f} sec")
        print(f"Max kernel age at stop: {max_age:.2f} sec")

    except ValueError:
        print("Invalid input. Please enter a numeric value (e.g., 12.34) or 'exit'.")
