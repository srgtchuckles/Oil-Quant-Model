import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

oil = yf.download("CL=F", period="5y")
xom = yf.download("XOM", period="5y")

data = pd.DataFrame()
data["oil_close"] = oil["Close"]
data["xom_close"] = xom["Close"]
data = data.dropna()

data["log_oil"] = np.log(data["oil_close"])
data["log_xom"] = np.log(data["xom_close"])

# Spread = how far XOM is from its normal relationship with oil
window = 60
data["spread"] = data["log_xom"] - data["log_oil"]
data["spread_mean"] = data["spread"].rolling(window).mean()
data["spread_std"] = data["spread"].rolling(window).std()
data["z_score"] = (data["spread"] - data["spread_mean"]) / data["spread_std"]

# Entry: z > 2 means XOM overpriced vs oil → short XOM
# Entry: z < -2 means XOM underpriced vs oil → long XOM
# Exit: when spread reverts to within ±0.5
ENTRY_Z = 2.0
EXIT_Z = 0.5

position = []
current_pos = 0
for _, row in data.iterrows():
    z = row["z_score"]
    if row["z_score"] > ENTRY_Z:
        current_pos = -1
    elif row["z_score"] < -ENTRY_Z:
        current_pos = 1
    elif abs(z) < EXIT_Z:
        current_pos = 0
    position.append(current_pos)

data["position"] = position
data["xom_return"] = data["xom_close"].pct_change()
data["strategy_return"] = data["position"].shift(1) * data["xom_return"]

data_clean = data.dropna()
data_clean = data_clean
cumret   = (1 + data_clean["strategy_return"]).cumprod()
buyhold  = (1 + data_clean["xom_return"]).cumprod()

total_return   = cumret.iloc[-1] - 1
buyhold_return = buyhold.iloc[-1] - 1
sharpe         = data_clean["strategy_return"].mean() / data_clean["strategy_return"].std() * np.sqrt(252)
rolling_max    = cumret.cummax()
max_dd         = ((cumret - rolling_max) / rolling_max).min()
days_active    = (data_clean["position"] != 0).sum()

print("--- Pairs Trading: XOM vs Oil ---")
print(f"Strategy return:   {total_return:+.1%}")
print(f"Buy & hold return: {buyhold_return:+.1%}")
print(f"Sharpe ratio:      {sharpe:.2f}")
print(f"Max drawdown:      {max_dd:.1%}")
print(f"Days in market:    {days_active} / {len(data_clean)}")

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(data_clean.index, cumret,  label="Pairs Strategy", color="steelblue")
axes[0].plot(data_clean.index, buyhold, label="Buy & Hold XOM", color="gray", alpha=0.6)
axes[0].set_title("Cumulative Returns: Pairs Strategy vs Buy & Hold XOM")
axes[0].set_ylabel("Growth of $1")
axes[0].legend()

axes[1].plot(data_clean.index, data_clean["z_score"], color="steelblue")
axes[1].axhline( ENTRY_Z, color="red",   linestyle="--", linewidth=0.8, label=f"Entry ±{ENTRY_Z}")
axes[1].axhline(-ENTRY_Z, color="red",   linestyle="--", linewidth=0.8)
axes[1].axhline( EXIT_Z,  color="green", linestyle="--", linewidth=0.8, label=f"Exit ±{EXIT_Z}")
axes[1].axhline(-EXIT_Z,  color="green", linestyle="--", linewidth=0.8)
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_title("Z-Score of XOM/Oil Spread")
axes[1].legend()

axes[2].plot(data_clean.index, data_clean["position"], color="steelblue", drawstyle="steps-post")
axes[2].set_title("Position (1 = Long XOM, -1 = Short XOM, 0 = Flat)")
axes[2].set_yticks([-1, 0, 1])

plt.tight_layout()
plt.savefig("PairsTrading.png", dpi=150)
plt.show()
