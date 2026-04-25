import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

oil = yf.download("CL=F", period="2y")
xom = yf.download("XOM", period="2y")

data = pd.DataFrame()
data["oil_close"] = oil["Close"]
data["xom_close"] = xom["Close"]

data["oil_return"] = data["oil_close"].pct_change()
data["xom_return"] = data["xom_close"].pct_change()
data["oil_vol20"] = data["oil_return"].rolling(20).std()

# Does oil lead XOM, and by how many days?
print("--- Does oil LEAD XOM? (correlation at each lag) ---")
for lag in range(6):
    corr = data["oil_return"].shift(lag).corr(data["xom_return"])
    print(f"  Oil return lag {lag} vs XOM return: {corr:.3f}")

# Does the relationship change based on how volatile oil is?
high_vol = data[data["oil_vol20"] > data["oil_vol20"].median()]
low_vol  = data[data["oil_vol20"] <= data["oil_vol20"].median()]
print(f"\nCorrelation in HIGH oil volatility regime: {high_vol['oil_return'].corr(high_vol['xom_return']):.3f}")
print(f"Correlation in LOW  oil volatility regime: {low_vol['oil_return'].corr(low_vol['xom_return']):.3f}")

# Rolling 60-day correlation over time
rolling_corr = data["oil_return"].rolling(60).corr(data["xom_return"])

# Plots
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

axes[0].plot(rolling_corr.index, rolling_corr, color="steelblue")
axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_title("Rolling 60-Day Correlation: Oil Return vs XOM Return")
axes[0].set_ylabel("Correlation")

lags = list(range(6))
corrs = [data["oil_return"].shift(lag).corr(data["xom_return"]) for lag in lags]
axes[1].bar(lags, corrs, color="steelblue")
axes[1].set_title("Oil → XOM Correlation by Lag (days)")
axes[1].set_xlabel("Oil return lagged by N days")
axes[1].set_ylabel("Correlation with XOM return")

axes[2].scatter(data["oil_return"], data["xom_return"], alpha=0.3, s=8, color="steelblue")
axes[2].set_title("Oil Daily Return vs XOM Daily Return (scatter)")
axes[2].set_xlabel("Oil return")
axes[2].set_ylabel("XOM return")

plt.tight_layout()
plt.show()
