from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import yfinance as yf

oil = yf.download("CL=F", period="2y")
xom = yf.download("XOM", period="2y")

data = pd.DataFrame()
data["oil_close"] = oil["Close"]
data["xom_close"] = xom["Close"]

data["xom_future_return"] = data["xom_close"].pct_change().shift(-1)

# Oil-only features — question is "how does oil move XOM?", not "predict XOM from itself"
data["oil_return"] = data["oil_close"].pct_change()
data["oil_return_lag1"] = data["oil_return"].shift(1)
data["oil_return_lag2"] = data["oil_return"].shift(2)
data["oil_ma5"] = data["oil_close"].rolling(5).mean()
data["oil_momentum"] = data["oil_close"] / data["oil_ma5"]
data["oil_vol5"] = data["oil_return"].rolling(5).std()

features = ["oil_return", "oil_return_lag1", "oil_return_lag2", "oil_momentum", "oil_vol5"]

data_clean = data[features + ["xom_future_return"]].dropna()
X = data_clean[features]
y = (data_clean["xom_future_return"] > 0).astype(int)

# Time-based split — past trains, future validates (random split leaks future info)
split = int(len(X) * 0.75)
train_X, val_X = X.iloc[:split], X.iloc[split:]
train_y, val_y = y.iloc[:split], y.iloc[split:]

model = RandomForestClassifier(n_estimators=100, random_state=1)
model.fit(train_X, train_y)

val_preds = model.predict(val_X)
print("Accuracy: {:.1%}".format(accuracy_score(val_y, val_preds)))

# Financial backtest
results = data_clean.iloc[split:].copy()
results["predicted"] = val_preds
results["strategy_return"] = results["predicted"] * results["xom_future_return"]
results["buyhold_return"] = results["xom_future_return"]

results["strategy_cumret"] = (1 + results["strategy_return"]).cumprod()
results["buyhold_cumret"] = (1 + results["buyhold_return"]).cumprod()

strategy_total = results["strategy_cumret"].iloc[-1] - 1
buyhold_total = results["buyhold_cumret"].iloc[-1] - 1

sharpe = results["strategy_return"].mean() / results["strategy_return"].std() * np.sqrt(252)

rolling_max = results["strategy_cumret"].cummax()
max_drawdown = ((results["strategy_cumret"] - rolling_max) / rolling_max).min()

in_market = results[results["predicted"] == 1]
win_rate = (in_market["strategy_return"] > 0).mean()

print("\n--- Financial Performance (validation period) ---")
print(f"Strategy return:    {strategy_total:+.1%}")
print(f"Buy & hold return:  {buyhold_total:+.1%}")
print(f"Sharpe ratio:       {sharpe:.2f}")
print(f"Max drawdown:       {max_drawdown:.1%}")
print(f"Win rate (days held): {win_rate:.1%}")
print(f"Days in market: {int(results['predicted'].sum())} / {len(results)}")

print("\nMonthly strategy returns:")
monthly = results["strategy_return"].resample("ME").sum()
for date, ret in monthly.items():
    print(f"  {date.strftime('%b %Y')}: {ret:+.1%}")
