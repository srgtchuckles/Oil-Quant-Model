# Quant Model 1

## Overview

*What is this model trying to predict or trade?*

## Hypothesis

*What market inefficiency or signal is this based on?*

## Data

- **Sources:**
- **Frequency:**
- **Universe:**

## Features / Signals

| Signal | Description | Expected Direction |
|--------|-------------|-------------------|
|        |             |                   |

## Model Architecture

*e.g., regression, ML model, rule-based, factor model*

## Backtesting

- **Period:**
- **Benchmark:**
- **Results:**

## Risk / Position Sizing

*How will positions be sized and risk managed?*

## Next Steps

1. Change the prediction horizon (biggest lever)
Next-day direction is nearly impossible to predict — even pros can't do it reliably. Predicting next-week or next-month XOM returns based on oil is much more tractable because the oil→XOM relationship operates over days, not hours.

2. Understand the relationship first, before ML
Before adding more features, look at the actual correlation structure: does oil lead XOM by 1 day? 3 days? Does it depend on oil volatility regime? A simple rolling correlation plot would tell you more than tuning a random forest.

3. Use better oil data
Daily close prices are noisy. EIA weekly crude inventory reports (released every Wednesday) are a known market-moving event — oil stocks react predictably when inventory surprises to the upside or downside. That's a real signal.

4. Crack spread
The crack spread (roughly: gasoline futures minus crude) is more directly tied to XOM's refining margins than raw oil price. It's a feature professional energy traders actually use.

I could spend a week tuning features and still get 54% accuracy on daily predictions, because the signal just isn't there at that resolution. Switching to weekly prediction + EIA inventory data is probably a better use of time than adding more ML complexity.

