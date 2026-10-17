# VWAP Cloud Engine — Asymmetric Kronos Training (M2 / Apple Silicon)

This repository contains the training pipeline for the Kronos VWAP Foundation Model with **Asymmetric 1:2 Risk-to-Reward labels (+0.15% Take Profit / -0.075% Stop Loss)** and a **Rolling 30-Candle Matrix**.

Optimized specifically for Apple Silicon M-series chips (M1/M2/M3/M4) using PyTorch Metal Performance Shaders (`mps`).

---

## Quick Start (For Colleague on Apple Silicon Mac)

### 1. Clone & Set Up Environment

Open Terminal on your Mac:

```bash
git clone https://github.com/Nijhum0007/vwap-cloud-engine-asym.git
cd vwap-cloud-engine-asym
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Run the 1-Year Training Pipeline

Run the automated training script:

```bash
python train_m2_asym.py
```

### What `train_m2_asym.py` does automatically:
1. **Pulls 1 Full Year of Data:** Downloads ~525,600 1-minute BTC/USDT candles from Binance (`2025-09-16 00:00 UTC` to `2026-09-16 23:59 UTC`) and caches to `btc_1m_1year_training.csv`.
2. **Computes Features:** Calculates Rolling-14 VWAP, Distance to VWAP, 5-period Momentum, 20-period Volume Surge, and Cumulative Volume Delta (CVD-14).
3. **Generates Asymmetric Labels:** Evaluates forward paths under 1:2 R:R (+0.15% TP / -0.075% SL) over a 45-minute horizon.
4. **Hardware Acceleration:** Auto-enables Apple Silicon GPU acceleration (`device='mps'`).
5. **Trains 10 Epochs:** Optimizes the 4-layer Kronos transformer classification head on a 30-minute lookback matrix (`seq_len=30`).
6. **Saves Model Weights:** Generates the trained weights file:
   ```text
   kronos_asym_30m.pth
   ```
7. **Runs Holdout EV Report:** Evaluates on the 20% holdout set and prints Win Rate, Gross PnL, Fees (0.6 bps), Net EV, and Student's T-Statistic.

*Estimated Runtime on M2 Mac:* **~15 to 25 minutes**.

---

### 3. Next Step: Send the Weights File

Once training finishes, send the output file back:
```text
kronos_asym_30m.pth
```
This file will be validated on the 30-day unseen out-of-sample dataset (`Sep 18, 2026` to `Oct 17, 2026`).
