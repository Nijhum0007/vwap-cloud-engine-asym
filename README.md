# VWAP Cloud Engine — Wide-Horizon Asymmetric Kronos Training (Apple Silicon M2)

This repository contains the high-performance training pipeline for the Kronos VWAP Foundation Model with:
* **Wide-Horizon 1:2 Asymmetric Labels:** **+0.30% Take Profit / -0.15% Stop Loss** over a **90-minute forward observation window**.
* **6-Feature Microstructure Matrix:** Rolling-14 VWAP, Distance to VWAP, 5-period Momentum, 20-period Volume Surge, Cumulative Volume Delta (CVD-14), and **Rolling-14 ATR (Volatility Sensor)**.
* **Regime-Conditioned Training:** Automatic shock filter that labels turbulent/flash-wick periods as Class 2 (Flat).
* **Inverse Class-Weighted Cross-Entropy:** Eliminates class imbalance bias.
* **Lookback Context:** 45-minute rolling matrix (`seq_len=45`, `in_features=6`).

Optimized specifically for Apple Silicon M-series chips (M1/M2/M3/M4) using PyTorch Metal Performance Shaders (`mps`).

---

## Quick Start (For Colleague on Apple Silicon Mac)

### 1. Pull Latest Repository & Set Up Environment

Open Terminal on your Mac:

```bash
git pull origin main
```
*(Or clone if starting fresh: `git clone https://github.com/Nijhum0007/vwap-cloud-engine-asym.git && cd vwap-cloud-engine-asym`)*

Activate your virtual environment and verify dependencies:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Run the 1-Year Wide-Horizon Training Pipeline

Run the automated training script:

```bash
python train_m2_wide_asym.py
```

### What `train_m2_wide_asym.py` does automatically:
1. **Pulls 1 Full Year of Data:** Downloads ~525,600 1-minute BTC/USDT candles from Binance (`2025-09-16 00:00 UTC` to `2026-09-16 23:59 UTC`) and caches to `btc_1m_1year_training.csv` (reuses existing cache if present).
2. **Computes 6-Feature Engine:** VWAP-14, Dist-to-VWAP, Momentum-5, Volume Surge, CVD-14, and ATR-14.
3. **Generates Regime-Conditioned 1:2 Labels:** Evaluates 90-minute forward paths (+0.30% TP / -0.15% SL) with shock-regime suppression.
4. **Hardware Acceleration:** Auto-enables Apple Silicon GPU acceleration (`device='mps'`).
5. **Trains 10 Epochs:** Optimizes the 4-layer Kronos transformer classification head on a 45-minute lookback matrix (`seq_len=45`, `in_features=6`) with AdamW and Cosine Annealing.
6. **Saves Model Weights:** Generates the trained weights file:
   ```text
   kronos_asym_wide_90m.pth
   ```
7. **Runs Holdout EV Report:** Evaluates on the 20% holdout set against the **34.67% breakeven hurdle** and prints Win Rate, Gross PnL, Fees (0.6 bps), Net EV, and Student's T-Statistic.

*Estimated Runtime on M2 Mac:* **~20 to 25 minutes**.

---

### 3. Next Step: Send the Weights File

Once training finishes, send the output file back:
```text
kronos_asym_wide_90m.pth
```
This file will be validated on the unseen Q1 2025 and October 2026 datasets before deployment into live paper trading.
