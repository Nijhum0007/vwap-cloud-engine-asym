"""
train_m2_wide_asym.py - Wide-Horizon Asymmetric 1:2 Kronos Retraining Engine
=============================================================================
Optimized specifically for Apple Silicon M-series (M1/M2/M3/M4) via PyTorch MPS.

Key Architecture Upgrades:
  1. WIDE HORIZON (Option A):
     - Take Profit: +0.30% (+30.0 bps)
     - Stop Loss:   -0.15% (-15.0 bps) [1:2 Asymmetric R:R]
     - Observation Horizon: 90 minutes
     - Fee Friction: 0.6 bps round-trip
     - Required Breakeven Win Rate: 34.67% (down from 36.00%)
  2. REGIME-AWARE FEATURE SENSOR (in_features=6):
     - Added Normalized Rolling-14 ATR as 6th input feature into Transformer.
  3. REGIME-CONDITIONED LABELING:
     - High-volatility shock candles (ATR > 5.5 bps or bar > 5.0 bps) are
       automatically labeled Class 2 (Flat), training the model to stand down in chaos.
  4. INVERSE CLASS-WEIGHTED CROSS-ENTROPY:
     - Neutralizes class imbalance to prevent bias toward the Flat class.
  5. CONTEXT MEMORY:
     - Rolling 45-minute lookback matrix (seq_len=45).

Output Model: kronos_asym_wide_90m.pth
"""

import os
import sys
import time
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Ensure Kronos directory is importable
KRONOS_DIR = os.path.join(os.path.dirname(__file__), 'Kronos')
if os.path.exists(KRONOS_DIR):
    sys.path.append(KRONOS_DIR)
try:
    from model.kronos import Kronos
except ImportError:
    print("[!] Error: Could not import Kronos. Ensure 'Kronos/' directory is present.")
    sys.exit(1)

# ==============================================================================
# HYPERPARAMETERS & TARGET GEOMETRY
# ==============================================================================
SEQ_LEN = 45           # 45-candle lookback matrix (45 mins microstructure context)
FORWARD_HORIZON = 90   # Look up to 90 candles ahead to resolve TP (+0.30%) vs SL (-0.15%)
TARGET_TP = 0.0030     # Take Profit: +0.30% (30.0 bps)
TARGET_SL = 0.0015     # Stop Loss:   -0.15% (15.0 bps) [1:2 Asymmetric R:R]
FEE_TIER = 0.00006     # 0.6 bps round-trip institutional friction

# Regime Noise Thresholds
SHOCK_ATR_THRESHOLD = 0.00055   # 5.5 bps: Market shock regime threshold
SHOCK_BAR_THRESHOLD = 0.00050   # 5.0 bps: Single expansion candle threshold

# Breakeven Win Rate Calculation: (SL + Fee) / (TP + SL)
BREAKEVEN_WR = ((TARGET_SL + FEE_TIER) / (TARGET_TP + TARGET_SL)) * 100  # 34.67%

IN_FEATURES = 6        # [Close, Dist_to_VWAP, Volume_Surge, Momentum_5, CVD_14, ATR_14]
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4

TRAIN_START = "2025-09-16 00:00:00"
TRAIN_END = "2026-09-16 23:59:00"
CACHE_FILE = "btc_1m_1year_training.csv"
OUTPUT_WEIGHTS = "kronos_asym_wide_90m.pth"


# ==============================================================================
# HARDWARE ACCELERATION SELECTION (APPLE SILICON MPS / CUDA / CPU)
# ==============================================================================
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("[*] Device: Apple Silicon GPU (MPS) detected! Metal acceleration active.")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[*] Device: NVIDIA CUDA GPU detected ({torch.cuda.get_device_name(0)}).")
    else:
        device = torch.device('cpu')
        print("[*] Device: Standard CPU.")
    return device


# ==============================================================================
# DATA INGESTION & 6-FEATURE ENGINE
# ==============================================================================
def fetch_or_load_1year_data(symbol="BTCUSDT"):
    if os.path.exists(CACHE_FILE):
        print(f"[*] Found cached 1-year training dataset: {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE)
        
        # Verify if ATR_14 is already computed in the cache
        if 'ATR_14' in df.columns or 'atr_14' in df.columns:
            print(f"[*] Loaded {len(df):,} rows with full 6-feature pipeline.")
            return df
        else:
            print("[*] Cached dataset needs ATR_14 feature. Computing now...")
            highs = df['high'] if 'high' in df.columns else df['High']
            lows = df['low'] if 'low' in df.columns else df['Low']
            closes = df['close'] if 'close' in df.columns else df['Close']
            tr = (highs - lows) / closes
            df['atr_14'] = tr.rolling(14, min_periods=1).mean()
            df.to_csv(CACHE_FILE, index=False)
            return df

    print(f"\n[*] Fetching 1 full year of 1-minute candle data from Binance ({symbol})...")
    print(f"    Timeline: {TRAIN_START} UTC to {TRAIN_END} UTC (~525,600 candles)")
    
    start_ts = int(pd.to_datetime(TRAIN_START).tz_localize('UTC').timestamp() * 1000)
    end_ts = int(pd.to_datetime(TRAIN_END).tz_localize('UTC').timestamp() * 1000)
    
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    current_start = start_ts
    
    session = requests.Session()
    session.headers.update({"User-Agent": "VWAP2-WideHorizonTraining/1.0"})
    
    pbar = tqdm(total=525600, desc="Downloading Binance 1m candles", unit=" candle")
    while True:
        params = {
            "symbol": symbol,
            "interval": "1m",
            "limit": 1000,
            "startTime": current_start,
            "endTime": end_ts
        }
        
        try:
            resp = session.get(url, params=params, timeout=12)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            time.sleep(2)
            continue
            
        if not data:
            break
            
        all_klines.extend(data)
        pbar.update(len(data))
        
        last_close_time = data[-1][0]
        current_start = last_close_time + 60000
        
        if current_start >= end_ts or len(data) < 1000:
            break
            
        time.sleep(0.04)
        
    pbar.close()
    
    columns = [
        "timestamp", "Open", "High", "Low", "Close", "Volume", 
        "close_time", "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore"
    ]
    df = pd.DataFrame(all_klines, columns=columns)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    for c in ["Open", "High", "Low", "Close", "Volume", "tb_base_av"]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    df.sort_values('date', inplace=True)
    df.drop_duplicates(subset=['date'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print("\n[*] Computing 6-Feature Microstructure Pipeline...")
    # 1. Rolling-14 VWAP & Dist-to-VWAP
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['Typical_Price'] * df['Volume']
    df['VWAP'] = df['VP'].rolling(window=14).sum() / df['Volume'].rolling(window=14).sum()
    df['Dist_to_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']
    
    # 2. 5-period Momentum
    df['Momentum_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
    
    # 3. 20-period Volume Surge
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Surge'] = df['Volume'] / df['Volume_MA_20']
    
    # 4. Cumulative Volume Delta (CVD-14)
    df['Taker_Sell_Volume'] = df['Volume'] - df['tb_base_av']
    df['Net_Volume'] = df['tb_base_av'] - df['Taker_Sell_Volume']
    df['CVD_14'] = df['Net_Volume'].rolling(window=14).sum()
    
    # 5. Rolling-14 ATR (Feature #6: Volatility Sensor)
    tr = (df['High'] - df['Low']) / df['Close']
    df['ATR_14'] = tr.rolling(window=14, min_periods=1).mean()
    
    # Format standard column names
    df['open'] = df['Open']
    df['high'] = df['High']
    df['low'] = df['Low']
    df['close'] = df['Close']
    df['volume'] = df['Volume']
    df['dist_to_vwap'] = df['Dist_to_VWAP']
    df['volume_surge'] = df['Volume_Surge']
    df['momentum_5'] = df['Momentum_5']
    df['cvd_14'] = df['CVD_14']
    df['atr_14'] = df['ATR_14']
    
    drop_cols = ['Typical_Price', 'VP', 'Volume_MA_20', 'tb_base_av', 'Taker_Sell_Volume', 'Net_Volume', 'timestamp', 'close_time', 'quote_av', 'trades', 'tb_quote_av', 'ignore']
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df.to_csv(CACHE_FILE, index=False)
    print(f"[*] Saved {len(df):,} cleaned rows with 6 features to {CACHE_FILE}.")
    return df


# ==============================================================================
# DATASET BUILDER WITH REGIME-CONDITIONED ASYMMETRIC LABELS
# ==============================================================================
class WideHorizonAsymmetricDataset(Dataset):
    def __init__(self, df, seq_len=SEQ_LEN, horizon=FORWARD_HORIZON, tp=TARGET_TP, sl=TARGET_SL):
        self.seq_len = seq_len
        self.horizon = horizon
        self.tp = tp
        self.sl = sl
        
        self.prices = df['close'].values.astype(np.float32)
        self.highs = df['high'].values.astype(np.float32)
        self.lows = df['low'].values.astype(np.float32)
        self.dist_vwap = df['dist_to_vwap'].values.astype(np.float32)
        self.vol_surge = df['volume_surge'].values.astype(np.float32)
        self.momentum = df['momentum_5'].values.astype(np.float32)
        self.cvd = df['cvd_14'].values.astype(np.float32)
        self.atr = df['atr_14'].values.astype(np.float32)
        
        self.labels = self._generate_regime_conditioned_labels()
        
        self.valid_indices = [
            i for i in range(seq_len, len(self.prices) - horizon)
            if self.labels[i] != -1
        ]
        
    def _generate_regime_conditioned_labels(self):
        print(f"\n[*] Generating Wide-Horizon 1:2 Labels (+{self.tp*100:.2f}% TP / -{self.sl*100:.2f}% SL over {self.horizon} mins)...")
        print(f"    Active Regime Gate: Shocks (ATR > {SHOCK_ATR_THRESHOLD*10000:.1f} bps or Bar > {SHOCK_BAR_THRESHOLD*10000:.1f} bps) -> Class 2 (Flat)")
        
        n = len(self.prices)
        labels = np.full(n, -1, dtype=np.int64)
        shock_count = 0
        
        for i in tqdm(range(n - self.horizon), desc="Labeling bars", unit=" bar"):
            entry = self.prices[i]
            cur_atr = self.atr[i]
            cur_bar_range = (self.highs[i] - self.lows[i]) / entry
            
            # METHOD A: Regime-Conditioned Override
            # If the market is in a violent shock regime, do NOT attempt directional 1:2 prediction.
            # Force Class 2 (Flat), teaching the neural net to stand down during turbulence.
            if cur_atr > SHOCK_ATR_THRESHOLD or cur_bar_range > SHOCK_BAR_THRESHOLD:
                labels[i] = 2
                shock_count += 1
                continue
            
            # Non-shock regime: evaluate forward 1:2 trajectory
            long_tp = entry * (1.0 + self.tp)
            long_sl = entry * (1.0 - self.sl)
            short_tp = entry * (1.0 - self.tp)
            short_sl = entry * (1.0 + self.sl)
            
            long_win = False
            short_win = False
            
            # Long trajectory (pessimistic routing: SL checked first)
            for j in range(i + 1, i + self.horizon):
                if self.lows[j] <= long_sl:
                    break
                if self.highs[j] >= long_tp:
                    long_win = True
                    break
                    
            # Short trajectory
            for j in range(i + 1, i + self.horizon):
                if self.highs[j] >= short_sl:
                    break
                if self.lows[j] <= short_tp:
                    short_win = True
                    break
                    
            if long_win and not short_win:
                labels[i] = 0   # Class 0: Long
            elif short_win and not long_win:
                labels[i] = 1   # Class 1: Short
            else:
                labels[i] = 2   # Class 2: Flat / No clean impulse
                
        # Compute distribution
        valid_labels = labels[labels != -1]
        counts = pd.Series(valid_labels).value_counts().to_dict()
        total_valid = len(valid_labels)
        
        c0 = counts.get(0, 0)
        c1 = counts.get(1, 0)
        c2 = counts.get(2, 0)
        
        print(f"\n[*] Label Distribution (Total Valid: {total_valid:,}):")
        print(f"    Class 0 (Long) : {c0:,} ({c0/total_valid*100:.2f}%)")
        print(f"    Class 1 (Short): {c1:,} ({c1/total_valid*100:.2f}%)")
        print(f"    Class 2 (Flat) : {c2:,} ({c2/total_valid*100:.2f}%) [Shocks forced: {shock_count:,}]")
        
        # Save class weights for inverse cross-entropy weighting
        self.class_weights = torch.tensor([
            total_valid / (3.0 * max(c0, 1)),
            total_valid / (3.0 * max(c1, 1)),
            total_valid / (3.0 * max(c2, 1))
        ], dtype=torch.float32)
        print(f"[*] Inverse Class Weights: Long={self.class_weights[0]:.2f}, Short={self.class_weights[1]:.2f}, Flat={self.class_weights[2]:.2f}")
        
        return labels

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        
        # Extract 45-candle lookback sequence
        p_seq = self.prices[real_idx - self.seq_len : real_idx]
        v_seq = self.dist_vwap[real_idx - self.seq_len : real_idx]
        vol_seq = self.vol_surge[real_idx - self.seq_len : real_idx]
        mom_seq = self.momentum[real_idx - self.seq_len : real_idx]
        cvd_seq = self.cvd[real_idx - self.seq_len : real_idx]
        atr_seq = self.atr[real_idx - self.seq_len : real_idx]
        
        # Local Z-Score normalization per window across all 6 features
        p_norm = (p_seq - np.mean(p_seq)) / (np.std(p_seq) + 1e-5)
        v_norm = (v_seq - np.mean(v_seq)) / (np.std(v_seq) + 1e-5)
        vol_norm = (vol_seq - np.mean(vol_seq)) / (np.std(vol_seq) + 1e-5)
        mom_norm = (mom_seq - np.mean(mom_seq)) / (np.std(mom_seq) + 1e-5)
        cvd_norm = (cvd_seq - np.mean(cvd_seq)) / (np.std(cvd_seq) + 1e-5)
        atr_norm = (atr_seq - np.mean(atr_seq)) / (np.std(atr_seq) + 1e-5)
        
        # Stack into [45, 6] Matrix
        matrix = np.column_stack((p_norm, v_norm, vol_norm, mom_norm, cvd_norm, atr_norm))
        tensor = torch.tensor(matrix, dtype=torch.float32)
        label = torch.tensor(self.labels[real_idx], dtype=torch.long)
        
        return tensor, label


# ==============================================================================
# HOLDOUT EVALUATION HARNESS
# ==============================================================================
def evaluate_holdout(model, df_holdout, device, threshold=0.45):
    print("\n[*] Running Fast Forward Evaluation on 20% Unseen Holdout Set...")
    model.eval()
    
    prices = df_holdout['close'].values.astype(np.float32)
    highs = df_holdout['high'].values.astype(np.float32)
    lows = df_holdout['low'].values.astype(np.float32)
    dist_vwap = df_holdout['dist_to_vwap'].values.astype(np.float32)
    vol_surge = df_holdout['volume_surge'].values.astype(np.float32)
    momentum = df_holdout['momentum_5'].values.astype(np.float32)
    cvd = df_holdout['cvd_14'].values.astype(np.float32)
    atr = df_holdout['atr_14'].values.astype(np.float32)
    
    n = len(prices)
    in_pos = False
    pos_type = None
    entry_p = 0.0
    entry_idx = 0
    
    wins = 0
    losses = 0
    gross_pnl = 0.0
    total_fees = 0.0
    trade_returns = []
    
    for i in tqdm(range(SEQ_LEN, n - 1), desc="Holdout Replay", unit=" bar"):
        if in_pos:
            is_entry = (i == entry_idx)
            c_high = prices[i] if is_entry else highs[i]
            c_low = prices[i] if is_entry else lows[i]
            
            exit_trig = False
            pnl = 0.0
            
            if pos_type == 'LONG':
                if c_low <= entry_p * (1.0 - TARGET_SL):
                    pnl = -TARGET_SL
                    exit_trig = True
                    losses += 1
                elif c_high >= entry_p * (1.0 + TARGET_TP):
                    pnl = TARGET_TP
                    exit_trig = True
                    wins += 1
            elif pos_type == 'SHORT':
                if c_high >= entry_p * (1.0 + TARGET_SL):
                    pnl = -TARGET_SL
                    exit_trig = True
                    losses += 1
                elif c_low <= entry_p * (1.0 - TARGET_TP):
                    pnl = TARGET_TP
                    exit_trig = True
                    wins += 1
                    
            if exit_trig:
                net_ret = pnl - FEE_TIER
                trade_returns.append(net_ret)
                gross_pnl += pnl
                total_fees += FEE_TIER
                in_pos = False
                pos_type = None
                
        if not in_pos:
            p_seq = prices[i - SEQ_LEN : i]
            v_seq = dist_vwap[i - SEQ_LEN : i]
            vol_seq = vol_surge[i - SEQ_LEN : i]
            mom_seq = momentum[i - SEQ_LEN : i]
            cvd_seq = cvd[i - SEQ_LEN : i]
            atr_seq = atr[i - SEQ_LEN : i]
            
            p_norm = (p_seq - np.mean(p_seq)) / (np.std(p_seq) + 1e-5)
            v_norm = (v_seq - np.mean(v_seq)) / (np.std(v_seq) + 1e-5)
            vol_norm = (vol_seq - np.mean(vol_seq)) / (np.std(vol_seq) + 1e-5)
            mom_norm = (mom_seq - np.mean(mom_seq)) / (np.std(mom_seq) + 1e-5)
            cvd_norm = (cvd_seq - np.mean(cvd_seq)) / (np.std(cvd_seq) + 1e-5)
            atr_norm = (atr_seq - np.mean(atr_seq)) / (np.std(atr_seq) + 1e-5)
            
            matrix = np.column_stack((p_norm, v_norm, vol_norm, mom_norm, cvd_norm, atr_norm))
            seq_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model.classify_continuous(seq_tensor).cpu().numpy()[0]
                
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            # Execution check with threshold
            if probs[0] > threshold and (probs[0] - probs[1]) > 0.10:
                in_pos = True
                pos_type = 'LONG'
                entry_p = prices[i]
                entry_idx = i
            elif probs[1] > threshold and (probs[1] - probs[0]) > 0.10:
                in_pos = True
                pos_type = 'SHORT'
                entry_p = prices[i]
                entry_idx = i
                
    closed = wins + losses
    wr = (wins / closed * 100.0) if closed > 0 else 0.0
    net_ev = gross_pnl - total_fees
    
    t_stat = 0.0
    if closed >= 2:
        mean_r = np.mean(trade_returns)
        std_r = np.std(trade_returns, ddof=1)
        if std_r > 1e-9:
            t_stat = mean_r / (std_r / np.sqrt(closed))
            
    print("\n" + "=" * 65)
    print("      WIDE-HORIZON ASYMMETRIC HOLDOUT REPORT (1:2 R:R)      ")
    print("=" * 65)
    print(f"Total Completed Trades : {closed}")
    print(f"Wins / Losses          : {wins} / {losses}")
    print(f"Observed Win Rate      : {wr:.2f}% (Breakeven Hurdle: {BREAKEVEN_WR:.2f}%)")
    print(f"Statistical Edge       : {wr - BREAKEVEN_WR:+.2f}%")
    print(f"Gross PnL (pre-fee)    : {gross_pnl * 100:+.2f}%")
    print(f"Total Fees (0.6 bps)   : -{total_fees * 100:.2f}%")
    print(f"Net Expected Value     : {net_ev * 100:+.2f}%")
    print(f"Student's T-Statistic  : {t_stat:+.2f}")
    print("=" * 65 + "\n")


# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def main():
    print("=" * 70)
    print("  KRONOS WIDE-HORIZON ASYMMETRIC 1:2 RETRAINING (APPLE SILICON M2) ")
    print("=" * 70)
    device = get_device()
    
    # 1. Ingest Data
    df = fetch_or_load_1year_data()
    
    # 2. 80/20 Train/Holdout Split
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_holdout = df.iloc[split_idx:].copy().reset_index(drop=True)
    print(f"[*] Split complete: {len(df_train):,} Train bars | {len(df_holdout):,} Holdout bars")
    
    # 3. Create Dataset & DataLoader
    train_dataset = WideHorizonAsymmetricDataset(df_train, seq_len=SEQ_LEN, horizon=FORWARD_HORIZON)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 4. Initialize Kronos Model with 6 input features
    print(f"\n[*] Initializing Kronos Transformer Model (seq_len={SEQ_LEN}, in_features={IN_FEATURES})...")
    model = Kronos(
        s1_bits=8, s2_bits=8, n_layers=4, d_model=128, n_heads=4, 
        ff_dim=512, ffn_dropout_p=0.1, attn_dropout_p=0.1, 
        resid_dropout_p=0.1, token_dropout_p=0.1, learn_te=False,
        in_features=IN_FEATURES
    ).to(device)
    
    # Weighted Cross-Entropy Loss to handle class imbalance
    class_weights = train_dataset.class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # 5. Training Loop
    print(f"\n[*] Starting {EPOCHS} Training Epochs on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for seq, labels in tqdm(dataloader, desc=f"Epoch {epoch+1:02d}/{EPOCHS:02d}", unit="batch"):
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model.classify_continuous(seq)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        scheduler.step()
        epoch_time = time.time() - start_time
        epoch_acc = 100.0 * correct / total
        avg_loss = total_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"    Epoch {epoch+1:02d}/{EPOCHS:02d} | Loss: {avg_loss:.4f} | Acc: {epoch_acc:.2f}% | LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
        
    # 6. Save Model Weights
    torch.save(model.state_dict(), OUTPUT_WEIGHTS)
    print(f"\n[+] Successfully trained and saved model weights to: '{OUTPUT_WEIGHTS}'")
    
    # 7. Evaluate on Holdout
    evaluate_holdout(model, df_holdout, device, threshold=0.45)
    
    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETE! SEND '{OUTPUT_WEIGHTS}' BACK TO WINDOWS PC ")
    print("=" * 70)


if __name__ == "__main__":
    main()

