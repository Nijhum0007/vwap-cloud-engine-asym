"""
train_m2_asym.py - High-Performance Asymmetric Kronos Training Pipeline
=======================================================================
Optimized for Apple Silicon M2 (MPS GPU acceleration).
Automatically pulls 1 year of 1-minute historical data (Sep 16, 2025 - Sep 16, 2026),
computes quantitative features, generates 1:2 Asymmetric Labels (+0.15% TP / -0.075% SL),
and trains the Kronos classification head with a rolling 30-candle sequence length.

Output Model: kronos_asym_30m.pth
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
    print("[!] Error: Could not import Kronos. Ensure 'Kronos/' directory is in the repository.")
    sys.exit(1)

# ==============================================================================
# HYPERPARAMETERS & GEOMETRIC TARGETS
# ==============================================================================
SEQ_LEN = 30           # 30-candle lookback matrix (30 mins of microstructure memory)
FORWARD_HORIZON = 45   # Look up to 45 candles ahead to resolve TP vs SL
TARGET_TP = 0.0015     # Take Profit: +0.15% (15 bps)
TARGET_SL = 0.00075    # Stop Loss:   -0.075% (7.5 bps) [1:2 Asymmetric R:R]
FEE_TIER = 0.00006     # 0.6 bps round-trip institutional friction

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4

TRAIN_START = "2025-09-16 00:00:00"
TRAIN_END = "2026-09-16 23:59:00"
CACHE_FILE = "btc_1m_1year_training.csv"

# ==============================================================================
# HARDWARE ACCELERATION SELECTION (APPLE SILICON MPS / CUDA / CPU)
# ==============================================================================
def get_device():
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print("[*] Device: Apple Silicon GPU (MPS) detected! High-speed training enabled.")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[*] Device: NVIDIA CUDA GPU detected ({torch.cuda.get_device_name(0)}).")
    else:
        device = torch.device('cpu')
        print("[*] Device: CPU (MPS/CUDA not available, running on standard CPU).")
    return device

# ==============================================================================
# DATA INGESTION & VWAP / CVD FEATURE ENGINE
# ==============================================================================
def fetch_or_load_1year_data(symbol="BTCUSDT"):
    if os.path.exists(CACHE_FILE):
        print(f"[*] Found cached 1-year training dataset: {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE)
        print(f"[*] Loaded {len(df):,} feature rows.")
        return df

    print(f"\n[*] Fetching 1 full year of 1-minute candle data from Binance ({symbol})...")
    print(f"    Timeline: {TRAIN_START} UTC to {TRAIN_END} UTC")
    
    start_ts = int(pd.to_datetime(TRAIN_START).tz_localize('UTC').timestamp() * 1000)
    end_ts = int(pd.to_datetime(TRAIN_END).tz_localize('UTC').timestamp() * 1000)
    
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    current_start = start_ts
    
    session = requests.Session()
    session.headers.update({"User-Agent": "VWAP2-M2Training/1.0"})
    
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
        except Exception as e:
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
            
        time.sleep(0.05) # Polite rate-limit pause
        
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
    
    print("[*] Computing Quantitative VWAP, CVD, Momentum, and Volume Surge features...")
    # 1. Rolling-14 VWAP
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['Typical_Price'] * df['Volume']
    df['VWAP'] = df['VP'].rolling(window=14).sum() / df['Volume'].rolling(window=14).sum()
    
    # 2. Distance to VWAP
    df['Dist_to_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']
    
    # 3. Momentum (5-period Rate of Change)
    df['Momentum_5'] = df['Close'].pct_change(periods=5)
    
    # 4. Volume Surge
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Surge'] = df['Volume'] / df['Volume_MA_20']
    
    # 5. Cumulative Volume Delta (CVD-14)
    df['Taker_Sell_Volume'] = df['Volume'] - df['tb_base_av']
    df['Net_Volume'] = df['tb_base_av'] - df['Taker_Sell_Volume']
    df['CVD_14'] = df['Net_Volume'].rolling(window=14).sum()
    
    df['open'] = df['Open']
    df['high'] = df['High']
    df['low'] = df['Low']
    df['close'] = df['Close']
    df['volume'] = df['Volume']
    df['dist_to_vwap'] = df['Dist_to_VWAP']
    df['volume_surge'] = df['Volume_Surge']
    df['momentum_5'] = df['Momentum_5']
    df['cvd_14'] = df['CVD_14']
    
    df.drop(columns=['Typical_Price', 'VP', 'Volume_MA_20', 'tb_base_av', 'Taker_Sell_Volume', 'Net_Volume', 'timestamp', 'close_time', 'quote_av', 'trades', 'tb_quote_av', 'ignore'], inplace=True, errors='ignore')
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df.to_csv(CACHE_FILE, index=False)
    print(f"[*] Saved {len(df):,} cleaned rows to {CACHE_FILE}.")
    return df

# ==============================================================================
# ASYMMETRIC 1:2 R:R DATASET BUILDER
# ==============================================================================
class AsymmetricVWAPDataset(Dataset):
    def __init__(self, df, seq_len=SEQ_LEN, horizon=FORWARD_HORIZON, tp=TARGET_TP, sl=TARGET_SL):
        self.seq_len = seq_len
        self.horizon = horizon
        self.tp = tp
        self.sl = sl
        
        self.prices = df['close'].values
        self.highs = df['high'].values
        self.lows = df['low'].values
        self.dist_vwap = df['dist_to_vwap'].values
        self.vol_surge = df['volume_surge'].values
        self.momentum = df['momentum_5'].values
        self.cvd = df['cvd_14'].values
        
        self.labels = self._generate_asymmetric_labels()
        
        self.valid_indices = [
            i for i in range(seq_len, len(self.prices) - horizon)
            if self.labels[i] != -1
        ]
        
    def _generate_asymmetric_labels(self):
        print("\n[*] Generating Asymmetric 1:2 R:R Labels (+0.15% TP / -0.075% SL)...")
        labels = np.full(len(self.prices), -1)
        
        for i in tqdm(range(len(self.prices) - self.horizon), desc="Labeling bars", unit=" bar"):
            entry = self.prices[i]
            
            # Asymmetric barriers
            long_tp = entry * (1 + self.tp)
            long_sl = entry * (1 - self.sl)
            short_tp = entry * (1 - self.tp)
            short_sl = entry * (1 + self.sl)
            
            long_win = False
            short_win = False
            
            # Check Long trajectory
            for j in range(i + 1, i + self.horizon):
                if self.lows[j] <= long_sl:
                    break
                if self.highs[j] >= long_tp:
                    long_win = True
                    break
                    
            # Check Short trajectory
            for j in range(i + 1, i + self.horizon):
                if self.highs[j] >= short_sl:
                    break
                if self.lows[j] <= short_tp:
                    short_win = True
                    break
                    
            if long_win and not short_win:
                labels[i] = 0 # Class 0: Long
            elif short_win and not long_win:
                labels[i] = 1 # Class 1: Short
            else:
                labels[i] = 2 # Class 2: Flat / No clean impulse
                
        # Count distribution
        counts = pd.Series(labels[labels != -1]).value_counts().to_dict()
        total_valid = sum(counts.values())
        print(f"[*] Label Distribution (Total: {total_valid:,}):")
        print(f"    Class 0 (Long) : {counts.get(0, 0):,} ({counts.get(0, 0)/total_valid*100:.1f}%)")
        print(f"    Class 1 (Short): {counts.get(1, 0):,} ({counts.get(1, 0)/total_valid*100:.1f}%)")
        print(f"    Class 2 (Flat) : {counts.get(2, 0):,} ({counts.get(2, 0)/total_valid*100:.1f}%)")
        return labels

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        
        p_seq = self.prices[real_idx - self.seq_len : real_idx]
        v_seq = self.dist_vwap[real_idx - self.seq_len : real_idx]
        vol_seq = self.vol_surge[real_idx - self.seq_len : real_idx]
        mom_seq = self.momentum[real_idx - self.seq_len : real_idx]
        cvd_seq = self.cvd[real_idx - self.seq_len : real_idx]
        
        # Local Z-score normalization across 30 candles
        p_seq = (p_seq - np.mean(p_seq)) / (np.std(p_seq) + 1e-5)
        v_seq = (v_seq - np.mean(v_seq)) / (np.std(v_seq) + 1e-5)
        vol_seq = (vol_seq - np.mean(vol_seq)) / (np.std(vol_seq) + 1e-5)
        mom_seq = (mom_seq - np.mean(mom_seq)) / (np.std(mom_seq) + 1e-5)
        cvd_seq = (cvd_seq - np.mean(cvd_seq)) / (np.std(cvd_seq) + 1e-5)
        
        seq = np.column_stack((p_seq, v_seq, vol_seq, mom_seq, cvd_seq))
        label = self.labels[real_idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# ==============================================================================
# HOLDOUT EV EVALUATION ENGINE
# ==============================================================================
def evaluate_holdout(model, df_holdout, device, threshold=0.40):
    print(f"\n[*] Evaluating Holdout Performance on {len(df_holdout):,} candles (Threshold: {threshold*100:.1f}%)...")
    model.eval()
    
    prices = df_holdout['close'].values
    highs = df_holdout['high'].values
    lows = df_holdout['low'].values
    dist_vwap = df_holdout['dist_to_vwap'].values
    vol_surge = df_holdout['volume_surge'].values
    momentum = df_holdout['momentum_5'].values
    cvd = df_holdout['cvd_14'].values
    
    in_position = False
    position_type = None
    entry_price = 0.0
    entry_idx = 0
    
    total_trades = 0
    wins = 0
    losses = 0
    gross_pnl = 0.0
    total_fees = 0.0
    trade_returns = []
    
    for i in tqdm(range(SEQ_LEN, len(df_holdout)), desc="Holdout Replay", unit=" bar"):
        if in_position:
            is_entry = (i == entry_idx)
            cur_high = prices[i] if is_entry else highs[i]
            cur_low = prices[i] if is_entry else lows[i]
            
            exit_trig = False
            pnl = 0.0
            
            if position_type == 'LONG':
                if cur_low <= entry_price * (1 - TARGET_SL):
                    pnl = -TARGET_SL
                    exit_trig = True
                    losses += 1
                elif cur_high >= entry_price * (1 + TARGET_TP):
                    pnl = TARGET_TP
                    exit_trig = True
                    wins += 1
            elif position_type == 'SHORT':
                if cur_high >= entry_price * (1 + TARGET_SL):
                    pnl = -TARGET_SL
                    exit_trig = True
                    losses += 1
                elif cur_low <= entry_price * (1 - TARGET_TP):
                    pnl = TARGET_TP
                    exit_trig = True
                    wins += 1
                    
            if exit_trig:
                net_ret = pnl - FEE_TIER
                trade_returns.append(net_ret)
                gross_pnl += pnl
                total_fees += FEE_TIER
                in_position = False
                position_type = None
                
        if not in_position:
            p_seq = prices[i - SEQ_LEN : i]
            v_seq = dist_vwap[i - SEQ_LEN : i]
            vol_seq = vol_surge[i - SEQ_LEN : i]
            mom_seq = momentum[i - SEQ_LEN : i]
            cvd_seq = cvd[i - SEQ_LEN : i]
            
            p_seq = (p_seq - np.mean(p_seq)) / (np.std(p_seq) + 1e-5)
            v_seq = (v_seq - np.mean(v_seq)) / (np.std(v_seq) + 1e-5)
            vol_seq = (vol_seq - np.mean(vol_seq)) / (np.std(vol_seq) + 1e-5)
            mom_seq = (mom_seq - np.mean(mom_seq)) / (np.std(mom_seq) + 1e-5)
            cvd_seq = (cvd_seq - np.mean(cvd_seq)) / (np.std(cvd_seq) + 1e-5)
            
            matrix = np.column_stack((p_seq, v_seq, vol_seq, mom_seq, cvd_seq))
            seq_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model.classify_continuous(seq_tensor).cpu().numpy()[0]
                
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            if probs[0] > threshold:
                in_position = True
                position_type = 'LONG'
                entry_price = prices[i]
                entry_idx = i
                total_trades += 1
            elif probs[1] > threshold:
                in_position = True
                position_type = 'SHORT'
                entry_price = prices[i]
                entry_idx = i
                total_trades += 1
                
    closed = wins + losses
    wr = (wins / closed * 100) if closed > 0 else 0.0
    net_ev = gross_pnl - total_fees
    
    t_stat = 0.0
    if closed >= 2:
        mean_r = np.mean(trade_returns)
        std_r = np.std(trade_returns, ddof=1)
        if std_r > 1e-9:
            t_stat = mean_r / (std_r / (closed ** 0.5))
            
    print("\n" + "=" * 50)
    print("      M2 ASYMMETRIC HOLDOUT REPORT (1:2 R:R)      ")
    print("=" * 50)
    print(f"Total Completed Trades : {closed}")
    print(f"Wins / Losses          : {wins} / {losses}")
    print(f"Observed Win Rate      : {wr:.2f}% (Breakeven: 36.00%)")
    print(f"Gross PnL (pre-fee)    : {gross_pnl * 100:+.3f}%")
    print(f"Total Fees (0.6 bps)   : -{total_fees * 100:.3f}%")
    print(f"Net Expected Value     : {net_ev * 100:+.3f}%")
    print(f"Student's T-Statistic  : {t_stat:+.2f}")
    print("=" * 50 + "\n")

# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================
def main():
    print("=" * 65)
    print("  KRONOS ASYMMETRIC 1:2 R:R RETRAINING ENGINE (APPLE SILICON M2) ")
    print("=" * 65)
    device = get_device()
    
    # 1. Ingest Data
    df = fetch_or_load_1year_data()
    
    # 2. 80/20 Train/Holdout Split
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy().reset_index(drop=True)
    df_holdout = df.iloc[split_idx:].copy().reset_index(drop=True)
    print(f"[*] Split complete: {len(df_train):,} Train bars | {len(df_holdout):,} Holdout bars")
    
    # 3. Create Dataset & DataLoader
    train_dataset = AsymmetricVWAPDataset(df_train, seq_len=SEQ_LEN, horizon=FORWARD_HORIZON)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    # 4. Initialize Kronos Model
    print("\n[*] Initializing Kronos Transformer Model (seq_len=30, in_features=5)...")
    model = Kronos(
        s1_bits=8, s2_bits=8, n_layers=4, d_model=128, n_heads=4, 
        ff_dim=512, ffn_dropout_p=0.1, attn_dropout_p=0.1, 
        resid_dropout_p=0.1, token_dropout_p=0.1, learn_te=False,
        in_features=5
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. Training Loop
    print(f"[*] Starting {EPOCHS} Training Epochs on {device}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for seq, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", unit="batch"):
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model.classify_continuous(seq)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_time = time.time() - start_time
        epoch_acc = 100.0 * correct / total
        avg_loss = total_loss / len(dataloader)
        print(f"    Epoch {epoch+1:02d}/{EPOCHS:02d} | Loss: {avg_loss:.4f} | Accuracy: {epoch_acc:.2f}% | Time: {epoch_time:.1f}s")
        
    # 6. Save Model Weights
    output_weights = "kronos_asym_30m.pth"
    torch.save(model.state_dict(), output_weights)
    print(f"\n[+] Successfully trained and saved model weights to: '{output_weights}'")
    
    # 7. Evaluate on Holdout
    evaluate_holdout(model, df_holdout, device, threshold=0.40)
    
    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE! SEND 'kronos_asym_30m.pth' BACK TO WINDOWS PC ")
    print("=" * 65)

if __name__ == "__main__":
    main()
