import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add Kronos model to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Kronos'))
try:
    from model.kronos import Kronos
except ImportError:
    print("Error: Could not import Kronos. Make sure it's in the correct directory.")
    sys.exit(1)

# --- HYPERPARAMETERS ---
SEQ_LEN = 60
TARGET_MOVE = 0.0015
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

# --- DATASET CREATION ---
class VWAPDataset(Dataset):
    def __init__(self, df, seq_len=60, target_move=0.0015):
        self.seq_len = seq_len
        self.target_move = target_move
        
        self.prices = df['close'].values
        self.highs = df['high'].values
        self.lows = df['low'].values
        self.volumes = df['volume'].values
        
        self.dist_vwap = df['dist_to_vwap'].values if 'dist_to_vwap' in df.columns else np.zeros_like(self.prices)
        self.vol_surge = df['volume_surge'].values if 'volume_surge' in df.columns else np.zeros_like(self.prices)
        self.momentum = df['momentum_5'].values if 'momentum_5' in df.columns else np.zeros_like(self.prices)
        
        self.labels = self._generate_labels()
        
        self.valid_indices = [
            i for i in range(seq_len, len(self.prices) - 60)
            if self.labels[i] != -1
        ]
        
    def _generate_labels(self):
        labels = np.full(len(self.prices), -1)
        print("[*] Generating Forward-Looking Labels for Training Set...")
        
        for i in tqdm(range(len(self.prices) - 60)):
            entry = self.prices[i]
            target_up = entry * (1 + self.target_move)
            target_down = entry * (1 - self.target_move)
            
            label = 2 
            for j in range(i + 1, i + 60):
                if self.highs[j] >= target_up:
                    label = 0 
                    break
                elif self.lows[j] <= target_down:
                    label = 1 
                    break
            labels[i] = label
            
        return labels

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        
        p_seq = self.prices[real_idx - self.seq_len : real_idx]
        v_seq = self.dist_vwap[real_idx - self.seq_len : real_idx]
        vol_seq = self.vol_surge[real_idx - self.seq_len : real_idx]
        mom_seq = self.momentum[real_idx - self.seq_len : real_idx]
        
        p_seq = (p_seq - np.mean(p_seq)) / (np.std(p_seq) + 1e-5)
        v_seq = (v_seq - np.mean(v_seq)) / (np.std(v_seq) + 1e-5)
        vol_seq = (vol_seq - np.mean(vol_seq)) / (np.std(vol_seq) + 1e-5)
        mom_seq = (mom_seq - np.mean(mom_seq)) / (np.std(mom_seq) + 1e-5)
        
        seq = np.column_stack((p_seq, v_seq, vol_seq, mom_seq))
        label = self.labels[real_idx]
        
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# --- CLOUD EVALUATION HOOK ---
def evaluate_holdout(model, df_holdout, device, target_move=0.0015, threshold=0.51, fee_tier=0.0003):
    print("\n[*] Starting Cloud-Side EV Evaluation on Holdout Set...")
    model.eval()
    
    in_position = False
    position_type = None
    entry_price = 0.0
    
    total_trades = 0
    wins = 0
    losses = 0
    gross_pnl_pct = 0.0
    total_fees_paid = 0.0
    
    prices = df_holdout['close'].values
    highs = df_holdout['high'].values
    lows = df_holdout['low'].values
    dist_vwap = df_holdout['dist_to_vwap'].values
    vol_surge = df_holdout['volume_surge'].values
    momentum = df_holdout['momentum_5'].values
    
    for i in tqdm(range(60, len(df_holdout))):
        if in_position:
            current_high = highs[i]
            current_low = lows[i]
            
            exit_triggered = False
            pnl_pct = 0.0
            
            if position_type == 'LONG':
                if current_low <= entry_price * (1 - target_move):
                    pnl_pct = -target_move
                    exit_triggered = True
                    losses += 1
                elif current_high >= entry_price * (1 + target_move):
                    pnl_pct = target_move
                    exit_triggered = True
                    wins += 1
            elif position_type == 'SHORT':
                if current_high >= entry_price * (1 + target_move):
                    pnl_pct = -target_move
                    exit_triggered = True
                    losses += 1
                elif current_low <= entry_price * (1 - target_move):
                    pnl_pct = target_move
                    exit_triggered = True
                    wins += 1
                    
            if exit_triggered:
                gross_pnl_pct += pnl_pct
                total_fees_paid += fee_tier
                in_position = False
                position_type = None
        
        if not in_position:
            p_seq = prices[i - 60 : i]
            v_seq = dist_vwap[i - 60 : i]
            vol_seq = vol_surge[i - 60 : i]
            mom_seq = momentum[i - 60 : i]
            
            p_seq = (p_seq - np.mean(p_seq)) / (np.std(p_seq) + 1e-5)
            v_seq = (v_seq - np.mean(v_seq)) / (np.std(v_seq) + 1e-5)
            vol_seq = (vol_seq - np.mean(vol_seq)) / (np.std(vol_seq) + 1e-5)
            mom_seq = (mom_seq - np.mean(mom_seq)) / (np.std(mom_seq) + 1e-5)
            
            matrix = np.column_stack((p_seq, v_seq, vol_seq, mom_seq))
            seq_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model.classify_continuous(seq_tensor).cpu().numpy()[0]
                
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()
            
            prob_long, prob_short = probs[0], probs[1]
            
            if prob_long > threshold:
                in_position = True
                position_type = 'LONG'
                entry_price = prices[i]
                total_trades += 1
            elif prob_short > threshold:
                in_position = True
                position_type = 'SHORT'
                entry_price = prices[i]
                total_trades += 1
                
    net_ev = gross_pnl_pct - total_fees_paid
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    print("\n=============================================")
    print("      CLOUD OUT-OF-SAMPLE EV REPORT          ")
    print("=============================================")
    print(f"Total Trades : {total_trades}")
    print(f"Wins / Losses: {wins} / {losses}")
    print(f"Win Rate     : {win_rate:.2f}%")
    print(f"Net EV (post-fee): {net_ev * 100:.3f}%")
    print("=============================================")

# --- TRAINING LOOP ---
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training on device: {device}")
    
    print("[*] Loading historical data from btc_1m_qlib_ready.csv...")
    try:
        df = pd.read_csv('btc_1m_qlib_ready.csv', engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)
    
    df.columns = [c.lower() for c in df.columns]
    print(f"[*] Loaded {len(df)} rows of actual market data.")
    
    # SPLIT DATASET 80% TRAIN / 20% HOLDOUT
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].copy()
    df_holdout = df.iloc[split_idx:].copy()
    print(f"[*] Split complete: {len(df_train)} Train | {len(df_holdout)} Holdout")
    
    dataset = VWAPDataset(df_train, seq_len=SEQ_LEN, target_move=TARGET_MOVE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    print("[*] Initializing Kronos PyTorch Model...")
    model = Kronos(
        s1_bits=8, s2_bits=8, n_layers=4, d_model=128, n_heads=4, 
        ff_dim=512, ffn_dropout_p=0.1, attn_dropout_p=0.1, 
        resid_dropout_p=0.1, token_dropout_p=0.1, learn_te=False,
        in_features=4
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("[*] Beginning Classification Head Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (seq, labels) in enumerate(dataloader):
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
            
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(dataloader):.4f} | Accuracy: {epoch_acc:.2f}%")
        
    print("[*] Training Complete! Saving trained weights...")
    torch.save(model.state_dict(), 'kronos_trained_weights.pth')
    print("[*] Successfully saved to 'kronos_trained_weights.pth'")
    
    # Automatically run the EV evaluation on the cloud right after training
    evaluate_holdout(model, df_holdout, device, target_move=TARGET_MOVE)

if __name__ == '__main__':
    main()
