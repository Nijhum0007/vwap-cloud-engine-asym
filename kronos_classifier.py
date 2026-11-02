import os
import sys
import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore')

# Add Kronos to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Kronos'))
from model.kronos import Kronos

class KronosSoftmaxEngine:
    def __init__(self, target_move=None):
        """
        Initializes the REAL Kronos PyTorch Classification Engine.
        Auto-detects whether Wide-Horizon 1:2 (6 features) or Legacy (5 features) weights are present.
        """
        print("[*] Initializing REAL Kronos PyTorch Classification Engine...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        # Check for wide-horizon weights first, then fallback to legacy
        wide_weights = os.path.join(os.path.dirname(__file__), 'kronos_asym_wide_90m.pth')
        legacy_weights = os.path.join(os.path.dirname(__file__), 'kronos_trained_weights.pth')
        
        if os.path.exists(wide_weights):
            self.weights_path = wide_weights
            self.in_features = 6
            self.seq_len = 45
            self.target_move = target_move or 0.0030
            self.stop_loss = 0.0015
            print(f"[*] Detected Wide-Horizon Asymmetric 1:2 Model ({self.weights_path})")
            print(f"    Features: {self.in_features} (includes ATR_14) | Seq Len: {self.seq_len} | TP: +{self.target_move*100:.2f}% | SL: -{self.stop_loss*100:.2f}%")
        elif os.path.exists(legacy_weights):
            self.weights_path = legacy_weights
            self.in_features = 5
            self.seq_len = 60
            self.target_move = target_move or 0.0015
            self.stop_loss = 0.00075
            print(f"[*] Detected Legacy Model ({self.weights_path})")
            print(f"    Features: {self.in_features} | Seq Len: {self.seq_len} | Target: +/- {self.target_move*100:.3f}%")
        else:
            self.weights_path = None
            self.in_features = 6
            self.seq_len = 45
            self.target_move = target_move or 0.0030
            self.stop_loss = 0.0015
            print("[!] WARNING: No trained weights file found!")

        # Initialize Base PyTorch Architecture
        self.model = Kronos(
            s1_bits=8, s2_bits=8, n_layers=4, d_model=128, n_heads=4, 
            ff_dim=512, ffn_dropout_p=0.1, attn_dropout_p=0.1, 
            resid_dropout_p=0.1, token_dropout_p=0.1, learn_te=False,
            in_features=self.in_features
        ).to(self.device)
        
        if self.weights_path:
            self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device))
            print(f"[*] Successfully loaded weights into device: {self.device}")
            
        self.model.eval()

    def _softmax(self, logits):
        """Standard Softmax function to convert raw AI output (logits) into percentages totaling 100%."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / exp_logits.sum()

    def predict(self, df):
        """
        Takes the live 1-minute historical dataframe, extracts the sequence, 
        and passes it into the PyTorch Neural Network.
        """
        if len(df) < self.seq_len:
            return {'Long': 0.0, 'Short': 0.0, 'Flat': 1.0}
            
        # 1. Extract the features
        close_col = 'close' if 'close' in df.columns else 'Close'
        prices = df[close_col].tail(self.seq_len).values.astype(np.float32)
        dist_vwap = df['Dist_to_VWAP'].tail(self.seq_len).values.astype(np.float32) if 'Dist_to_VWAP' in df.columns else np.zeros_like(prices)
        vol_surge = df['Volume_Surge'].tail(self.seq_len).values.astype(np.float32) if 'Volume_Surge' in df.columns else np.ones_like(prices)
        momentum = df['Momentum_5'].tail(self.seq_len).values.astype(np.float32) if 'Momentum_5' in df.columns else np.zeros_like(prices)
        cvd_14 = df['CVD_14'].tail(self.seq_len).values.astype(np.float32) if 'CVD_14' in df.columns else np.zeros_like(prices)
        
        # 2. Local Z-Score normalization
        p_norm = (prices - np.mean(prices)) / (np.std(prices) + 1e-5)
        v_norm = (dist_vwap - np.mean(dist_vwap)) / (np.std(dist_vwap) + 1e-5)
        vol_norm = (vol_surge - np.mean(vol_surge)) / (np.std(vol_surge) + 1e-5)
        mom_norm = (momentum - np.mean(momentum)) / (np.std(momentum) + 1e-5)
        cvd_norm = (cvd_14 - np.mean(cvd_14)) / (np.std(cvd_14) + 1e-5)
        
        if self.in_features == 6:
            # Feature #6: ATR_14
            if 'ATR_14' in df.columns:
                atr_vals = df['ATR_14'].tail(self.seq_len).values.astype(np.float32)
            elif 'atr_14' in df.columns:
                atr_vals = df['atr_14'].tail(self.seq_len).values.astype(np.float32)
            else:
                high_col = 'high' if 'high' in df.columns else 'High'
                low_col = 'low' if 'low' in df.columns else 'Low'
                h = df[high_col].tail(self.seq_len + 14).values
                l = df[low_col].tail(self.seq_len + 14).values
                c = df[close_col].tail(self.seq_len + 14).values
                tr = (h - l) / c
                atr_s = pd.Series(tr).rolling(14, min_periods=1).mean()
                atr_vals = atr_s.tail(self.seq_len).values.astype(np.float32)
                
            atr_norm = (atr_vals - np.mean(atr_vals)) / (np.std(atr_vals) + 1e-5)
            matrix = np.column_stack((p_norm, v_norm, vol_norm, mom_norm, cvd_norm, atr_norm))
        else:
            matrix = np.column_stack((p_norm, v_norm, vol_norm, mom_norm, cvd_norm))
        
        seq_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits_tensor = self.model.classify_continuous(seq_tensor)
            
        logits = logits_tensor.cpu().numpy()[0]
        probabilities = self._softmax(logits)
        
        return {
            'Long': float(probabilities[0]),
            'Short': float(probabilities[1]),
            'Flat': float(probabilities[2])
        }

if __name__ == "__main__":
    import pandas as pd
    engine = KronosSoftmaxEngine()
    df_test = pd.DataFrame({
        'close': np.random.normal(85000, 50, 60),
        'Dist_to_VWAP': np.random.normal(0, 0.01, 60),
        'Volume_Surge': np.random.normal(1, 0.5, 60),
        'Momentum_5': np.random.normal(0, 0.005, 60),
        'CVD_14': np.random.normal(0, 100, 60),
        'ATR_14': np.random.normal(0.0003, 0.0001, 60)
    })
    probs = engine.predict(df_test)
    print("\n--- Test Scenario (Neural Network Inference) ---")
    print(f"Bucket A (Long):  {probs['Long'] * 100:.2f}%")
    print(f"Bucket B (Short): {probs['Short'] * 100:.2f}%")
    print(f"Bucket C (Flat):  {probs['Flat'] * 100:.2f}%")
