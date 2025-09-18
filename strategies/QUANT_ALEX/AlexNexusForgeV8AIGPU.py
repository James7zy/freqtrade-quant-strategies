import logging
import numpy as np
import pandas as pd
import pickle
import warnings
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from importlib import metadata
from functools import lru_cache
import talib.abstract as ta
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter, BooleanParameter
from freqtrade.persistence import Trade

# Suppress deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

# Optional NVIDIA Management Library (for accurate GPU utilization)
try:  # Lightweight, fail-safe wrapper
    import pynvml  # type: ignore
    try:
        pynvml.nvmlInit()
        def _gpu_utilization_pct() -> int | None:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return int(util.gpu)
            except Exception:
                return None
    except Exception:
        def _gpu_utilization_pct() -> int | None:  # Fallback if init failed
            return None
except Exception:
    def _gpu_utilization_pct() -> int | None:  # No pynvml installed
        return None

try:
    import xgboost as xgb
    from sklearn.ensemble import (
        RandomForestClassifier, GradientBoostingClassifier, 
        ExtraTreesClassifier, AdaBoostClassifier, VotingClassifier
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
    from sklearn.feature_selection import SelectKBest, f_classif
    SKLEARN_AVAILABLE = True

    # Check sklearn version using modern approach
    try:
        sklearn_version = metadata.version("scikit-learn")
        logger.info(f"Using scikit-learn version: {sklearn_version}")
    except Exception as e:
        logger.debug(f"Could not get sklearn version: {e}")

except ImportError as e:
    logger.warning(f"scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

# GPU Training imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    # Check for GPU
    GPU_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if GPU_AVAILABLE else 'cpu')
    if GPU_AVAILABLE:
        logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        logger.info("No GPU detected, using CPU")
except ImportError as e:
    logger.warning(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False
    GPU_AVAILABLE = False
    DEVICE = 'cpu'

try:
    import xgboost as xgb
    XGB_GPU_AVAILABLE = xgb.gpu.is_gpu_available() if hasattr(xgb, 'gpu') else False
    if XGB_GPU_AVAILABLE:
        logger.info("XGBoost GPU support detected")
except ImportError:
    XGB_GPU_AVAILABLE = False
    logger.warning("XGBoost not available")

# Modern PyWavelets import
try:
    import pywt
    WAVELETS_AVAILABLE = True
    try:
        pywt_version = metadata.version("PyWavelets")
        logger.info(f"Using PyWavelets version: {pywt_version}")
    except Exception as e:
        logger.debug(f"Could not get PyWavelets version: {e}")
except ImportError as e:
    logger.warning(f"PyWavelets not available: {e}")
    WAVELETS_AVAILABLE = False
logger = logging.getLogger(__name__)

# Define Murrey Math level names for consistency
MML_LEVEL_NAMES = [
    "[-3/8]P", "[-2/8]P", "[-1/8]P", "[0/8]P", "[1/8]P",
    "[2/8]P", "[3/8]P", "[4/8]P", "[5/8]P", "[6/8]P",
    "[7/8]P", "[8/8]P", "[+1/8]P", "[+2/8]P", "[+3/8]P"
]
def calculate_minima_maxima(df, window):
    if df is None or df.empty:
        return np.zeros(0), np.zeros(0)

    minima = np.zeros(len(df))
    maxima = np.zeros(len(df))

    for i in range(window, len(df)):
        window_data = df['ha_close'].iloc[i - window:i + 1]
        if df['ha_close'].iloc[i] == window_data.min() and (window_data == df['ha_close'].iloc[i]).sum() == 1:
            minima[i] = -window
        if df['ha_close'].iloc[i] == window_data.max() and (window_data == df['ha_close'].iloc[i]).sum() == 1:
            maxima[i] = window

    return minima, maxima


def calc_slope_advanced(series, period):
    """
    Enhanced linear regression slope calculation with Wavelet Transform and FFT analysis
    for superior trend detection and noise filtering
    """
    if len(series) < period:
        return 0

    # Use only the last 'period' values for consistency
    y = series.values[-period:]

    # Enhanced data validation
    if np.isnan(y).any() or np.isinf(y).any():
        return 0

    # Check for constant values (no trend)
    if np.all(y == y[0]):
        return 0

    try:
        # === 1. WAVELET DENOISING ===
        if WAVELETS_AVAILABLE and len(y) >= 8:
            wavelet = 'db4'
            try:
                w = pywt.Wavelet(wavelet)
                max_level = pywt.dwt_max_level(len(y), w.dec_len)
                use_level = min(3, max_level)  # cap at 3 but adapt if shorter series
            except Exception:
                use_level = 1
            if use_level >= 1:
                coeffs = pywt.wavedec(y, wavelet, level=use_level, mode='periodization')
                threshold = 0.1 * np.std(coeffs[-1]) if len(coeffs) > 1 else 0.0
                coeffs_thresh = list(coeffs)
                for i in range(1, len(coeffs_thresh)):
                    coeffs_thresh[i] = pywt.threshold(coeffs_thresh[i], threshold, mode='soft')
                y_denoised = pywt.waverec(coeffs_thresh, wavelet, mode='periodization')
                if len(y_denoised) != len(y):
                    y_denoised = y_denoised[:len(y)]
            else:
                y_denoised = y
        else:
            y_denoised = y

        # === 2. FFT FREQUENCY ANALYSIS ===
        # Analyze dominant frequencies to identify trend components
        if len(y_denoised) >= 4:
            # Apply FFT
            fft_values = fft(y_denoised)
            freqs = fftfreq(len(y_denoised))

            # Get magnitude spectrum
            magnitude = np.abs(fft_values)

            # Find dominant frequency (excluding DC component)
            non_dc_indices = np.where(freqs != 0)[0]
            if len(non_dc_indices) > 0:
                dominant_freq_idx = non_dc_indices[np.argmax(magnitude[non_dc_indices])]
                dominant_freq = freqs[dominant_freq_idx]

                # Calculate trend strength based on frequency content
                trend_frequency_weight = 1.0 / (1.0 + abs(dominant_freq) * 10)
            else:
                trend_frequency_weight = 1.0
        else:
            trend_frequency_weight = 1.0

        # === 3. MULTI-SCALE SLOPE CALCULATION ===
        x = np.linspace(0, period-1, period)

        # Original slope calculation
        slope_original = np.polyfit(x, y, 1)[0]

        # Wavelet-denoised slope calculation
        slope_denoised = np.polyfit(x, y_denoised, 1)[0]

        # === 4. WAVELET-BASED TREND DECOMPOSITION ===
        if WAVELETS_AVAILABLE and len(y) >= 8:
            # Extract trend component using wavelet approximation
            approx_coeffs = coeffs[0]  # Approximation coefficients (trend)

            # Reconstruct trend component
            trend_component = pywt.upcoef(
                'a', approx_coeffs, wavelet, level=3, take=len(y))
            if len(trend_component) > len(y):
                trend_component = trend_component[:len(y)]
            elif len(trend_component) < len(y):
                # Pad with last value if needed
                pad_length = len(y) - len(trend_component)
                trend_component = np.pad(trend_component, (0, pad_length), mode='edge')

            # Calculate slope of trend component
            slope_trend = np.polyfit(x, trend_component, 1)[0]
        else:
            slope_trend = slope_denoised

        # === 5. FREQUENCY-WEIGHTED SLOPE COMBINATION ===
        # Weight slopes based on signal characteristics
        weights = {
            'original': 0.3,
            'denoised': 0.4,
            'trend': 0.3
        }
        
        # Adjust weights based on noise level
        noise_level = np.std(y - y_denoised) / np.std(y) if np.std(y) > 0 else 0
        if noise_level > 0.1:  # High noise
            weights = {'original': 0.2, 'denoised': 0.5, 'trend': 0.3}
        elif noise_level < 0.05:  # Low noise
            weights = {'original': 0.4, 'denoised': 0.3, 'trend': 0.3}
        
        # Combined slope calculation
        slope_combined = (
            slope_original * weights['original'] +
            slope_denoised * weights['denoised'] +
            slope_trend * weights['trend']
        )
        
        # Apply frequency weighting
        final_slope = slope_combined * trend_frequency_weight
        
        # === 6. ENHANCED VALIDATION ===
        if np.isnan(final_slope) or np.isinf(final_slope):
            return (slope_original if not
                   (np.isnan(slope_original) or np.isinf(slope_original))
                   else 0)

        # Normalize extreme slopes
        max_reasonable_slope = np.std(y) / period
        if abs(final_slope) > max_reasonable_slope * 15:
            return np.sign(final_slope) * max_reasonable_slope * 15

        return final_slope

    except Exception:
        # Fallback to enhanced simple method if advanced processing fails
        try:
            # Apply simple moving average smoothing as fallback
            if len(y) >= 3:
                y_smooth = (
                    pd.Series(y).rolling(window=3, center=True)
                    .mean().bfill().ffill().values
                )
                x = np.linspace(0, period-1, period)
                slope = np.polyfit(x, y_smooth, 1)[0]

                if not (np.isnan(slope) or np.isinf(slope)):
                    return slope

            # Ultimate fallback: simple difference
            simple_slope = (y[-1] - y[0]) / (period - 1)
            return (simple_slope if not
                   (np.isnan(simple_slope) or np.isinf(simple_slope))
                   else 0)

        except Exception:
            return 0


def calculate_advanced_trend_strength_with_wavelets(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced trend strength calculation using Wavelet Transform and FFT analysis
    """
    try:
        # === WAVELET-ENHANCED SLOPE CALCULATION ===
        dataframe['slope_5_advanced'] = dataframe['close'].rolling(5).apply(
            lambda x: calc_slope_advanced(x, 5), raw=False
        )
        dataframe['slope_10_advanced'] = dataframe['close'].rolling(10).apply(
            lambda x: calc_slope_advanced(x, 10), raw=False
        )
        dataframe['slope_20_advanced'] = dataframe['close'].rolling(20).apply(
            lambda x: calc_slope_advanced(x, 20), raw=False
        )
        
        # === WAVELET TREND DECOMPOSITION ===
        def wavelet_trend_analysis(series, window=20):
            """Analyze trend using adaptive wavelet (haar/db4), safe levels, symmetric mode, robust threshold."""
            if not WAVELETS_AVAILABLE or len(series) < window:
                return pd.Series([0.0] * len(series), index=series.index)
            results: list[float] = []
            for i in range(len(series)):
                if i < window:
                    results.append(0.0)
                    continue
                window_data = series.iloc[i-window+1:i+1].values
                n = len(window_data)
                if n < 12:
                    results.append(0.0)
                    continue
                wavelet_name = 'haar' if n < 24 else 'db4'
                try:
                    w = pywt.Wavelet(wavelet_name)
                    max_level = pywt.dwt_max_level(n, w.dec_len)
                except Exception:
                    max_level = 1
                if n < 48:
                    max_level = min(max_level, 2)
                use_level = max(1, min(3, max_level))
                try:
                    coeffs = pywt.wavedec(window_data, wavelet_name, level=use_level, mode='symmetric')
                    # Estimate sigma from finest detail
                    if len(coeffs) > 1 and len(coeffs[-1]):
                        detail = coeffs[-1]
                        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
                        thr = sigma * np.sqrt(2 * np.log(n)) if sigma > 0 else 0.0
                    else:
                        thr = 0.0
                    for j in range(1, len(coeffs)):
                        coeffs[j] = pywt.threshold(coeffs[j], thr, mode='soft')
                    approx = coeffs[0]
                    trend_strength = np.std(approx) / (np.std(window_data) + 1e-9)
                    direction = 0
                    if len(approx) >= 2:
                        direction = 1 if approx[-1] > approx[0] else -1
                    score = trend_strength * direction
                    if not np.isfinite(score):
                        score = 0.0
                    # Clamp extreme outliers
                    results.append(float(np.clip(score, -5, 5)))
                except Exception:
                    results.append(0.0)
            return pd.Series(results, index=series.index)
        
        # Apply wavelet trend analysis
        dataframe['wavelet_trend_strength'] = wavelet_trend_analysis(dataframe['close'])
        
        # === FFT-BASED CYCLE DETECTION ===
        def fft_cycle_analysis(series, window=50):
            """Detect market cycles using FFT"""
            if len(series) < window:
                return (pd.Series([0] * len(series), index=series.index),
                       pd.Series([0] * len(series), index=series.index))
            
            cycle_strength = []
            dominant_period = []
            
            for i in range(len(series)):
                if i < window:
                    cycle_strength.append(0)
                    dominant_period.append(0)
                    continue
                
                # Get window data
                window_data = series.iloc[i-window+1:i+1].values
                
                try:
                    # Remove linear trend
                    x = np.arange(len(window_data))
                    slope, intercept = np.polyfit(x, window_data, 1)
                    detrended = window_data - (slope * x + intercept)
                    
                    # Apply FFT
                    fft_values = fft(detrended)
                    freqs = fftfreq(len(detrended))
                    magnitude = np.abs(fft_values)
                    
                    # Find dominant cycle (excluding DC component)
                    positive_freqs = freqs[1:len(freqs)//2]
                    positive_magnitude = magnitude[1:len(magnitude)//2]
                    
                    if len(positive_magnitude) > 0:
                        max_idx = np.argmax(positive_magnitude)
                        dominant_freq = positive_freqs[max_idx]
                        dominant_per = 1.0 / (abs(dominant_freq) + 1e-8)
                        
                        # Cycle strength (normalized)
                        cycle_str = positive_magnitude[max_idx] / (
                            np.sum(positive_magnitude) + 1e-8)
                    else:
                        dominant_per = 0
                        cycle_str = 0
                    
                    cycle_strength.append(cycle_str)
                    dominant_period.append(dominant_per)
                    
                except Exception:
                    cycle_strength.append(0)
                    dominant_period.append(0)
            
            return (pd.Series(cycle_strength, index=series.index),
                   pd.Series(dominant_period, index=series.index))
        
        # Apply FFT cycle analysis
        (dataframe['cycle_strength'],
         dataframe['dominant_cycle_period']) = fft_cycle_analysis(dataframe['close'])
        
        # === ENHANCED TREND STRENGTH CALCULATION ===
        # Normalize advanced slopes by price
        dataframe['trend_strength_5_advanced'] = (
            dataframe['slope_5_advanced'] / dataframe['close'] * 100)
        dataframe['trend_strength_10_advanced'] = (
            dataframe['slope_10_advanced'] / dataframe['close'] * 100)
        dataframe['trend_strength_20_advanced'] = (
            dataframe['slope_20_advanced'] / dataframe['close'] * 100)
        
        # Wavelet-weighted combined trend strength
        dataframe['trend_strength_wavelet'] = (
            dataframe['trend_strength_5_advanced'] * 0.4 +
            dataframe['trend_strength_10_advanced'] * 0.35 +
            dataframe['trend_strength_20_advanced'] * 0.25
        )
        
        # Incorporate wavelet trend analysis
        dataframe['trend_strength_combined'] = (
            dataframe['trend_strength_wavelet'] * 0.7 +
            dataframe['wavelet_trend_strength'] * 0.3
        )
        
        # === CYCLE-ADJUSTED TREND STRENGTH ===
        # Adjust trend strength based on cycle analysis
        dataframe['trend_strength_cycle_adjusted'] = dataframe['trend_strength_combined'].copy()
        
        # Boost trend strength when aligned with dominant cycle
        strong_cycle_mask = dataframe['cycle_strength'] > 0.3
        dataframe.loc[strong_cycle_mask, 'trend_strength_cycle_adjusted'] *= (
            1 + dataframe.loc[strong_cycle_mask, 'cycle_strength'])
        
        # === FINAL TREND CLASSIFICATION WITH ADVANCED FEATURES ===
        strong_threshold = 0.02
        
        # Enhanced trend classification
        dataframe['strong_uptrend_advanced'] = (
            (dataframe['trend_strength_cycle_adjusted'] > strong_threshold) &
            (dataframe['wavelet_trend_strength'] > 0) &
            (dataframe['cycle_strength'] > 0.1)
        )
        
        dataframe['strong_downtrend_advanced'] = (
            (dataframe['trend_strength_cycle_adjusted'] < -strong_threshold) &
            (dataframe['wavelet_trend_strength'] < 0) &
            (dataframe['cycle_strength'] > 0.1)
        )
        
        dataframe['ranging_advanced'] = (
            (dataframe['trend_strength_cycle_adjusted'].abs() < strong_threshold * 0.5) |
            (dataframe['cycle_strength'] < 0.05)  # Very weak cycles indicate ranging
        )
        
        # === TREND CONFIDENCE SCORE ===
        # Calculate confidence based on agreement between methods
        methods_agreement = (
            (np.sign(dataframe['trend_strength_5_advanced']) ==
             np.sign(dataframe['trend_strength_10_advanced'])).astype(int) +
            (np.sign(dataframe['trend_strength_10_advanced']) ==
             np.sign(dataframe['trend_strength_20_advanced'])).astype(int) +
            (np.sign(dataframe['trend_strength_wavelet']) ==
             np.sign(dataframe['wavelet_trend_strength'])).astype(int)
        )
        
        dataframe['trend_confidence'] = methods_agreement / 3.0
        
        # High confidence trends
        dataframe['high_confidence_trend'] = (
            (dataframe['trend_confidence'] >= 0.67) &
            (dataframe['cycle_strength'] > 0.2) &
            (dataframe['trend_strength_cycle_adjusted'].abs() > strong_threshold * 0.8)
        )
        
        return dataframe
        
    except Exception as e:
        logger.warning(f"Advanced trend analysis failed: {e}. Using fallback method.")
        # Return dataframe with fallback values
        fallback_columns = [
            'slope_5_advanced', 'slope_10_advanced', 'slope_20_advanced',
            'wavelet_trend_strength', 'cycle_strength', 'dominant_cycle_period',
            'trend_strength_5_advanced', 'trend_strength_10_advanced',
            'trend_strength_20_advanced', 'trend_strength_wavelet',
            'trend_strength_combined', 'trend_strength_cycle_adjusted',
            'strong_uptrend_advanced', 'strong_downtrend_advanced',
            'ranging_advanced', 'trend_confidence', 'high_confidence_trend'
        ]
        
        for col in fallback_columns:
            if 'strength' in col:
                dataframe[col] = 0.0
            else:
                dataframe[col] = False
        
        return dataframe


# === ADVANCED GPU NEURAL NETWORK AND TRAINING CLASSES ===

class AdvancedNeuralNetwork(nn.Module):
    """
    Advanced Neural Network optimized for GPU training
    """
    def __init__(self, input_size: int, hidden_sizes: list = None, 
                 dropout_rate: float = 0.3, use_batch_norm: bool = True):
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(nn.ReLU())
            
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

class GPUModelTrainer:
    """
    GPU-optimized model trainer with mixed precision and advanced features
    """
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.model = None
        self.scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None
        self.best_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
        
    def create_model(self, input_size: int, config: dict = None):
        """Create and initialize model on GPU"""
        if config is None:
            config = {
                'hidden_sizes': [512, 256, 128, 64],
                'dropout_rate': 0.3,
                'use_batch_norm': True
            }
        
        self.model = AdvancedNeuralNetwork(
            input_size=input_size,
            hidden_sizes=config['hidden_sizes'],
            dropout_rate=config['dropout_rate'],
            use_batch_norm=config['use_batch_norm']
        ).to(self.device)
        
        return self.model
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, batch_size: int = 512, 
                    validation_split: float = 0.2):
        """Prepare data for GPU training"""
        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1)).to(self.device)
        
        # Split into train/validation
        n_samples = len(X)
        n_val = int(n_samples * validation_split)
        indices = torch.randperm(n_samples)
        
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]
        
        # Create datasets
        train_dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
        val_dataset = TensorDataset(X_tensor[val_indices], y_tensor[val_indices])
        
        # Create data loaders with optimal settings for GPU
        num_workers = 4 if self.device == 'cuda' else 0
        pin_memory = self.device == 'cuda'
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0
        )
        
        return train_loader, val_loader
    
    def train_model(self, train_loader, val_loader, epochs: int = 100, 
                   learning_rate: float = 0.001, weight_decay: float = 1e-4):
        """Train model with mixed precision and early stopping"""
        
        optimizer = optim.AdamW(self.model.parameters(), 
                              lr=learning_rate, 
                              weight_decay=weight_decay)
        
        criterion = nn.BCELoss()
        
        # Learning rate scheduler (verbose removed; manual logging below)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=5
        )
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                if self.scaler and self.device == 'cuda':
                    # Mixed precision training
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # Standard training
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    if self.scaler and self.device == 'cuda':
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_X)
                            loss = criterion(outputs, batch_y)
                    else:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                self.patience_counter = 0
                # Save best model state
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
            
            if epoch % 10 == 0:
                current_lr = scheduler.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch}: Train {avg_train_loss:.4f} | Val {avg_val_loss:.4f} | LR {current_lr:.6f}"
                )
            
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if hasattr(self, 'best_model_state'):
            self.model.load_state_dict(self.best_model_state)
        
        return train_losses, val_losses
    
    def predict(self, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Make predictions on GPU"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
                
                if self.scaler and self.device == 'cuda':
                    with torch.cuda.amp.autocast():
                        batch_pred = self.model(batch_X)
                else:
                    batch_pred = self.model(batch_X)
                
                predictions.append(batch_pred.cpu().numpy())
        
        return np.vstack(predictions).flatten()

# === GPU MEMORY OPTIMIZATION FUNCTIONS ===

def optimize_gpu_memory():
    """Optimize GPU memory usage"""
    if GPU_AVAILABLE:
        torch.cuda.empty_cache()
        # Set memory fraction if needed
        # torch.cuda.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory

def monitor_training_performance(pair: str, training_time: float, gpu_memory: float):
    """Monitor and log training performance"""
    logger.info(f"=== TRAINING PERFORMANCE for {pair} ===")
    logger.info(f"Training time: {training_time:.2f}s")
    if GPU_AVAILABLE:
        # If caller passed 0 attempt to get live stats
        if gpu_memory == 0:
            try:
                import torch  # local import
                gpu_memory = torch.cuda.memory_allocated() / 1024**2
                peak_mem = torch.cuda.max_memory_allocated() / 1024**2
                logger.info(f"GPU memory used: {gpu_memory:.1f} MB (peak {peak_mem:.1f} MB)")
            except Exception:
                logger.info(f"GPU memory used: {gpu_memory:.1f} MB")
        else:
            logger.info(f"GPU memory used: {gpu_memory:.1f} MB")
        util = _gpu_utilization_pct()
        if util is not None:
            logger.info(f"GPU utilization: {util}%")
        else:
            logger.info("GPU utilization: (pynvml not available)")
    logger.info("=" * 40)

# === ADVANCED PREDICTIVE ANALYSIS SYSTEM ===

class AdvancedPredictiveEngine:
    """
    GPU-Enhanced machine learning engine for high-precision trade entry prediction
    """
    def __init__(self):
        # Core containers
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.prediction_history = {}
        self.is_trained = {}
        self.xgb_feature_maps: dict[str, dict] = {}

        # GPU config
        self.gpu_trainers = {}
        self.gpu_models = {}
        self.use_gpu = GPU_AVAILABLE and TORCH_AVAILABLE
        self.device = DEVICE

        if not self.use_gpu and TORCH_AVAILABLE:
            try:
                torch.cuda.init()
                if torch.cuda.device_count() > 0:
                    self.use_gpu = True
                    self.device = torch.device('cuda:0')
                    logger.info("FORCED GPU activation successful!")
            except Exception as e:
                logger.warning(f"GPU force activation failed: {e}")
        if self.use_gpu:
            logger.info("AdvancedPredictiveEngine initialized with GPU acceleration")
        else:
            logger.info("AdvancedPredictiveEngine running on CPU only")

        self.gpu_config = {
            'batch_size': 4096 if self.use_gpu else 256,
            'epochs': 500 if self.use_gpu else 50,
            'learning_rate': 0.0003,
            'weight_decay': 1e-5,
            'hidden_sizes': [2048, 1024, 512, 256, 128, 64] if self.use_gpu else [128, 64],
            'dropout_rate': 0.5,
            'use_batch_norm': True,
            'validation_split': 0.1,
            'gradient_clip_norm': 0.5,
            'use_mixed_precision': True,
            'accumulation_steps': 4,
            'scheduler_patience': 10,
            'early_stopping_patience': 50,
            'warmup_epochs': 50,
            'pin_memory': True,
            'num_workers': 4,
            'prefetch_factor': 4,
            'persistent_workers': True
        }

        self.xgb_gpu_params = {
            'device': 'cuda',
            'tree_method': 'hist',
            'max_bin': 1024,
            'grow_policy': 'lossguide',
            'max_leaves': 511,
        }

        self.xgb_entry_params = {
            'n_estimators': 2000,
            'max_depth': 20,
            'learning_rate': 0.02,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'colsample_bylevel': 0.8,
            'colsample_bynode': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 0.05,
            'min_child_weight': 1,
            'gamma': 0.01,
            'max_delta_step': 0,
            'scale_pos_weight': 1,
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'verbosity': 1,
            **self.xgb_gpu_params
        }

        self.xgb_early_stopping_rounds = 100

        self.training_cache = {}
        self.last_train_time = {}
        self.last_train_index = {}
        self.retrain_interval_hours = 24
        self.initial_train_candles = 2000
        self.min_new_candles_for_retrain = 30
        self.strategy_start_time = datetime.utcnow()
        self.retrain_after_startup_hours = 24
        self.enable_startup_retrain = True

        self.max_training_samples = 5000
        self.use_data_augmentation = True
        self.ensemble_size = 5

        self.models_dir = Path("user_data/strategies/XGBoost")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.training_times = {}
        self.gpu_memory_usage = {}

        self._load_models_from_disk()
        if self.use_gpu:
            logger.info(f"GPU Training enabled on {self.device}")
            logger.info(f"GPU Config: {self.gpu_config}")
        else:
            logger.info("GPU not available, using CPU training")

    # ----------------- ASSET EXISTENCE HELPERS -----------------
    def _required_asset_paths(self, pair: str) -> list[Path]:
        """Return list of required core asset file paths for a pair.

        Updated: Strategy now uses a single native XGBoost booster (xgboost_enhanced)
        plus scaler + metadata. Old RandomForest / GradientBoosting assets are no longer
        mandatory; their absence should not force retraining.
        """
        paths = [
            self._get_model_filepath(pair, "model_xgboost_enhanced"),
            self._get_model_filepath(pair, "scaler"),
            self._get_model_filepath(pair, "metadata"),
        ]
        return paths

    def _assets_exist(self, pair: str) -> bool:

        booster_path = self._required_asset_paths(pair)[0]
        exists = booster_path.exists()
        if not exists:
            logger.debug(f"Booster file missing for {pair}: {booster_path.name}")
        return exists

    def mark_trained_if_assets(self, pair: str):
        """Mark pair as trained if asset files exist (called at startup)."""
        if self._assets_exist(pair):
            self.is_trained[pair] = True

    def _get_model_filepath(self, pair: str, model_type: str) -> Path:
        """Get the filepath for saving/loading models"""
        safe_pair = pair.replace('/', '_').replace(':', '_')
        return self.models_dir / f"{safe_pair}_{model_type}.pkl"

    def _save_models_to_disk(self, pair: str):
        """Save trained models to disk for persistence"""
        try:
            if pair not in self.models:
                return

            # Create directory if it doesn't exist
            self.models_dir.mkdir(parents=True, exist_ok=True)

            # Save models with proper handling for different model types
            for model_name, model in self.models[pair].items():
                try:
                    # Skip GPU trainer objects (they contain PyTorch models that need special handling)
                    if model_name == 'neural_network_gpu':
                        logger.info(f"Skipping GPU model save for {pair} (requires special PyTorch handling)")
                        continue

                    filepath = self._get_model_filepath(pair, f"model_{model_name}")

                    # Ensure parent directory exists
                    filepath.parent.mkdir(parents=True, exist_ok=True)

                    # Special handling for native XGBoost Booster
                    try:
                        import xgboost as xgb  # Local import to avoid issues if not installed
                        from xgboost.core import Booster as XGBBooster  # type: ignore
                    except Exception:  # pragma: no cover - safety
                        xgb = None
                        XGBBooster = None

                    if xgb is not None and XGBBooster is not None and isinstance(model, XGBBooster):
                        # Serialize booster raw bytes inside pickle for compatibility with existing .pkl pattern
                        booster_bytes = model.save_raw()
                        with filepath.open('wb') as f:
                            pickle.dump({'booster_raw': booster_bytes, 'type': 'xgboost_booster'}, f)
                    else:
                        with filepath.open('wb') as f:
                            pickle.dump(model, f)

                    logger.debug(f"Saved {model_name} model for {pair}")

                except Exception as e:
                    logger.warning(f"Failed to save {model_name} model for {pair}: {e}")

            # Save scaler
            if pair in self.scalers:
                try:
                    scaler_filepath = self._get_model_filepath(pair, "scaler")
                    scaler_filepath.parent.mkdir(parents=True, exist_ok=True)

                    with scaler_filepath.open('wb') as f:
                        pickle.dump(self.scalers[pair], f)

                    logger.debug(f"Saved scaler for {pair}")
                except Exception as e:
                    logger.warning(f"Failed to save scaler for {pair}: {e}")

            # Save feature importance and metadata
            if pair in self.feature_importance:
                try:
                    metadata_filepath = self._get_model_filepath(pair, "metadata")
                    metadata_filepath.parent.mkdir(parents=True, exist_ok=True)

                    metadata = {
                        'feature_importance': self.feature_importance[pair],
                        'is_trained': self.is_trained.get(pair, False),
                        'timestamp': datetime.now().isoformat(),
                        'xgb_feature_map': self.xgb_feature_maps.get(pair)
                    }

                    with metadata_filepath.open('wb') as f:
                        pickle.dump(metadata, f)

                    logger.debug(f"Saved metadata for {pair}")
                except Exception as e:
                    logger.warning(f"Failed to save metadata for {pair}: {e}")

            logger.info(f"Models saved to disk for {pair}")

        except Exception as e:
            logger.warning(f"Failed to save models for {pair}: {e}")

    def _load_models_from_disk(self):
        """Load existing models from disk"""
        try:
            if not self.models_dir.exists():
                return

            # Find all model files
            model_files = list(self.models_dir.glob("*_model_*.pkl"))

            pairs_found = set()
            for model_file in model_files:
                # Extract pair name from filename
                filename = model_file.stem
                parts = filename.split('_model_')
                if len(parts) == 2:
                    pair_safe = parts[0]
                    pair = pair_safe.replace('_', '/')
                    if ':' not in pair and len(parts[0].split('_')) > 1:
                        # Handle cases like BTC_USDT_USDT -> BTC/USDT:USDT
                        parts_pair = parts[0].split('_')
                        if len(parts_pair) >= 3:
                            pair = f"{parts_pair[0]}/{parts_pair[1]}:{parts_pair[2]}"
                    pairs_found.add(pair)

            # Load models for each pair
            for pair in pairs_found:
                try:
                    self._load_pair_models(pair)
                except Exception as e:
                    logger.warning(f"Failed to load models for {pair}: {e}")

            if pairs_found:
                logger.info(f"Loaded ML models from disk for {len(pairs_found)} pairs: {list(pairs_found)}")

        except Exception as e:
            logger.warning(f"Failed to load models from disk: {e}")

    def _load_pair_models(self, pair: str):
        """Load models for a specific pair"""
        try:
            # Load models - handle both old and new model types
            models = {}
            model_types = ['random_forest', 'gradient_boosting', 'xgboost_enhanced', 'neural_network_gpu']

            for model_name in model_types:
                model_filepath = self._get_model_filepath(pair, f"model_{model_name}")
                if model_filepath.exists():
                    try:
                        # Skip GPU models for now (they need special PyTorch handling)
                        if model_name == 'neural_network_gpu':
                            logger.info(f"Skipping GPU model load for {pair} (requires special PyTorch handling)")
                            continue

                        with model_filepath.open('rb') as f:
                            loaded_obj = pickle.load(f)
                            # Detect serialized booster raw format
                            if isinstance(loaded_obj, dict) and loaded_obj.get('type') == 'xgboost_booster' and 'booster_raw' in loaded_obj:
                                try:
                                    import xgboost as xgb  # Local import
                                    booster = xgb.Booster()
                                    booster.load_model(bytearray(loaded_obj['booster_raw']))
                                    models[model_name] = booster
                                except Exception as bx:
                                    logger.warning(f"Failed to reconstruct XGBoost Booster for {pair}: {bx}")
                            else:
                                models[model_name] = loaded_obj
                        logger.debug(f"Loaded {model_name} model for {pair}")
                    except Exception as e:
                        logger.warning(f"Failed to load {model_name} model for {pair}: {e}")

            if models:
                self.models[pair] = models

            # Load scaler
            scaler_filepath = self._get_model_filepath(pair, "scaler")
            if scaler_filepath.exists():
                try:
                    with scaler_filepath.open('rb') as f:
                        self.scalers[pair] = pickle.load(f)
                    logger.debug(f"Loaded scaler for {pair}")
                except Exception as e:
                    logger.warning(f"Failed to load scaler for {pair}: {e}")

            # Load metadata
            metadata_filepath = self._get_model_filepath(pair, "metadata")
            if metadata_filepath.exists():
                try:
                    with metadata_filepath.open('rb') as f:
                        metadata = pickle.load(f)
                        self.feature_importance[pair] = metadata.get('feature_importance', {})
                        self.is_trained[pair] = metadata.get('is_trained', False)
                        # Restore feature map if present
                        fmap = metadata.get('xgb_feature_map')
                        if fmap:
                            self.xgb_feature_maps[pair] = fmap
                    logger.debug(f"Loaded metadata for {pair}")
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {pair}: {e}")

        except Exception as e:
            logger.warning(f"Failed to load models for {pair}: {e}")

    def _cleanup_old_models(self, max_age_days: int = 7):
        """Remove models older than specified days"""
        try:
            cutoff_time = datetime.now() - pd.Timedelta(days=max_age_days)

            for model_file in self.models_dir.glob("*.pkl"):
                if model_file.stat().st_mtime < cutoff_time.timestamp():
                    model_file.unlink()
                    logger.info(f"Removed old model file: {model_file.name}")

        except Exception as e:
            logger.warning(f"Failed to cleanup old models: {e}")

    def extract_advanced_features(self, dataframe: pd.DataFrame, lookback: int = 100) -> pd.DataFrame:
        """Extract sophisticated features for ML prediction with variance validation"""
        df = dataframe.copy()

        # === 1. ENHANCED PRICE ACTION FEATURES ===
        # Multi-period price patterns with variance
        for period in [1, 2, 3, 5]:
            df[f'price_velocity_{period}'] = df['close'].pct_change(period)
            df[f'price_acceleration_{period}'] = df[f'price_velocity_{period}'].diff(1)

            # Add rolling statistics for variance
            df[f'price_velocity_std_{period}'] = df[f'price_velocity_{period}'].rolling(20).std()
            df[f'price_velocity_skew_{period}'] = df[f'price_velocity_{period}'].rolling(20).skew()

        # Volatility-adjusted momentum
        returns = df['close'].pct_change(1)
        vol_20 = returns.rolling(20).std()
        df['vol_adjusted_momentum'] = returns / (vol_20 + 1e-10)

        # Price position within recent range
        for window in [10, 20, 50]:
            high_window = df['high'].rolling(window).max()
            low_window = df['low'].rolling(window).min()
            range_size = high_window - low_window
            df[f'price_position_{window}'] = (df['close'] - low_window) / (range_size + 1e-10)

        # Support/Resistance with dynamic thresholds
        if 'minima_sort_threshold' in df.columns:
            support_distance = abs(df['low'] - df['minima_sort_threshold']) / (df['close'] + 1e-10)
            df['support_strength'] = (support_distance < 0.02).astype(int).rolling(20).mean()
            df['support_distance_norm'] = support_distance
        else:
            df['support_strength'] = np.random.uniform(0.3, 0.7, len(df))  # Random baseline
            df['support_distance_norm'] = np.random.uniform(0, 0.1, len(df))

        if 'maxima_sort_threshold' in df.columns:
            resistance_distance = abs(df['high'] - df['maxima_sort_threshold']) / (df['close'] + 1e-10)
            df['resistance_strength'] = (resistance_distance < 0.02).astype(int).rolling(20).mean()
            df['resistance_distance_norm'] = resistance_distance
        else:
            df['resistance_strength'] = np.random.uniform(0.3, 0.7, len(df))
            df['resistance_distance_norm'] = np.random.uniform(0, 0.1, len(df))

        # === 2. VOLUME DYNAMICS ===
        # Volume profile analysis
        df['volume_profile_score'] = self._calculate_volume_profile_score(df)
        df['volume_imbalance'] = self._calculate_volume_imbalance(df)
        df['smart_money_index'] = self._calculate_smart_money_index(df)

        # Volume-price correlation
        df['volume_price_correlation'] = df['volume'].rolling(20).corr(df['close'])
        df['volume_breakout_strength'] = self._calculate_volume_breakout_strength(df)

        # === 3. VOLATILITY CLUSTERING ===
        df['volatility_regime'] = self._calculate_volatility_regime(df)
        df['volatility_persistence'] = self._calculate_volatility_persistence(df)
        df['volatility_mean_reversion'] = self._calculate_volatility_mean_reversion(df)

        # === 4. MOMENTUM DECOMPOSITION ===
        for period in [3, 5, 8, 13, 21]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'momentum_strength_{period}'] = abs(df[f'momentum_{period}'])
            df[f'momentum_consistency_{period}'] = (
                np.sign(df[f'momentum_{period}']).rolling(5).mean()
            )

        # Momentum regime detection
        df['momentum_regime'] = self._classify_momentum_regime(df)
        df['momentum_divergence_strength'] = self._calculate_momentum_divergence(df)

        # === 5. MICROSTRUCTURE FEATURES ===
        df['spread_proxy'] = (df['high'] - df['low']) / df['close']
        df['market_impact'] = df['volume'] * df['spread_proxy']
        df['order_flow_imbalance'] = self._calculate_order_flow_imbalance(df)
        df['liquidity_index'] = self._calculate_liquidity_index(df)

        # === 6. STATISTICAL FEATURES ===
        for window in [10, 20, 50]:
            returns = df['close'].pct_change(1)
            df[f'skewness_{window}'] = returns.rolling(window).apply(
                lambda x: skew(x.dropna()) if len(x.dropna()) > 3 else 0
            )
            df[f'kurtosis_{window}'] = returns.rolling(window).apply(
                lambda x: kurtosis(x.dropna()) if len(x.dropna()) > 3 else 0
            )
            df[f'entropy_{window}'] = self._calculate_entropy(df['close'], window)

        # === 7. REGIME DETECTION FEATURES ===
        df['market_regime'] = self._detect_market_regime(df)
        df['regime_stability'] = self._calculate_regime_stability(df)
        df['regime_transition_probability'] = self._calculate_regime_transition_prob(df)

        return df

    def _calculate_volume_profile_score(self, df: pd.DataFrame, window: int = 50) -> pd.Series:
        """Calculate volume profile score"""
        def volume_profile(data):
            if len(data) < 10:
                return 0.5

            prices = data['close'].values
            volumes = data['volume'].values

            # Create price bins
            price_min, price_max = prices.min(), prices.max()
            if price_max == price_min:
                return 0.5

            bins = np.linspace(price_min, price_max, 10)

            # Calculate volume at each price level
            volume_at_price = []
            for i in range(len(bins) - 1):
                mask = (prices >= bins[i]) & (prices < bins[i + 1])
                vol_sum = volumes[mask].sum()
                volume_at_price.append(vol_sum)

            # Point of Control (POC) - price level with highest volume
            if sum(volume_at_price) == 0:
                return 0.5

            poc_index = np.argmax(volume_at_price)
            current_price = prices[-1]
            poc_price = (bins[poc_index] + bins[poc_index + 1]) / 2

            # Score based on distance from POC
            distance_ratio = abs(current_price - poc_price) / (price_max - price_min + 1e-10)
            score = 1 - distance_ratio  # Closer to POC = higher score

            return max(0, min(1, score))

        # Apply rolling calculation
        result = []
        for i in range(len(df)):
            if i < window:
                result.append(0.5)
            else:
                window_data = df.iloc[i-window+1:i+1][['close', 'volume']]
                score = volume_profile(window_data)
                result.append(score)

        return pd.Series(result, index=df.index)

    def _calculate_volume_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume imbalance between buying and selling"""
        up_volume = df['volume'].where(df['close'] > df['open'], 0)
        down_volume = df['volume'].where(df['close'] < df['open'], 0)

        total_volume = up_volume + down_volume
        imbalance = (up_volume - down_volume) / (total_volume + 1e-10)

        return imbalance.rolling(10).mean()

    def _calculate_smart_money_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Smart Money Index (SMI)"""
        price_change = abs(df['close'].pct_change(1))
        volume_norm = df['volume'] / df['volume'].rolling(20).mean()

        smi = volume_norm / (price_change + 1e-10)
        return smi.rolling(10).mean()

    def _calculate_volume_breakout_strength(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volume breakout strength"""
        volume_ma = df['volume'].rolling(20).mean()
        volume_ratio = df['volume'] / volume_ma

        price_breakout = (
            (df['close'] > df['close'].rolling(20).max().shift(1)) |
            (df['close'] < df['close'].rolling(20).min().shift(1))
        ).astype(int)

        breakout_strength = volume_ratio * price_breakout
        return breakout_strength.rolling(5).mean()

    def _calculate_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect volatility regime"""
        returns = df['close'].pct_change(1)
        volatility = returns.rolling(20).std()
        vol_ma = volatility.rolling(50).mean()

        regime = pd.Series(1, index=df.index)  # Default normal
        regime[volatility < vol_ma * 0.7] = 0   # Low volatility
        regime[volatility > vol_ma * 1.5] = 2   # High volatility

        return regime

    def _calculate_volatility_persistence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility persistence"""
        returns = df['close'].pct_change(1)
        volatility = returns.rolling(5).std()

        persistence = volatility.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) > 10 else 0
        )

        return persistence

    def _calculate_volatility_mean_reversion(self, df: pd.DataFrame) -> pd.Series:
        """Calculate volatility mean reversion tendency"""
        returns = df['close'].pct_change(1)
        volatility = returns.rolling(10).std()
        vol_ma = volatility.rolling(50).mean()

        vol_zscore = (volatility - vol_ma) / (volatility.rolling(50).std() + 1e-10)
        mean_reversion = -vol_zscore

        return mean_reversion

    def _classify_momentum_regime(self, df: pd.DataFrame) -> pd.Series:
        """Classify momentum regime"""
        mom_3 = df['close'].pct_change(3)
        mom_8 = df['close'].pct_change(8)
        mom_21 = df['close'].pct_change(21)
        
        regime = pd.Series(0, index=df.index)  # Neutral
        
        strong_up = (mom_3 > 0.02) & (mom_8 > 0.05) & (mom_21 > 0.1)
        regime[strong_up] = 2
        
        mod_up = (mom_3 > 0) & (mom_8 > 0) & (mom_21 > 0) & ~strong_up
        regime[mod_up] = 1
        
        mod_down = (mom_3 < 0) & (mom_8 < 0) & (mom_21 < 0) & (mom_21 > -0.1)
        regime[mod_down] = -1
        
        strong_down = (mom_3 < -0.02) & (mom_8 < -0.05) & (mom_21 < -0.1)
        regime[strong_down] = -2
        
        return regime
    
    def _calculate_momentum_divergence(self, df: pd.DataFrame) -> pd.Series:
        """Calculate momentum divergence strength"""
        price_momentum = df['close'].pct_change(10)
        
        if 'rsi' in df.columns:
            rsi_momentum = df['rsi'].diff(10)
        else:
            rsi_momentum = pd.Series(0, index=df.index)
            
        volume_momentum = df['volume'].pct_change(10)
        
        # Normalize momentums using rolling z-score
        price_norm = (price_momentum - price_momentum.rolling(50).mean()) / (
            price_momentum.rolling(50).std() + 1e-10)
        rsi_norm = (rsi_momentum - rsi_momentum.rolling(50).mean()) / (
            rsi_momentum.rolling(50).std() + 1e-10)
        volume_norm = (volume_momentum - volume_momentum.rolling(50).mean()) / (
            volume_momentum.rolling(50).std() + 1e-10)
        
        price_rsi_div = abs(price_norm - rsi_norm)
        price_volume_div = abs(price_norm - volume_norm)
        
        divergence_strength = (price_rsi_div + price_volume_div) / 2
        return divergence_strength.rolling(5).mean()
    
    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.Series:
        """Calculate order flow imbalance"""
        price_impact = (df['close'] - df['open']) / df['open']
        volume_impact = df['volume'] / df['volume'].rolling(20).mean()
        
        flow_imbalance = price_impact * volume_impact
        return flow_imbalance.rolling(5).mean()
    
    def _calculate_liquidity_index(self, df: pd.DataFrame) -> pd.Series:
        """Calculate market liquidity index"""
        spread = (df['high'] - df['low']) / df['close']
        volume_norm = df['volume'] / df['volume'].rolling(50).mean()
        
        liquidity = volume_norm / (spread + 1e-10)
        return liquidity.rolling(10).mean()

    def _calculate_entropy(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate information entropy"""
        def entropy(data):
            if len(data) < 5:
                return 0

            returns = np.diff(data) / (data[:-1] + 1e-10)

            bins = np.histogram_bin_edges(returns, bins=10)
            hist, _ = np.histogram(returns, bins=bins)

            probs = hist / (hist.sum() + 1e-10)
            probs = probs[probs > 0]

            ent = -np.sum(probs * np.log2(probs + 1e-10))
            return ent

        return series.rolling(window).apply(entropy, raw=False)

    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """Detect overall market regime"""
        if 'trend_strength' in df.columns:
            trend_regime = np.sign(df['trend_strength'])
        else:
            trend_regime = pd.Series(0, index=df.index)

        vol_regime = self._calculate_volatility_regime(df) - 1
        momentum_regime = self._classify_momentum_regime(df)

        market_regime = (
            trend_regime * 0.4 +
            vol_regime * 0.3 +
            momentum_regime * 0.3
        )

        return market_regime.rolling(5).mean()

    def _calculate_regime_stability(self, df: pd.DataFrame) -> pd.Series:
        """Calculate regime stability"""
        regime = self._detect_market_regime(df)
        regime_changes = abs(regime.diff(1))
        stability = 1 / (regime_changes.rolling(20).mean() + 1e-10)
        return stability

    def _calculate_regime_transition_prob(self, df: pd.DataFrame) -> pd.Series:
        """Calculate probability of regime transition"""
        regime = self._detect_market_regime(df)

        transitions = []
        for i in range(1, len(regime)):
            if not (pd.isna(regime.iloc[i]) or pd.isna(regime.iloc[i-1])):
                transition = abs(regime.iloc[i] - regime.iloc[i-1]) > 0.5
                transitions.append(transition)
            else:
                transitions.append(False)

        transition_prob = pd.Series([False] + transitions, index=regime.index)
        prob_smooth = transition_prob.astype(int).rolling(20).mean()

        return prob_smooth

    def create_target_variable(self, df: pd.DataFrame, forward_periods: int = 5,
                              profit_threshold: float | None = None,
                              dynamic: bool = True,
                              quantile: float = 0.80,
                              k_atr: float = 1.0,
                              k_vol: float = 1.2,
                              min_abs: float = 0.0025,
                              max_abs: float = 0.05) -> pd.Series:
        """Create target variable with optional dynamic profit threshold.

        Dynamic threshold logic (if dynamic=True and profit_threshold not provided):
          1. Compute ATR% (14) and rolling return volatility (20).
          2. base_series = k_atr * ATR% + k_vol * vola20
          3. base_scalar = median(base_series)
          4. q_thr = 85th percentile of forward_returns (future move distribution)
          5. blended = 0.5 * base_scalar + 0.5 * q_thr
          6. final_thr = clip(blended, min_abs, max_abs)
        This produces a stable scalar threshold per training batch (reproducible) rather than per-row noise.
        """
        # Forward returns used by several strategies
        forward_returns = df['close'].pct_change(forward_periods).shift(-forward_periods)

        # === DYNAMIC THRESHOLD CALCULATION ===
        if dynamic and (profit_threshold is None):
            try:
                # ATR% calculation
                high = df['high']
                low = df['low']
                close = df['close']
                prev_close = close.shift(1)
                tr = np.maximum(high - low, np.maximum((high - prev_close).abs(), (low - prev_close).abs()))
                atr = tr.rolling(14).mean()
                atr_pct = (atr / close).clip(lower=0)

                # Return volatility
                returns_1 = close.pct_change()
                vola20 = returns_1.rolling(20).std()

                base_series = k_atr * atr_pct + k_vol * vola20
                base_scalar = float(np.nanmedian(base_series.tail(300))) if len(base_series.dropna()) > 30 else float(np.nanmedian(base_series))

                # Distribution-based quantile of forward returns (future info okay for label construction stage)
                q_thr = float(forward_returns.quantile(quantile)) if forward_returns.notna().any() else min_abs
                if not np.isfinite(q_thr):
                    q_thr = min_abs
                # Rebalanced blend favors distribution quantile slightly to raise positives
                blended = 0.4 * base_scalar + 0.6 * q_thr
                profit_threshold = float(np.clip(blended, min_abs, max_abs))
            except Exception as e:
                logger.warning(f"Dynamic threshold failed ({e}), falling back to default 0.015")
                profit_threshold = 0.015
        elif profit_threshold is None:
            profit_threshold = 0.015

        # === STRATEGY 1: SIMPLE FORWARD RETURNS ===
        simple_target = (forward_returns > profit_threshold).astype(int)

        # === STRATEGY 2: MAXIMUM PROFIT POTENTIAL ===
        forward_highs = df['high'].rolling(forward_periods).max().shift(-forward_periods)
        max_profit_potential = (forward_highs - df['close']) / df['close']
        profit_target = (max_profit_potential > profit_threshold).astype(int)

        # === STRATEGY 3: RISK-ADJUSTED RETURNS ===
        forward_lows = df['low'].rolling(forward_periods).min().shift(-forward_periods)
        max_loss_potential = (forward_lows - df['close']) / df['close']
        risk_adjusted_return = forward_returns / (abs(max_loss_potential) + 1e-10)
        risk_target = (
            (forward_returns > profit_threshold * 0.7) &
            (risk_adjusted_return > 0.5)
        ).astype(int)

        # === STRATEGY 4: VOLATILITY-ADJUSTED TARGET ===
        returns_std = df['close'].pct_change().rolling(20).std()
        volatility_adjusted_threshold = profit_threshold * (1 + returns_std)
        vol_target = (forward_returns > volatility_adjusted_threshold).astype(int)

        # === ENSEMBLE VOTE ===
        combined_target = simple_target + profit_target + risk_target + vol_target
        final_target = (combined_target >= 2).astype(int)

        positive_ratio = final_target.mean()
        logger.info(
            "Target created (forward=%s) dynamic_thr=%.4f positives=%s/%s ratio=%.3f" % (
                forward_periods, profit_threshold, final_target.sum(), len(final_target), positive_ratio))

        if positive_ratio < 0.08:
            logger.warning(
                "Very low positive ratio (%.3f) at threshold %.4f" % (positive_ratio, profit_threshold))
        elif positive_ratio > 0.50:
            logger.warning(
                "High positive ratio (%.3f) at threshold %.4f" % (positive_ratio, profit_threshold))

        return final_target

    def _quick_xgb_param_tune(self, X_train, y_train, X_val, y_val, base_params: dict) -> dict:
        """Ultra-fast heuristic tuning over small param grid (no heavy search).
        Uses AUC if possible, else accuracy. Keeps GPU params intact.
        """
        try:
            import xgboost as xgb
            from sklearn.metrics import roc_auc_score, accuracy_score
            # Minimal grid focusing on depth / learning rate / subsample combos
            depths = [base_params.get('max_depth', 8)]
            if base_params.get('max_depth', 0) >= 16:
                depths = [base_params['max_depth'], max(base_params['max_depth'] - 2, 12)]
            lrs = [base_params.get('learning_rate', 0.02)]
            if base_params.get('learning_rate', 0.02) >= 0.02:
                lrs += [base_params['learning_rate'] * 0.7]
            subsamples = [base_params.get('subsample', 0.8)]
            if subsamples[0] >= 0.8:
                subsamples += [0.7]

            best_score = -1
            best_params = base_params.copy()
            eval_set = [(X_train, y_train), (X_val, y_val)]
            for d in depths:
                for lr in lrs:
                    for ss in subsamples:
                        trial_params = base_params.copy()
                        trial_params.update({
                            'max_depth': d,
                            'learning_rate': lr,
                            'subsample': ss,
                            'n_estimators': min(base_params.get('n_estimators', 1000), 800),
                            'verbosity': 0
                        })
                        model = xgb.XGBClassifier(**trial_params)
                        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
                        try:
                            proba = model.predict_proba(X_val)[:, 1]
                            score = roc_auc_score(y_val, proba)
                        except Exception:
                            pred = model.predict(X_val)
                            score = accuracy_score(y_val, pred)
                        if score > best_score:
                            best_score = score
                            best_params = trial_params
            logger.info(f"Quick XGB tune best score={best_score:.4f} params={{'max_depth': {best_params.get('max_depth')}, 'lr': {best_params.get('learning_rate')}, 'subsample': {best_params.get('subsample')}}}")
            # Restore original n_estimators if larger for final training
            best_params['n_estimators'] = base_params.get('n_estimators', best_params.get('n_estimators', 1000))
            best_params['verbosity'] = base_params.get('verbosity', 0)
            return best_params
        except Exception as e:
            logger.debug(f"Param tune skipped due to error: {e}")
            return base_params

    def train_predictive_models(self, df: pd.DataFrame, pair: str) -> dict:
        """Enhanced training with GPU support"""
        if not SKLEARN_AVAILABLE:
            return {'status': 'sklearn_not_available'}

        try:
            import time
            start_time = time.time()
            
            # Decide training slice - INTENSIVE DATA USAGE
            if pair not in self.is_trained or not self.is_trained[pair]:
                # Use much more data for initial training
                base_df = df.tail(self.max_training_samples).copy()
                logger.info(f"Initial INTENSIVE training for {pair} using {len(base_df)} candles")
            else:
                last_idx = self.last_train_index.get(pair, 0)
                new_candles = len(df) - last_idx
                
                if new_candles < self.min_new_candles_for_retrain:
                    logger.info(f"Skipping retrain for {pair} - only {new_candles} new candles")
                    return {'status': 'insufficient_new_data'}
                
                # Use larger sliding window for retraining (more intensive)
                base_df = df.tail(min(self.max_training_samples, len(df))).copy()
                logger.info(f"INTENSIVE retraining {pair} using {len(base_df)} candles ({new_candles} new)")
                
                # GPU memory optimization before intensive training
                if self.use_gpu:
                    optimize_gpu_memory()
                    logger.info("GPU memory optimized for intensive training")

            # Feature extraction and target creation
            feature_df = self.extract_advanced_features(base_df)
            target = self.create_target_variable(base_df)

            # Feature selection
            feature_columns = []
            exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'date',
                           'enter_long', 'enter_short', 'exit_long', 'exit_short']

            for col in feature_df.columns:
                if col not in exclude_cols and feature_df[col].dtype in ['float64', 'int64', 'bool']:
                    feature_columns.append(col)

            X = feature_df[feature_columns].fillna(0)
            y = target.fillna(0)

            # Remove invalid samples
            valid_mask = ~(pd.isna(y) | pd.isna(X).any(axis=1))
            X = X[valid_mask]
            y = y[valid_mask]

            if len(X) < 100:
                logger.warning(f"Insufficient valid data for {pair}: {len(X)} samples")
                return {'status': 'insufficient_data'}

            # Feature variance filtering
            feature_variance = X.var()
            non_constant_features = feature_variance[feature_variance > 1e-10].index.tolist()
            
            if len(non_constant_features) < 5:
                logger.warning(f"Too few variable features for {pair}: {len(non_constant_features)}")
                return {'status': 'insufficient_features'}

            X = X[non_constant_features]
            
            # Feature scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Initialize containers
            models = {}
            training_results = {}
            
            # === GPU NEURAL NETWORK TRAINING ===
            if self.use_gpu and len(X_scaled) > 500:
                try:
                    gpu_trainer = GPUModelTrainer(device=self.device)
                    
                    # Create model
                    model = gpu_trainer.create_model(
                        input_size=X_scaled.shape[1],
                        config=self.gpu_config
                    )
                    
                    # Prepare data
                    train_loader, val_loader = gpu_trainer.prepare_data(
                        X_scaled, y.values,
                        batch_size=self.gpu_config['batch_size'],
                        validation_split=self.gpu_config['validation_split']
                    )
                    
                    # Train model
                    train_losses, val_losses = gpu_trainer.train_model(
                        train_loader, val_loader,
                        epochs=self.gpu_config['epochs'],
                        learning_rate=self.gpu_config['learning_rate'],
                        weight_decay=self.gpu_config['weight_decay']
                    )
                    
                    models['neural_network_gpu'] = gpu_trainer
                    training_results['neural_network_gpu'] = {
                        'final_train_loss': train_losses[-1] if train_losses else None,
                        'final_val_loss': val_losses[-1] if val_losses else None,
                        'epochs_trained': len(train_losses),
                        'device': str(self.device)
                    }
                    
                    logger.info(f"GPU Neural Network trained for {pair}: "
                              f"Final Val Loss: {val_losses[-1]:.4f}")
                    
                    # Monitor GPU memory
                    if self.device == 'cuda':
                        memory_used = torch.cuda.max_memory_allocated() / 1024**2  # MB
                        self.gpu_memory_usage[pair] = memory_used
                        logger.info(f"Peak GPU memory usage: {memory_used:.1f} MB")
                        torch.cuda.empty_cache()  # Clear cache
                        
                except Exception as e:
                    logger.warning(f"GPU Neural Network training failed for {pair}: {e}")
            
            # === ULTRA-MAXIMUM XGBOOST WITH FORCED GPU SUPPORT ===
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.15, random_state=42, stratify=y
                )
                
                # FORCE XGBoost to use GPU regardless of detection
                xgb_params = self.xgb_entry_params.copy()

                # === DYNAMIC CLASS IMBALANCE HANDLING ===
                try:
                    positive_ratio = float(y.mean()) if len(y) else 0.0
                    if 0 < positive_ratio < 0.5:
                        raw_spw = (1 - positive_ratio) / max(positive_ratio, 1e-6)
                        damp_spw = raw_spw ** 0.5  # soften
                        dynamic_spw = float(min(damp_spw, 10.0))
                        xgb_params['scale_pos_weight'] = dynamic_spw
                        logger.info(
                            f"Dynamic scale_pos_weight raw={raw_spw:.2f} damp={dynamic_spw:.2f} (pos_ratio={positive_ratio:.3f}) for {pair}")
                except Exception as pr_e:
                    logger.debug(f"Dynamic class weight calc failed: {pr_e}")

                # === LIGHT PARAMETER ADAPTATION ===
                # Reduce depth slightly for very large feature sets to curb overfitting
                try:
                    n_features = X_scaled.shape[1]
                    if n_features > 150 and xgb_params.get('max_depth', 0) > 18:
                        xgb_params['max_depth'] = 18
                    if n_features > 220 and xgb_params.get('max_depth', 0) > 16:
                        xgb_params['max_depth'] = 16
                    # Adjust n_estimators based on sample size (keep large but adaptive)
                    n_samples = X_scaled.shape[0]
                    if n_samples > 4500:
                        xgb_params['n_estimators'] = 2500
                    elif n_samples < 1500 and xgb_params['n_estimators'] > 1800:
                        xgb_params['n_estimators'] = 1800
                except Exception as adapt_e:
                    logger.debug(f"Adaptive param tuning skipped: {adapt_e}")
                
                # Ensure GPU usage with new XGBoost 2.0+ syntax
                if not XGB_GPU_AVAILABLE:
                    # Try to force GPU even if not detected
                    try:
                        xgb_params.update({
                            'device': 'cuda',
                            'tree_method': 'hist'
                        })
                        logger.info("FORCING XGBoost GPU usage despite detection failure")
                    except Exception:
                        logger.warning("GPU forcing failed, falling back to optimized CPU")
                        xgb_params.update({
                            'device': 'cpu',
                            'tree_method': 'hist',
                            'n_jobs': -1
                        })
                
                # Initialize XGBoost with proper error handling + tuning + early stopping via native API
                import xgboost as xgb

                gpu_enabled = xgb_params.get('device') == 'cuda'
                logger.info(f"XGBoost {'GPU' if gpu_enabled else 'CPU'} training (native) for {pair}")
                if gpu_enabled:
                    logger.info(
                        f"GPU Params: dev={xgb_params.get('device')} method={xgb_params.get('tree_method')} depth={xgb_params.get('max_depth')} est={xgb_params.get('n_estimators')}"
                    )

                # Quick heuristic tuning (uses sklearn wrapper internally) to refine a few params
                tuned_params = self._quick_xgb_param_tune(
                    X_train, y_train, X_test, y_test, xgb_params
                )

                # Build native booster params
                booster_params = {
                    'objective': 'binary:logistic',
                    'max_depth': tuned_params.get('max_depth', xgb_params.get('max_depth')), 
                    'eta': tuned_params.get('learning_rate', xgb_params.get('learning_rate')), 
                    'subsample': tuned_params.get('subsample', xgb_params.get('subsample')), 
                    'colsample_bytree': tuned_params.get('colsample_bytree', xgb_params.get('colsample_bytree', 0.8)),
                    'colsample_bylevel': tuned_params.get('colsample_bylevel', xgb_params.get('colsample_bylevel', 1.0)),
                    'colsample_bynode': tuned_params.get('colsample_bynode', xgb_params.get('colsample_bynode', 1.0)),
                    'lambda': tuned_params.get('reg_lambda', xgb_params.get('reg_lambda', 1.0)),
                    'alpha': tuned_params.get('reg_alpha', xgb_params.get('reg_alpha', 0.0)),
                    'min_child_weight': tuned_params.get('min_child_weight', xgb_params.get('min_child_weight', 1)),
                    'gamma': tuned_params.get('gamma', xgb_params.get('gamma', 0.0)),
                    'tree_method': xgb_params.get('tree_method', 'hist'),
                    'device': xgb_params.get('device', 'cuda'),
                    'max_bin': xgb_params.get('max_bin', 1024),
                    'grow_policy': xgb_params.get('grow_policy', 'lossguide'),
                    'eval_metric': 'logloss'
                }
                if 'scale_pos_weight' in tuned_params:
                    booster_params['scale_pos_weight'] = tuned_params['scale_pos_weight']

                num_boost_round = tuned_params.get('n_estimators', xgb_params.get('n_estimators', 1000))
                early_stopping_rounds = getattr(self, 'xgb_early_stopping_rounds', 100)

                # Sanitize feature names (XGBoost native restriction: no [], <, > etc.)
                def _sanitize(name: str) -> str:
                    # Keep alnum and underscore, replace others with '_'
                    import re
                    sanitized = re.sub(r'[^0-9a-zA-Z_]', '_', name)
                    # Collapse multiple underscores
                    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
                    if not sanitized:
                        sanitized = 'f'
                    return sanitized

                sanitized = []
                used = set()
                orig_to_san = {}
                san_to_orig = {}
                for feat in non_constant_features:
                    s = _sanitize(feat)
                    base = s
                    idx = 1
                    # Ensure uniqueness
                    while s in used:
                        s = f"{base}_{idx}"
                        idx += 1
                    used.add(s)
                    sanitized.append(s)
                    orig_to_san[feat] = s
                    san_to_orig[s] = feat

                # Store mapping for inference
                self.xgb_feature_maps[pair] = {
                    'orig_to_sanitized': orig_to_san,
                    'sanitized_to_orig': san_to_orig,
                    'ordered_original': non_constant_features,
                    'ordered_sanitized': sanitized
                }

                # DMatrix with sanitized feature names
                dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=sanitized)
                dvalid = xgb.DMatrix(X_test, label=y_test, feature_names=sanitized)
                evals = [(dtrain, 'train'), (dvalid, 'validation')]

                booster = xgb.train(
                    booster_params,
                    dtrain,
                    num_boost_round=num_boost_round,
                    evals=evals,
                    early_stopping_rounds=early_stopping_rounds,
                    verbose_eval=False
                )
                # Orientation sanity check
                try:
                    pred_slice = booster.predict(dtrain)[:500]
                    y_slice = y_train.values[:500]
                    if len(pred_slice) > 10:
                        import numpy as np
                        corr = float(np.corrcoef(pred_slice, y_slice)[0, 1])
                        logger.info(
                            f"Orientation check corr(pred,y)={corr:.3f} (pos_ratio_train={positive_ratio:.3f}) for {pair}")
                except Exception as oc_e:
                    logger.debug(f"Orientation check failed: {oc_e}")

                # Predictions
                train_proba = booster.predict(dtrain)
                test_proba = booster.predict(dvalid)
                from sklearn.metrics import roc_auc_score, accuracy_score
                try:
                    train_auc = roc_auc_score(y_train, train_proba)
                    test_auc = roc_auc_score(y_test, test_proba)
                except Exception:
                    train_auc = test_auc = None
                train_pred = (train_proba >= 0.5).astype(int)
                test_pred = (test_proba >= 0.5).astype(int)
                train_score = accuracy_score(y_train, train_pred)
                test_score = accuracy_score(y_test, test_pred)

                # Feature importance (gain)
                raw_fi = booster.get_score(importance_type='gain')
                feature_importance = {}
                # Keys now are sanitized names; map back
                for s_name, val in raw_fi.items():
                    orig = san_to_orig.get(s_name, s_name)
                    feature_importance[orig] = val
                
                models['xgboost_enhanced'] = booster
                training_results['xgboost_enhanced'] = {
                    'train_accuracy': train_score,
                    'test_accuracy': test_score,
                    'train_auc': train_auc,
                    'test_auc': test_auc,
                    'gpu_enabled': booster_params.get('device') == 'cuda',
                    'feature_importance': feature_importance,
                    'best_iteration': getattr(booster, 'best_iteration', None),
                    'best_score': getattr(booster, 'best_score', None),
                    'num_boost_round': num_boost_round,
                    'early_stopping_rounds': early_stopping_rounds
                }

                logger.info(
                    f"XGB(native) trained {pair} acc(train/test)={train_score:.3f}/{test_score:.3f} "
                    f"AUC(train/test)={(train_auc or 0):.3f}/{(test_auc or 0):.3f} best_iter={getattr(booster, 'best_iteration', 'n/a')}"
                )
                # Capture peak GPU memory after XGBoost if on GPU
                if self.use_gpu and GPU_AVAILABLE:
                    try:
                        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
                        prev = self.gpu_memory_usage.get(pair, 0)
                        if peak_mem > prev:
                            self.gpu_memory_usage[pair] = peak_mem
                    except Exception:
                        pass
                
            except Exception as e:
                logger.error(f"XGBoost training failed for {pair}: {e}")
                # NO FALLBACK - Return error immediately
                return {
                    'status': 'error', 
                    'error': f'XGBoost training failed: {str(e)}',
                    'pair': pair,
                    'gpu_attempted': xgb_params.get('device') == 'cuda'
                }
            
            # Secondary XGBoost removed (single-model focus per user request)
            
            # === TRAINING SUMMARY AND CLEANUP ===
            logger.info(f"Training completed for {pair}")
            logger.info(f"Models trained: {list(models.keys())}")
            
            # Store models and metadata
            self.models[pair] = models
            self.scalers[pair] = scaler
            self.feature_importance[pair] = {
                'feature_columns': non_constant_features,
                **{f'{model_name}_importance': result.get('feature_importance', {}) 
                   for model_name, result in training_results.items()}
            }
            self.is_trained[pair] = True
            
            # Save to disk
            self._save_models_to_disk(pair)
            
            training_time = time.time() - start_time
            self.training_times[pair] = training_time
            
            # Enhanced result summary
            result = {
                'status': 'success',
                'models_trained': list(models.keys()),
                'training_time': training_time,
                'training_results': training_results,
                'feature_columns': non_constant_features,
                'gpu_enabled': self.use_gpu,
                'samples_used': len(X_scaled),
                'features_used': non_constant_features,
                'intensive_mode': True,
                'gpu_memory_peak': self.gpu_memory_usage.get(pair, 0)
            }
            
            # Performance logging
            logger.info(f"🚀 INTENSIVE Training completed for {pair} in {training_time:.2f}s")
            logger.info(f"🤖 Models trained: {list(models.keys())}")
            # Save results
            if models:
                self.models[pair] = models
                self.scalers[pair] = scaler
                self.feature_importance[pair] = training_results
                
                # CRITICAL: Save the exact feature columns used for training
                self.feature_importance[pair]['feature_columns'] = non_constant_features
                
                self.is_trained[pair] = True
                self.last_train_time[pair] = datetime.utcnow()
                self.last_train_index[pair] = len(df)
                
                # Save to disk
                self._save_models_to_disk(pair)
                
                training_time = time.time() - start_time
                self.training_times[pair] = training_time
                
                # Enhanced performance monitoring for intensive training
                monitor_training_performance(pair, training_time, 
                                           self.gpu_memory_usage.get(pair, 0))
                
                logger.info(f"🚀 INTENSIVE Training completed for {pair} in {training_time:.2f}s")
                logger.info(f"🤖 Models trained: {list(models.keys())}")
                logger.info(f"📊 Features used: {len(non_constant_features)}")
                logger.info(f"💾 Samples processed: {len(X):,}")
                logger.info(f"⚡ GPU acceleration: {'ENABLED' if self.use_gpu else 'DISABLED'}")

                # GPU cleanup after intensive training
                if self.use_gpu:
                    self.cleanup_gpu_memory()

                return {
                    'status': 'success',
                    'models_trained': list(models.keys()),
                    'training_time': training_time,
                    'training_results': training_results,
                    'gpu_enabled': self.use_gpu,
                    'samples_used': len(X),
                    'features_used': non_constant_features,
                    'intensive_mode': True,
                    'gpu_memory_peak': self.gpu_memory_usage.get(pair, 0)
                }
            else:
                return {'status': 'no_models_trained'}

        except Exception as e:
            logger.error(f"Training failed for {pair}: {e}")
            return {'status': 'error', 'error': str(e)}

    def _get_feature_importance(self, model, feature_columns):
        """Extract feature importance from different model types"""
        try:
            if hasattr(model, 'feature_importances_'):
                return dict(zip(feature_columns, model.feature_importances_, strict=True))
            elif hasattr(model, 'coef_'):
                # For linear models like LogisticRegression
                importance = abs(model.coef_[0])
                return dict(zip(feature_columns, importance, strict=True))
            else:
                return {}
        except Exception:
            return {}
    
    def predict_entry_probability(self, df: pd.DataFrame, pair: str) -> pd.Series:
        """Enhanced prediction with balanced long/short probabilities"""
        if pair not in self.is_trained or not self.is_trained[pair]:
            logger.warning(f"Model not trained for {pair}, returning neutral predictions")
            return pd.Series([0.5] * len(df), index=df.index)
            
        # Track prediction statistics for this pair
        if pair not in self.prediction_history:
            self.prediction_history[pair] = {
                'last_100_predictions': [],
                'long_short_ratio': 1.0
            }
        
        try:
            # Extract features
            feature_df = self.extract_advanced_features(df)
            
            # Get the EXACT features used during training
            if (pair not in self.scalers or 
                pair not in self.feature_importance or 
                'feature_columns' not in self.feature_importance[pair]):
                logger.warning(f"Training features not found for {pair}, using fallback")
                return pd.Series([0.5] * len(df), index=df.index)
            
            # Get the exact feature columns used in training
            training_feature_columns = self.feature_importance[pair]['feature_columns']
            scaler = self.scalers[pair]
            
            # Check which features are missing in current dataframe
            missing_features = [col for col in training_feature_columns if col not in feature_df.columns]
            if missing_features:
                logger.warning(f"Missing features for {pair}: {missing_features[:5]}...")
                # Add missing features with default values
                for col in missing_features:
                    feature_df[col] = 0.0
            
            # Use ONLY the features from training, in the same order
            X = feature_df[training_feature_columns].fillna(0)
            # Scale features using the same scaler from training
            X_scaled = scaler.transform(X)

            predictions = []
            models = self.models[pair]
            
            # Calculate prediction bias adjustment
            hist = self.prediction_history[pair]['last_100_predictions']
            if len(hist) >= 100:
                mean_pred = np.mean(hist)
                # Adjust for bias if predictions are consistently skewed
                bias_adjustment = 0.5 - mean_pred
                self.prediction_history[pair]['bias_adjustment'] = bias_adjustment
            else:
                bias_adjustment = self.prediction_history[pair].get('bias_adjustment', 0.0)

            # GPU Neural Network prediction
            if 'neural_network_gpu' in models and self.use_gpu:
                try:
                    gpu_trainer = models['neural_network_gpu']
                    gpu_pred = gpu_trainer.predict(X_scaled, batch_size=1024)
                    predictions.append(gpu_pred)
                    logger.debug(
                        f"GPU NN prediction for {pair}: {gpu_pred[-1]:.3f}"
                    )
                except Exception as e:
                    logger.warning(f"GPU prediction failed for {pair}: {e}")

            # XGBoost native booster prediction (preferred)
            if 'xgboost_enhanced' in models:
                try:
                    booster = models['xgboost_enhanced']
                    # Detect if it's native Booster (no predict_proba attribute)
                    if hasattr(booster, 'predict_proba'):
                        # Legacy path (sklearn wrapper)
                        xgb_pred = booster.predict_proba(X_scaled)[:, 1]
                    else:
                        import xgboost as xgb  # Local import
                        # Retrieve feature map for pair
                        fmap = self.xgb_feature_maps.get(pair)
                        if fmap and 'ordered_sanitized' in fmap:
                            # Build DMatrix with sanitized order
                            san_order = fmap['ordered_sanitized']
                            orig_order = fmap['ordered_original']
                            # Re-order scaled features to original order used at training
                            X_reordered = X[orig_order].values
                            dmatrix = xgb.DMatrix(
                                X_reordered,
                                feature_names=san_order
                            )
                            xgb_pred = booster.predict(dmatrix)
                        else:
                            # Fallback: direct predict with numeric feature names
                            dmatrix = xgb.DMatrix(X_scaled)
                            xgb_pred = booster.predict(dmatrix)
                    predictions.append(xgb_pred)
                except Exception as e:
                    logger.warning(
                        f"Primary XGBoost prediction failed for {pair}: {e}"
                    )

            # Ensemble prediction - Single XGBoost (optionally with GPU NN)
            if predictions:
                # Dynamic weighted ensemble based on available models
                weights = []
                confidence_scores = []
                
                # GPU Neural Network (40% base weight)
                if 'neural_network_gpu' in models and len(predictions) > 0:
                    weights.append(0.4)
                    if len(predictions) > 0:
                        confidence_scores.append(np.std(predictions[0]))  # First prediction (GPU)
                
                # Primary XGBoost (40% base weight) 
                if 'xgboost_enhanced' in models:
                    weights.append(0.4)
                    # Find XGBoost prediction index
                    xgb_idx = 1 if 'neural_network_gpu' in models else 0
                    if xgb_idx < len(predictions):
                        confidence_scores.append(np.std(predictions[xgb_idx]))
                
                # Adjust weights for available models (max 2)
                if len(predictions) == 1:
                    weights = [1.0]
                elif len(predictions) == 2:
                    # Favor XGBoost over NN when both
                    weights = [0.3, 0.7] if 'neural_network_gpu' in models else [0.5, 0.5]
                
                # Normalize weights
                weights = np.array(weights[:len(predictions)])
                weights = weights / weights.sum()
                
                # Apply confidence-based weight adjustment
                if confidence_scores and len(confidence_scores) == len(weights):
                    # Lower standard deviation = higher confidence = higher weight
                    confidence_weights = 1.0 / (np.array(confidence_scores) + 1e-6)
                    confidence_weights = confidence_weights / confidence_weights.sum()
                    # Blend 70% original weights + 30% confidence weights
                    weights = 0.7 * weights + 0.3 * confidence_weights
                    weights = weights / weights.sum()  # Re-normalize
                
                # Weighted average with stability enhancement
                ensemble_pred = np.average(predictions, axis=0, weights=weights)
                
                # Apply prediction calibration
                ensemble_pred = np.clip(ensemble_pred, 0.02, 0.98)  # Prevent extreme values
                
                # Light smoothing for stability (only if enough data points)
                if len(ensemble_pred) > 5:
                    smoothed = np.convolve(ensemble_pred, np.ones(3)/3, mode='same')
                    ensemble_pred = 0.85 * ensemble_pred + 0.15 * smoothed
                
                logger.debug(f"OPTIMIZED Ensemble prediction for {pair}: weights={weights}, pred={ensemble_pred[-1]:.3f}")
                
                return pd.Series(ensemble_pred, index=df.index)
            else:
                return pd.Series([0.5] * len(df), index=df.index)
                
        except Exception as e:
            logger.warning(f"Prediction failed for {pair}: {e}")
            return pd.Series([0.5] * len(df), index=df.index)

    def cleanup_gpu_memory(self):
        """Cleanup GPU memory to prevent OOM errors"""
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")

    def get_gpu_status(self) -> dict:
        """Get current GPU status and memory usage"""
        status = {
            'gpu_available': GPU_AVAILABLE,
            'torch_available': TORCH_AVAILABLE,
            'xgb_gpu_available': XGB_GPU_AVAILABLE,
            'device': str(self.device),
            'training_times': self.training_times,
            'gpu_memory_usage': self.gpu_memory_usage
        }
        
        if GPU_AVAILABLE:
            status['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**2  # MB
            status['gpu_memory_reserved'] = torch.cuda.memory_reserved() / 1024**2  # MB
            status['gpu_device_name'] = torch.cuda.get_device_name(0)
        
        return status
    
    def _detect_market_condition(self, df: pd.DataFrame) -> str:
        """Detect current market condition for dynamic model selection"""
        try:
            # Simple market regime detection
            recent_data = df.tail(20)
            
            if len(recent_data) < 10:
                return 'unknown'
            
            # Calculate volatility
            returns = recent_data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Calculate trend strength
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            if volatility > 0.03:  # High volatility threshold
                return 'volatile'
            elif abs(price_change) > 0.05:  # Strong trend threshold
                return 'trending'
            else:
                return 'ranging'
                
        except Exception:
            return 'unknown'


# Initialize the predictive engine globally
predictive_engine = AdvancedPredictiveEngine()


def calculate_advanced_predictive_signals(dataframe: pd.DataFrame, pair: str) -> pd.DataFrame:
    """GPU-Enhanced predictive signals with optimized memory management"""
    try:
        # Clear GPU cache before training
        if predictive_engine.use_gpu:
            predictive_engine.cleanup_gpu_memory()
        
        need_training = False
        assets_exist = False
        
        try:
            predictive_engine.mark_trained_if_assets(pair)
            assets_exist = predictive_engine._assets_exist(pair)
        except Exception:
            assets_exist = False

        # Check time since strategy startup
        now_utc = datetime.utcnow()
        hours_since_startup = (now_utc - predictive_engine.strategy_start_time).total_seconds() / 3600.0

        if not assets_exist:
            need_training = True
            logger.info(f"GPU Training needed for {pair} - no existing models")
        elif (predictive_engine.enable_startup_retrain and 
              hours_since_startup >= predictive_engine.retrain_after_startup_hours):
            need_training = True
            logger.info(f"GPU Retraining needed for {pair} - {hours_since_startup:.1f}h since startup")

        if need_training:
            logger.info(f"Starting GPU-enhanced training for {pair}")
            training_result = predictive_engine.train_predictive_models(dataframe, pair)
            
            if training_result.get('status') == 'success':
                # Log only concise success message without full result details
                logger.info(f"GPU Training successful for {pair}")
            else:
                logger.warning(f"GPU Training failed for {pair}: {training_result}")
        
        # Enhanced ML probability prediction with GPU acceleration
        dataframe['ml_entry_probability'] = predictive_engine.predict_entry_probability(dataframe, pair)
        
        # Debug prediction values for analysis
        if pair in ['BTC/USDT:USDT', 'ETH/USDT:USDT']:
            recent_predictions = dataframe['ml_entry_probability'].tail(10)
            logger.info(f"🤖 ML PREDICTIONS {pair} | last_10=[{', '.join([f'{x:.3f}' for x in recent_predictions])}] "
                       f"mean={recent_predictions.mean():.3f} trained={predictive_engine.is_trained.get(pair, False)}")
        
        # Get momentum and volatility regime safely
        momentum_regime = dataframe.get('momentum_regime')
        volatility_regime = dataframe.get('volatility_regime')
        quantum_coherence = dataframe.get('quantum_momentum_coherence')
        neural_pattern = dataframe.get('neural_pattern_score')
        
        # Advanced confidence scoring with safe comparisons
        ml_high_conf_conditions = (dataframe['ml_entry_probability'] > 0.8)
        
        if momentum_regime is not None:
            ml_high_conf_conditions &= (momentum_regime > 0)
        
        if volatility_regime is not None:
            ml_high_conf_conditions &= (volatility_regime < 2)  # Not high volatility
            
        dataframe['ml_high_confidence'] = ml_high_conf_conditions.astype(int)
        
        # Ultra-high confidence entries with safe checks
        ml_ultra_conf_conditions = (dataframe['ml_entry_probability'] > 0.9)
        
        if quantum_coherence is not None:
            ml_ultra_conf_conditions &= (quantum_coherence > 0.7)
        else:
            ml_ultra_conf_conditions &= (dataframe['rsi'].between(30, 70))
            
        if neural_pattern is not None:
            ml_ultra_conf_conditions &= (neural_pattern > 0.7)
        else:
            ml_ultra_conf_conditions &= (dataframe['volume'] > dataframe['avg_volume'])
            
        dataframe['ml_ultra_confidence'] = ml_ultra_conf_conditions.astype(int)
        
        # Enhanced score combination
        if 'ultimate_score' in dataframe.columns:
            dataframe['ml_enhanced_score'] = (
                dataframe['ml_entry_probability'] * 0.6 +
                dataframe['ultimate_score'] * 0.4
            )
        else:
            dataframe['ml_enhanced_score'] = dataframe['ml_entry_probability']
        
        # Model agreement indicator
        if pair in predictive_engine.models and len(predictive_engine.models[pair]) > 1:
            # Calculate agreement between models
            model_count = len(predictive_engine.models[pair])
            if model_count >= 2:
                # High agreement when multiple models agree
                agreement_score = 0.8 if dataframe['ml_entry_probability'].iloc[-1] > 0.6 else 0.3
                dataframe['ml_model_agreement'] = agreement_score
            else:
                dataframe['ml_model_agreement'] = 0.7
        else:
            dataframe['ml_model_agreement'] = 0.5
        
        # Clean up GPU memory after processing
        if predictive_engine.use_gpu:
            predictive_engine.cleanup_gpu_memory()
        
        return dataframe
        
    except Exception as e:
        logger.warning(f"GPU-Enhanced predictive analysis failed for {pair}: {e}")
        dataframe['ml_entry_probability'] = 0.5
        dataframe['ml_enhanced_score'] = dataframe.get('ultimate_score', 0.5)
        dataframe['ml_high_confidence'] = 0
        dataframe['ml_ultra_confidence'] = 0
        dataframe['ml_model_agreement'] = 0.5
        return dataframe
        dataframe['ml_high_confidence'] = 0
        dataframe['ml_ultra_confidence'] = 0
        dataframe['ml_model_agreement'] = 0.5
        return dataframe


def calculate_quantum_momentum_analysis(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Quantum-inspired momentum analysis for ultra-precise predictions"""
    try:
        momentum_periods = [3, 5, 8, 13, 21, 34]
        momentum_matrix = pd.DataFrame()
        
        for period in momentum_periods:
            momentum_matrix[f'mom_{period}'] = dataframe['close'].pct_change(period)
        
        dataframe['quantum_momentum_coherence'] = (
            momentum_matrix.std(axis=1) / (momentum_matrix.mean(axis=1).abs() + 1e-10)
        )
        
        # Calculate momentum entanglement using correlation matrix
        def calculate_entanglement(window_data):
            if len(window_data) < 10:
                return 0
            try:
                corr_matrix = window_data.corr()
                if corr_matrix.empty or corr_matrix.isna().all().all():
                    return 0
                # Get upper triangular correlation values (excluding diagonal)
                upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
                correlations = corr_matrix.values[upper_tri_indices]
                # Remove NaN values and calculate mean
                valid_correlations = correlations[~np.isnan(correlations)]
                return valid_correlations.mean() if len(valid_correlations) > 0 else 0
            except Exception:
                return 0
        
        entanglement_values = []
        for i in range(len(momentum_matrix)):
            if i < 20:
                entanglement_values.append(0.5)
            else:
                window_data = momentum_matrix.iloc[i-19:i+1]
                entanglement = calculate_entanglement(window_data)
                entanglement_values.append(entanglement)
        
        dataframe['momentum_entanglement'] = pd.Series(entanglement_values, index=dataframe.index)
        
        price_uncertainty = dataframe['close'].rolling(20).std()
        momentum_uncertainty = momentum_matrix['mom_8'].rolling(20).std()
        dataframe['heisenberg_uncertainty'] = price_uncertainty * momentum_uncertainty
        
        if 'maxima_sort_threshold' in dataframe.columns:
            resistance_distance = (
                dataframe['maxima_sort_threshold'] - dataframe['close']) / dataframe['close']
            dataframe['quantum_tunnel_up_prob'] = np.exp(-resistance_distance.abs() * 10)
        else:
            dataframe['quantum_tunnel_up_prob'] = 0.5
        
        if 'minima_sort_threshold' in dataframe.columns:
            support_distance = (
                dataframe['close'] - dataframe['minima_sort_threshold']) / dataframe['close']
            dataframe['quantum_tunnel_down_prob'] = np.exp(-support_distance.abs() * 10)
        else:
            dataframe['quantum_tunnel_down_prob'] = 0.5
        
        return dataframe
        
    except Exception as e:
        logger.warning(f"Quantum momentum analysis failed: {e}")
        dataframe['quantum_momentum_coherence'] = 0.5
        dataframe['momentum_entanglement'] = 0.5
        dataframe['heisenberg_uncertainty'] = 1.0
        dataframe['quantum_tunnel_up_prob'] = 0.5
        dataframe['quantum_tunnel_down_prob'] = 0.5
        return dataframe


def calculate_neural_pattern_recognition(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Neural pattern recognition for complex market patterns"""
    try:
        dataframe['body_size'] = abs(dataframe['close'] - dataframe['open']) / dataframe['close']
        dataframe['upper_shadow'] = (
            dataframe['high'] - np.maximum(dataframe['open'], dataframe['close'])
        ) / dataframe['close']
        dataframe['lower_shadow'] = (
            np.minimum(dataframe['open'], dataframe['close']) - dataframe['low']
        ) / dataframe['close']
        dataframe['candle_range'] = (dataframe['high'] - dataframe['low']) / dataframe['close']

        pattern_memory = []
        for i in range(len(dataframe)):
            if i < 5:
                pattern_memory.append(0)
                continue

            recent_patterns = dataframe[['body_size', 'upper_shadow', 'lower_shadow']].iloc[i-4:i+1]
            pattern_signature = recent_patterns.values.flatten()
            pattern_norm = np.linalg.norm(pattern_signature)

            if pattern_norm > 0:
                pattern_score = min(1.0, pattern_norm / 0.1)
            else:
                pattern_score = 0

            pattern_memory.append(pattern_score)

        dataframe['neural_pattern_score'] = pd.Series(pattern_memory, index=dataframe.index)
        dataframe['pattern_prediction_confidence'] = dataframe['neural_pattern_score'].rolling(10).std()

        return dataframe

    except Exception as e:
        logger.warning(f"Neural pattern recognition failed: {e}")
        dataframe['neural_pattern_score'] = 0.5
        dataframe['pattern_prediction_confidence'] = 0.5
        dataframe['body_size'] = 0.01
        dataframe['candle_range'] = 0.02
        return dataframe

def calculate_exit_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced exit signals based on market deterioration"""
    # === MOMENTUM DETERIORATION ===
    dataframe['momentum_deteriorating'] = (
        (dataframe['momentum_quality'] < dataframe['momentum_quality'].shift(1)) &
        (dataframe['momentum_acceleration'] < 0) &
        (dataframe['price_momentum'] < dataframe['price_momentum'].shift(1))
    ).astype(int)

    # === VOLUME DETERIORATION ===
    dataframe['volume_deteriorating'] = (
        (dataframe['volume_strength'] < 0.8) &
        (dataframe['selling_pressure'] > dataframe['buying_pressure']) &
        (dataframe['volume_pressure'] < 0)
    ).astype(int)

    # === STRUCTURE DETERIORATION ===
    dataframe['structure_deteriorating'] = (
        (dataframe['structure_score'] < -1) &
        (dataframe['bearish_structure'] > dataframe['bullish_structure']) &
        (dataframe['structure_break_down'] == 1)
    ).astype(int)

    # === CONFLUENCE BREAKDOWN ===
    dataframe['confluence_breakdown'] = (
        (dataframe['confluence_score'] < 2) &
        (dataframe['near_resistance'] == 1) &
        (dataframe['volume_spike'] == 0)
    ).astype(int)

    # === TREND WEAKNESS ===
    dataframe['trend_weakening'] = (
        (dataframe['trend_strength'] < 0) &
        (dataframe['close'] < dataframe['ema50']) &
        (dataframe['strong_downtrend'] == 1)
    ).astype(int)

    # === ULTIMATE EXIT SCORE ===
    dataframe['exit_pressure'] = (
        dataframe['momentum_deteriorating'] * 2 +
        dataframe['volume_deteriorating'] * 2 +
        dataframe['structure_deteriorating'] * 2 +
        dataframe['confluence_breakdown'] * 1 +
        dataframe['trend_weakening'] * 1
    )

    # === RSI OVERBOUGHT WITH DIVERGENCE ===
    dataframe['rsi_exit_signal'] = (
        (dataframe['rsi'] > 75) &
        (
            (dataframe['rsi_divergence_bear'] == 1) |
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
            (dataframe['close'] < dataframe['close'].shift(1))
        )
    ).astype(int)

    # === PROFIT TAKING LEVELS ===
    mml_resistance_levels = ['[6/8]P', '[8/8]P']
    dataframe['near_resistance_level'] = 0

    for level in mml_resistance_levels:
        if level in dataframe.columns:
            near_level = (
                (dataframe['close'] >= dataframe[level] * 0.99) &
                (dataframe['close'] <= dataframe[level] * 1.02)
            ).astype(int)
            dataframe['near_resistance_level'] += near_level

    # === VOLATILITY SPIKE EXIT ===
    dataframe['volatility_spike'] = (
        dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 1.5
    ).astype(int)

    # === EXHAUSTION SIGNALS ===
    dataframe['bullish_exhaustion'] = (
        (dataframe['consecutive_green'] >= 4) &
        (dataframe['rsi'] > 70) &
        (dataframe['volume'] < dataframe['avg_volume'] * 0.8) &
        (dataframe['momentum_acceleration'] < 0)
    ).astype(int)

    return dataframe


def calculate_dynamic_profit_targets(dataframe: pd.DataFrame, entry_type_col: str = 'entry_type') -> pd.DataFrame:
    """Calculate dynamic profit targets based on entry quality and market conditions"""

    # Base profit targets based on ATR
    dataframe['base_profit_target'] = dataframe['atr'] * 2

    # Adjust based on entry type
    dataframe['profit_multiplier'] = 1.0
    if entry_type_col in dataframe.columns:
        dataframe.loc[dataframe[entry_type_col] == 3, 'profit_multiplier'] = 2.0  # High quality
        dataframe.loc[dataframe[entry_type_col] == 2, 'profit_multiplier'] = 1.5  # Medium quality
        dataframe.loc[dataframe[entry_type_col] == 1, 'profit_multiplier'] = 1.2  # Backup
        dataframe.loc[dataframe[entry_type_col] == 4, 'profit_multiplier'] = 2.5  # Breakout
        dataframe.loc[dataframe[entry_type_col] == 5, 'profit_multiplier'] = 1.8  # Reversal

    # Final profit target
    dataframe['dynamic_profit_target'] = dataframe['base_profit_target'] * dataframe['profit_multiplier']

    return dataframe


def calculate_advanced_stop_loss(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['base_stop_loss'] = dataframe['atr'] * 1.5
    if 'minima_sort_threshold' in dataframe.columns:
        dataframe['support_stop_loss'] = dataframe['close'] - dataframe['minima_sort_threshold']
        dataframe['support_stop_loss'] = dataframe['support_stop_loss'].clip(
            dataframe['base_stop_loss'] * 0.5,
            dataframe['base_stop_loss'] * 1.5  # Reduced from 2.0
        )
        dataframe['final_stop_loss'] = np.minimum(
            dataframe['base_stop_loss'],
            dataframe['support_stop_loss']
        ).clip(-0.15, -0.01)  # Hard cap at -15%
    else:
        dataframe['final_stop_loss'] = dataframe['base_stop_loss'].clip(-0.15, -0.01)
    return dataframe

def calculate_confluence_score(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Multi-factor confluence analysis - much better than BTC correlation"""

    # Support/Resistance Confluence
    dataframe['near_support'] = (
        (dataframe['close'] <= dataframe['minima_sort_threshold'] * 1.02) &
        (dataframe['close'] >= dataframe['minima_sort_threshold'] * 0.98)
    ).astype(int)

    dataframe['near_resistance'] = (
        (dataframe['close'] <= dataframe['maxima_sort_threshold'] * 1.02) &
        (dataframe['close'] >= dataframe['maxima_sort_threshold'] * 0.98)
    ).astype(int)

    # MML Level Confluence
    mml_levels = ['[0/8]P', '[2/8]P', '[4/8]P', '[6/8]P', '[8/8]P']
    dataframe['near_mml'] = 0

    for level in mml_levels:
        if level in dataframe.columns:
            near_level = (
                (dataframe['close'] <= dataframe[level] * 1.015) &
                (dataframe['close'] >= dataframe[level] * 0.985)
            ).astype(int)
            dataframe['near_mml'] += near_level

    # Volume Confluence
    dataframe['volume_spike'] = (
        dataframe['volume'] > dataframe['avg_volume'] * 1.5
    ).astype(int)

    # RSI Confluence Zones
    dataframe['rsi_oversold'] = (dataframe['rsi'] < 30).astype(int)
    dataframe['rsi_overbought'] = (dataframe['rsi'] > 70).astype(int)
    dataframe['rsi_neutral'] = (
        (dataframe['rsi'] >= 40) & (dataframe['rsi'] <= 60)
    ).astype(int)

    # EMA Confluence
    dataframe['above_ema'] = (dataframe['close'] > dataframe['ema50']).astype(int)

    # CONFLUENCE SCORE (0-6)
    dataframe['confluence_score'] = (
        dataframe['near_support'] +
        dataframe['near_mml'].clip(0, 2) +  # Max 2 points for MML
        dataframe['volume_spike'] +
        dataframe['rsi_oversold'] +
        dataframe['above_ema'] +
        (dataframe['trend_strength'] > 0.01).astype(int)  # Positive trend
    )

    return dataframe


def calculate_smart_volume(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Advanced volume analysis - beats any external correlation"""

    # Volume-Price Trend (VPT)
    price_change_pct = (dataframe['close'] - dataframe['close'].shift(1)) / dataframe['close'].shift(1)
    dataframe['vpt'] = (dataframe['volume'] * price_change_pct).fillna(0).cumsum()

    # Volume moving averages
    dataframe['volume_sma20'] = dataframe['volume'].rolling(20).mean()
    dataframe['volume_sma50'] = dataframe['volume'].rolling(50).mean()

    # Volume strength
    dataframe['volume_strength'] = dataframe['volume'] / dataframe['volume_sma20']

    # Smart money indicators
    dataframe['accumulation'] = (
        (dataframe['close'] > dataframe['open']) &  # Green candle
        (dataframe['volume'] > dataframe['volume_sma20'] * 1.2) &  # High volume
        (dataframe['close'] > (dataframe['high'] + dataframe['low']) / 2)  # Close in upper half
    ).astype(int)

    dataframe['distribution'] = (
        (dataframe['close'] < dataframe['open']) &  # Red candle
        (dataframe['volume'] > dataframe['volume_sma20'] * 1.2) &  # High volume
        (dataframe['close'] < (dataframe['high'] + dataframe['low']) / 2)  # Close in lower half
    ).astype(int)

    # Buying/Selling pressure
    dataframe['buying_pressure'] = dataframe['accumulation'].rolling(5).sum()
    dataframe['selling_pressure'] = dataframe['distribution'].rolling(5).sum()

    # Net volume pressure
    dataframe['volume_pressure'] = dataframe['buying_pressure'] - dataframe['selling_pressure']

    # Volume trend
    dataframe['volume_trend'] = (
        dataframe['volume_sma20'] > dataframe['volume_sma50']
    ).astype(int)

    # Money flow
    typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
    money_flow = typical_price * dataframe['volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_flow_sum = positive_flow.rolling(14).sum()
    negative_flow_sum = negative_flow.rolling(14).sum()

    dataframe['money_flow_ratio'] = positive_flow_sum / (negative_flow_sum + 1e-10)
    dataframe['money_flow_index'] = 100 - (100 / (1 + dataframe['money_flow_ratio']))

    return dataframe


def calculate_advanced_momentum(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Multi-timeframe momentum system - superior to BTC correlation"""

    # Multi-timeframe momentum
    dataframe['momentum_3'] = dataframe['close'].pct_change(6)
    dataframe['momentum_7'] = dataframe['close'].pct_change(14)
    dataframe['momentum_14'] = dataframe['close'].pct_change(28)
    dataframe['momentum_21'] = dataframe['close'].pct_change(21)

    # Momentum acceleration
    dataframe['momentum_acceleration'] = (
        dataframe['momentum_3'] - dataframe['momentum_3'].shift(3)
    )

    # Momentum consistency
    dataframe['momentum_consistency'] = (
        (dataframe['momentum_3'] > 0).astype(int) +
        (dataframe['momentum_7'] > 0).astype(int) +
        (dataframe['momentum_14'] > 0).astype(int)
    )

    # Momentum divergence with volume
    dataframe['price_momentum_rank'] = dataframe['momentum_7'].rolling(20).rank(pct=True)
    dataframe['volume_momentum_rank'] = dataframe['volume_strength'].rolling(20).rank(pct=True)

    dataframe['momentum_divergence'] = (
        dataframe['price_momentum_rank'] - dataframe['volume_momentum_rank']
    ).abs()

    # Momentum strength
    dataframe['momentum_strength'] = (
        dataframe['momentum_3'].abs() +
        dataframe['momentum_7'].abs() +
        dataframe['momentum_14'].abs()
    ) / 3

    # Momentum quality score (0-5)
    dataframe['momentum_quality'] = (
        (dataframe['momentum_3'] > 0).astype(int) +
        (dataframe['momentum_7'] > 0).astype(int) +
        (dataframe['momentum_acceleration'] > 0).astype(int) +
        (dataframe['volume_strength'] > 1.1).astype(int) +
        (dataframe['momentum_divergence'] < 0.3).astype(int)
    )

    # Rate of Change
    dataframe['roc_5'] = dataframe['close'].pct_change(5) * 100
    dataframe['roc_10'] = dataframe['close'].pct_change(10) * 100
    dataframe['roc_20'] = dataframe['close'].pct_change(20) * 100

    # Momentum oscillator
    dataframe['momentum_oscillator'] = (
        dataframe['roc_5'] + dataframe['roc_10'] + dataframe['roc_20']
    ) / 3

    return dataframe


def calculate_market_structure(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Market structure analysis - intrinsic trend recognition"""

    # Higher highs, higher lows detection
    dataframe['higher_high'] = (
        (dataframe['high'] > dataframe['high'].shift(1)) &
        (dataframe['high'].shift(1) > dataframe['high'].shift(2))
    ).astype(int)

    dataframe['higher_low'] = (
        (dataframe['low'] > dataframe['low'].shift(1)) &
        (dataframe['low'].shift(1) > dataframe['low'].shift(2))
    ).astype(int)

    dataframe['lower_high'] = (
        (dataframe['high'] < dataframe['high'].shift(1)) &
        (dataframe['high'].shift(1) < dataframe['high'].shift(2))
    ).astype(int)

    dataframe['lower_low'] = (
        (dataframe['low'] < dataframe['low'].shift(1)) &
        (dataframe['low'].shift(1) < dataframe['low'].shift(2))
    ).astype(int)

    # Market structure scores
    dataframe['bullish_structure'] = (
        dataframe['higher_high'].rolling(5).sum() +
        dataframe['higher_low'].rolling(5).sum()
    )

    dataframe['bearish_structure'] = (
        dataframe['lower_high'].rolling(5).sum() +
        dataframe['lower_low'].rolling(5).sum()
    )

    dataframe['structure_score'] = (
        dataframe['bullish_structure'] - dataframe['bearish_structure']
    )

    # Swing highs and lows
    # Live-safe pivot detection without using future candles (no shift(-1))
    # Confirm swing at the PREVIOUS candle using only information up to current bar.
    # A swing high at t-1 is when high[t-1] > high[t-2] and high[t-1] > high[t].
    prev_high = dataframe['high'].shift(1)
    prev_low = dataframe['low'].shift(1)
    dataframe['swing_high'] = (
        (prev_high > dataframe['high'].shift(2)) &
        (prev_high > dataframe['high'])
    ).astype(int)

    # A swing low at t-1 is when low[t-1] < low[t-2] and low[t-1] < low[t].
    dataframe['swing_low'] = (
        (prev_low < dataframe['low'].shift(2)) &
        (prev_low < dataframe['low'])
    ).astype(int)

    # Market structure breaks
    # Use previous candle values where the swing was confirmed to avoid lookahead bias
    swing_highs = prev_high.where(dataframe['swing_high'] == 1)
    swing_lows = prev_low.where(dataframe['swing_low'] == 1)

    # Structure break detection
    dataframe['structure_break_up'] = (
        dataframe['close'] > swing_highs.ffill()
    ).astype(int)

    dataframe['structure_break_down'] = (
        dataframe['close'] < swing_lows.ffill()
    ).astype(int)

    # Trend strength based on structure
    dataframe['structure_trend_strength'] = (
        dataframe['structure_score'] / 10  # Normalize
    ).clip(-1, 1)

    # Support and resistance strength
    dataframe['support_strength'] = dataframe['swing_low'].rolling(20).sum()
    dataframe['resistance_strength'] = dataframe['swing_high'].rolling(20).sum()

    return dataframe


def calculate_advanced_entry_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Advanced entry signal generation"""

    # Multi-factor signal strength
    dataframe['signal_strength'] = 0

    # Confluence signals
    dataframe['confluence_signal'] = (dataframe['confluence_score'] >= 3).astype(int)
    dataframe['signal_strength'] += dataframe['confluence_signal'] * 2

    # Volume signals
    dataframe['volume_signal'] = (
        (dataframe['volume_pressure'] >= 2) &
        (dataframe['volume_strength'] > 1.2)
    ).astype(int)
    dataframe['signal_strength'] += dataframe['volume_signal'] * 2

    # Momentum signals
    dataframe['momentum_signal'] = (
        (dataframe['momentum_quality'] >= 3) &
        (dataframe['momentum_acceleration'] > 0)
    ).astype(int)
    dataframe['signal_strength'] += dataframe['momentum_signal'] * 2

    # Structure signals
    dataframe['structure_signal'] = (
        (dataframe['structure_score'] > 0) &
        (dataframe['structure_break_up'] == 1)
    ).astype(int)
    dataframe['signal_strength'] += dataframe['structure_signal'] * 1

    # RSI position signal
    dataframe['rsi_signal'] = (
        (dataframe['rsi'] > 30) & (dataframe['rsi'] < 70)
    ).astype(int)
    dataframe['signal_strength'] += dataframe['rsi_signal'] * 1

    # Trend alignment signal
    dataframe['trend_signal'] = (
        (dataframe['close'] > dataframe['ema50']) &
        (dataframe['trend_strength'] > 0)
    ).astype(int)
    dataframe['signal_strength'] += dataframe['trend_signal'] * 1

    # Money flow signal
    dataframe['money_flow_signal'] = (
        dataframe['money_flow_index'] > 50
    ).astype(int)
    dataframe['signal_strength'] += dataframe['money_flow_signal'] * 1

    return dataframe


class AlexNexusForgeV8AIGPU(IStrategy):

    # General strategy parameters
    timeframe = "1h"
    startup_candle_count: int = 1000
    stoploss = -0.11
    trailing_stop = True
    trailing_stop_positive = 0.005  # Trail at 0.5% below peak profit
    trailing_stop_positive_offset = 0.03  # Start trailing only at 3% profit
    trailing_only_offset_is_reached = True  # Ensure trailing only starts after offset is reached
    use_custom_stoploss = True
    can_short = True
    use_exit_signal = True
    ignore_roi_if_entry_signal = True
    process_only_new_candles = False
    use_custom_exits_advanced = True
    use_emergency_exits = True

    
    regime_change_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    regime_change_sensitivity = DecimalParameter(0.3, 0.8, default=0.5, decimals=2, space="sell", optimize=True, load=True)
    
    # Flash Move Detection
    flash_move_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    flash_move_threshold = DecimalParameter(0.03, 0.08, default=0.05, decimals=3, space="sell", optimize=True, load=True)
    flash_move_candles = IntParameter(3, 10, default=5, space="sell", optimize=True, load=True)
    
    # Volume Spike Detection
    volume_spike_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    volume_spike_multiplier = DecimalParameter(2.0, 5.0, default=3.0, decimals=1, space="sell", optimize=True, load=True)
    
    # Emergency Exit Protection
    emergency_exit_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    emergency_exit_profit_threshold = DecimalParameter(0.005, 0.03, default=0.015, decimals=3, space="sell", optimize=True, load=True)
    
    # Market Sentiment Protection
    sentiment_protection_enabled = BooleanParameter(default=True, space="sell", optimize=True, load=True)
    sentiment_shift_threshold = DecimalParameter(0.2, 0.4, default=0.3, decimals=2, space="sell", optimize=True, load=True)

    # 🔧ATR STOPLOSS PARAMETERS (Anpassbar machen)
    atr_stoploss_multiplier = DecimalParameter(0.8, 2.0, default=1.0, decimals=1, space="sell", optimize=True, load=True)
    atr_stoploss_minimum = DecimalParameter(-0.25, -0.10, default=-0.12, decimals=2, space="sell", optimize=True, load=True)
    atr_stoploss_maximum = DecimalParameter(-0.30, -0.15, default=-0.18, decimals=2, space="sell", optimize=True, load=True)
    atr_stoploss_ceiling = DecimalParameter(-0.10, -0.06, default=-0.06, decimals=2, space="sell", optimize=True, load=True)
    # DCA parameters
    initial_safety_order_trigger = DecimalParameter(
        low=-0.02, high=-0.01, default=-0.018, decimals=3, space="buy", optimize=True, load=True
    )
    max_safety_orders = IntParameter(1, 3, default=1, space="buy", optimize=True, load=True)
    safety_order_step_scale = DecimalParameter(
        low=1.05, high=1.5, default=1.25, decimals=2, space="buy", optimize=True, load=True
    )
    safety_order_volume_scale = DecimalParameter(
        low=1.1, high=2.0, default=1.4, decimals=1, space="buy", optimize=True, load=True
    )
    h2 = IntParameter(20, 60, default=40, space="buy", optimize=True, load=True)
    h1 = IntParameter(10, 40, default=20, space="buy", optimize=True, load=True)
    h0 = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)
    cp = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)

    # Entry parameters
    increment_for_unique_price = DecimalParameter(
        low=1.0005, high=1.002, default=1.001, decimals=4, space="buy", optimize=True, load=True
    )
    last_entry_price: Optional[float] = None

    # Protection parameters
    cooldown_lookback = IntParameter(2, 48, default=1, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=4, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # Murrey Math level parameters
    mml_const1 = DecimalParameter(1.0, 1.1, default=1.0699, decimals=4, space="buy", optimize=True, load=True)
    mml_const2 = DecimalParameter(0.99, 1.0, default=0.99875, decimals=5, space="buy", optimize=True, load=True)
    indicator_mml_window = IntParameter(32, 128, default=64, space="buy", optimize=True, load=True)

    # Dynamic Stoploss parameters
    # Add these parameters
    stoploss_atr_multiplier = DecimalParameter(1.0, 3.0, default=1.5, space="sell", optimize=True)
    stoploss_max_reasonable = DecimalParameter(-0.30, -0.15, default=-0.20, space="sell", optimize=True)

    # === Hyperopt Parameters ===
    dominance_threshold = IntParameter(1, 10, default=3, space="buy", optimize=True)
    tightness_factor = DecimalParameter(0.5, 2.0, default=1.0, space="buy", optimize=True)
    long_rsi_threshold = IntParameter(50, 65, default=50, space="buy", optimize=True)
    short_rsi_threshold = IntParameter(30, 45, default=35, space="sell", optimize=True)

    # Dynamic Leverage parameters
    leverage_window_size = IntParameter(20, 100, default=70, space="buy", optimize=True, load=True)
    leverage_base = DecimalParameter(5.0, 20.0, default=5.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_rsi_low = DecimalParameter(20.0, 40.0, default=30.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_rsi_high = DecimalParameter(60.0, 80.0, default=70.0, decimals=1, space="buy", optimize=True, load=True)
    leverage_long_increase_factor = DecimalParameter(1.1, 2.0, default=1.5, decimals=1, space="buy", optimize=True,
                                                     load=True)
    leverage_long_decrease_factor = DecimalParameter(0.3, 0.9, default=0.5, decimals=1, space="buy", optimize=True,
                                                     load=True)
    leverage_volatility_decrease_factor = DecimalParameter(0.5, 0.95, default=0.8, decimals=2, space="buy",
                                                           optimize=True, load=True)
    leverage_atr_threshold_pct = DecimalParameter(0.01, 0.05, default=0.03, decimals=3, space="buy", optimize=True,
                                                  load=True)

    # Indicator parameters
    indicator_extrema_order = IntParameter(3, 15, default=8, space="buy", optimize=True, load=True)  # War 5
    indicator_mml_window = IntParameter(50, 200, default=50, space="buy", optimize=True, load=True)  # War 50
    indicator_rolling_window_threshold = IntParameter(20, 100, default=50, space="buy", optimize=True, load=True)  # War 20
    indicator_rolling_check_window = IntParameter(5, 20, default=10, space="buy", optimize=True, load=True)  # War 5


    
    # Market breadth parameters
    market_breadth_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    market_breadth_threshold = DecimalParameter(0.3, 0.6, default=0.45, space="buy", optimize=True)
    
    # Total market cap parameters
    total_mcap_filter_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    total_mcap_ma_period = IntParameter(20, 100, default=50, space="buy", optimize=True)
    
    # Market regime parameters
    regime_filter_enabled = BooleanParameter(default=True, space="buy", optimize=True)
    regime_lookback_period = IntParameter(24, 168, default=48, space="buy", optimize=True)  # hours
    
    # Fear & Greed parameters
    fear_greed_enabled = BooleanParameter(default=False, space="buy", optimize=True)  # Optional
    fear_greed_extreme_threshold = IntParameter(20, 30, default=25, space="buy", optimize=True)
    fear_greed_greed_threshold = IntParameter(70, 80, default=75, space="buy", optimize=True)
    # Momentum
    avoid_strong_trends = BooleanParameter(default=True, space="buy", optimize=True)
    trend_strength_threshold = DecimalParameter(0.01, 0.05, default=0.02, space="buy", optimize=True)
    momentum_confirmation_candles = IntParameter(1, 5, default=2, space="buy", optimize=True)

    # Dynamic exit based on entry quality
    dynamic_exit_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    exit_on_confluence_loss = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    exit_on_structure_break = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    
    # Profit target multipliers based on entry type
    high_quality_profit_multiplier = DecimalParameter(1.2, 3.0, default=2.0, space="sell", optimize=True, load=True)
    medium_quality_profit_multiplier = DecimalParameter(1.0, 2.5, default=1.5, space="sell", optimize=True, load=True)
    backup_profit_multiplier = DecimalParameter(0.8, 2.0, default=1.2, space="sell", optimize=True, load=True)
    
    # Advanced exit thresholds
    volume_decline_exit_threshold = DecimalParameter(0.3, 0.8, default=0.5, space="sell", optimize=True, load=True)
    momentum_decline_exit_threshold = IntParameter(1, 4, default=2, space="sell", optimize=True, load=True)
    structure_deterioration_threshold = DecimalParameter(-3.0, 0.0, default=-1.5, space="sell", optimize=True, load=True)
    
    # RSI exit levels
    rsi_overbought_exit = IntParameter(70, 85, default=75, space="sell", optimize=True, load=True)
    rsi_divergence_exit_enabled = BooleanParameter(default=True, space="sell", optimize=False, load=True)
    
    # Trailing stop improvements
    use_advanced_trailing = BooleanParameter(default=False, space="sell", optimize=False, load=True)
    trailing_stop_positive_offset_high_quality = DecimalParameter(0.02, 0.08, default=0.04, space="sell", optimize=True, load=True)
    trailing_stop_positive_offset_medium_quality = DecimalParameter(0.015, 0.06, default=0.03, space="sell", optimize=True, load=True)
    
    # === NEUE ADVANCED PARAMETERS ===
    # Confluence Analysis
    confluence_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    confluence_threshold = DecimalParameter(2.0, 4.0, default=2.5, space="buy", optimize=True, load=True)  # War 3.0
    
    # Volume Analysis
    volume_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    volume_strength_threshold = DecimalParameter(1.1, 2.0, default=1.3, space="buy", optimize=True, load=True)
    volume_pressure_threshold = IntParameter(1, 3, default=1, space="buy", optimize=True, load=True)  # War 2

    
    # Momentum Analysis
    momentum_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    momentum_quality_threshold = IntParameter(2, 4, default=2, space="buy", optimize=True, load=True)  # War 3
    
    # Market Structure Analysis
    structure_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    structure_score_threshold = DecimalParameter(-2.0, 5.0, default=0.5, space="buy", optimize=True, load=True)
    
    # Ultimate Score
    ultimate_score_threshold = DecimalParameter(0.5, 3.0, default=1.5, space="buy", optimize=True, load=True)
    
    # Advanced Entry Filters
    require_volume_confirmation = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    require_momentum_confirmation = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    require_structure_confirmation = BooleanParameter(default=True, space="buy", optimize=False, load=True)

    # ✅ Replace your old ROI with this:
    minimal_roi = {
        "0": 0.06,
        "5": 0.055,
        "10": 0.04,
        "20": 0.03,
        "40": 0.025,
        "80": 0.02,
        "160": 0.015,
        "320": 0.01
    }

    # Plot configuration for backtesting UI
    plot_config = {
        "main_plot": {
            # Trend indicators
            "ema50": {"color": "gray", "type": "line"},
            
            # Support/Resistance
            "minima_sort_threshold": {"color": "#4ae747", "type": "line"},
            "maxima_sort_threshold": {"color": "#5b5e4b", "type": "line"},
        },
        "subplots": {
            "extrema_analysis": {
                "s_extrema": {"color": "#f53580", "type": "line"},
                "maxima": {"color": "#a29db9", "type": "scatter"},
                "minima": {"color": "#aac7fc", "type": "scatter"},
            },
            "murrey_math_levels": {
                "[4/8]P": {"color": "blue", "type": "line"},        # 50% MML
                "[6/8]P": {"color": "green", "type": "line"},       # 75% MML
                "[2/8]P": {"color": "orange", "type": "line"},      # 25% MML
                "[8/8]P": {"color": "red", "type": "line"},         # 100% MML
                "[0/8]P": {"color": "red", "type": "line"},         # 0% MML
                "mmlextreme_oscillator": {"color": "purple", "type": "line"},
            },
            "rsi_analysis": {
                "rsi": {"color": "purple", "type": "line"},
                "rsi_divergence_bull": {"color": "green", "type": "scatter"},
                "rsi_divergence_bear": {"color": "red", "type": "scatter"},
            },
            "confluence_analysis": {
                "confluence_score": {"color": "gold", "type": "line"},
                "near_support": {"color": "green", "type": "scatter"},
                "near_resistance": {"color": "red", "type": "scatter"},
                "near_mml": {"color": "blue", "type": "line"},
                "volume_spike": {"color": "orange", "type": "scatter"},
            },
            "volume_analysis": {
                "volume_strength": {"color": "cyan", "type": "line"},
                "volume_pressure": {"color": "magenta", "type": "line"},
                "buying_pressure": {"color": "green", "type": "line"},
                "selling_pressure": {"color": "red", "type": "line"},
                "money_flow_index": {"color": "yellow", "type": "line"},
            },
            "momentum_analysis": {
                "momentum_quality": {"color": "brown", "type": "line"},
                "momentum_acceleration": {"color": "pink", "type": "line"},
                "momentum_consistency": {"color": "lime", "type": "line"},
                "momentum_oscillator": {"color": "navy", "type": "line"},
            },
            "structure_analysis": {
                "structure_score": {"color": "teal", "type": "line"},
                "bullish_structure": {"color": "green", "type": "line"},
                "bearish_structure": {"color": "red", "type": "line"},
                "structure_break_up": {"color": "lime", "type": "scatter"},
                "structure_break_down": {"color": "crimson", "type": "scatter"},
            },
            "trend_strength": {
                "trend_strength": {"color": "indigo", "type": "line"},
                "trend_strength_5": {"color": "lightblue", "type": "line"},
                "trend_strength_10": {"color": "mediumblue", "type": "line"},
                "trend_strength_20": {"color": "darkblue", "type": "line"},
            },
            "ultimate_signals": {
                "ultimate_score": {"color": "gold", "type": "line"},
                "signal_strength": {"color": "silver", "type": "line"},
                "high_quality_setup": {"color": "lime", "type": "scatter"},
                "entry_type": {"color": "white", "type": "line"},
            },
            "market_conditions": {
                "strong_uptrend": {"color": "green", "type": "scatter"},
                "strong_downtrend": {"color": "red", "type": "scatter"},
                "ranging": {"color": "yellow", "type": "scatter"},
                "strong_up_momentum": {"color": "lime", "type": "scatter"},
                "strong_down_momentum": {"color": "crimson", "type": "scatter"},
            },
            "di_analysis": {
                "DI_values": {"color": "orange", "type": "line"},
                "DI_catch": {"color": "red", "type": "scatter"},
                "plus_di": {"color": "green", "type": "line"},
                "minus_di": {"color": "red", "type": "line"},
            }
        },
    }

    # Helper method to check if we have an active position in the opposite direction
    def has_active_trade(self, pair: str, side: str) -> bool:
        """
        Check if there's an active trade in the specified direction
        """
        try:
            trades = Trade.get_open_trades()
            for trade in trades:
                if trade.pair == pair:
                    if side == "long" and not trade.is_short:
                        return True
                    elif side == "short" and trade.is_short:
                        return True
        except Exception as e:
            logger.warning(f"Error checking active trades for {pair}: {e}")
        return False

    @staticmethod
    def _calculate_mml_core(mn: float, finalH: float, mx: float, finalL: float,
                            mml_c1: float, mml_c2: float) -> Dict[str, float]:
        dmml_calc = ((finalH - finalL) / 8.0) * mml_c1
        if dmml_calc == 0 or np.isinf(dmml_calc) or np.isnan(dmml_calc) or finalH == finalL:
            return {key: finalL for key in MML_LEVEL_NAMES}
        mml_val = (mx * mml_c2) + (dmml_calc * 3)
        if np.isinf(mml_val) or np.isnan(mml_val):
            return {key: finalL for key in MML_LEVEL_NAMES}
        ml = [mml_val - (dmml_calc * i) for i in range(16)]
        return {
            "[-3/8]P": ml[14], "[-2/8]P": ml[13], "[-1/8]P": ml[12],
            "[0/8]P": ml[11], "[1/8]P": ml[10], "[2/8]P": ml[9],
            "[3/8]P": ml[8], "[4/8]P": ml[7], "[5/8]P": ml[6],
            "[6/8]P": ml[5], "[7/8]P": ml[4], "[8/8]P": ml[3],
            "[+1/8]P": ml[2], "[+2/8]P": ml[1], "[+3/8]P": ml[0],
        }

    def calculate_rolling_murrey_math_levels_optimized(self, df: pd.DataFrame, window_size: int) -> Dict[str, pd.Series]:
        """
        OPTIMIZED Version - Calculate MML levels every 5 candles using only past data
        """
        murrey_levels_data: Dict[str, list] = {key: [np.nan] * len(df) for key in MML_LEVEL_NAMES}
        mml_c1 = self.mml_const1.value
        mml_c2 = self.mml_const2.value
        
        calculation_step = 5
        
        for i in range(0, len(df), calculation_step):
            if i < window_size:
                continue
                
            # Use data up to the previous candle for the rolling window
            window_end = i - 1
            window_start = window_end - window_size + 1
            if window_start < 0:
                window_start = 0
                
            window_data = df.iloc[window_start:window_end]
            mn_period = window_data["low"].min()
            mx_period = window_data["high"].max()
            current_close = df["close"].iloc[window_end] if window_end > 0 else df["close"].iloc[0]
            
            if pd.isna(mn_period) or pd.isna(mx_period) or mn_period == mx_period:
                for key in MML_LEVEL_NAMES:
                    murrey_levels_data[key][window_end] = current_close
                continue
                
            levels = self._calculate_mml_core(mn_period, mx_period, mx_period, mn_period, mml_c1, mml_c2)
            
            for key in MML_LEVEL_NAMES:
                murrey_levels_data[key][window_end] = levels.get(key, current_close)
        
        # Interpolate using only past data up to each point
        for key in MML_LEVEL_NAMES:
            series = pd.Series(murrey_levels_data[key], index=df.index)
            # Interpolate forward only up to the current point, avoiding future data
            series = series.expanding().mean().ffill()  # Use expanding mean as a safe alternative
            murrey_levels_data[key] = series.tolist()
        
        return {key: pd.Series(data, index=df.index) for key, data in murrey_levels_data.items()}

    def calculate_synthetic_market_breadth(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Calculate synthetic market breadth using technical indicators
        Simulates market sentiment based on multiple factors
        """
        try:
            # RSI component (30% weight)
            rsi_component = (dataframe['rsi'] - 50) / 50  # Normalize to -1 to 1
            
            # Volume component (25% weight)
            volume_ma = dataframe['volume'].rolling(20).mean()
            volume_component = (dataframe['volume'] / volume_ma - 1).clip(-1, 1)
            
            # Momentum component (25% weight)
            momentum_3 = dataframe['close'].pct_change(3)
            momentum_component = np.tanh(momentum_3 * 100)  # Smooth normalization
            
            # Volatility component (20% weight) - inverted (lower vol = higher breadth)
            atr_normalized = dataframe['atr'] / dataframe['close']
            atr_ma = atr_normalized.rolling(20).mean()
            volatility_component = -(atr_normalized / atr_ma - 1).clip(-1, 1)
            
            # Combine components with weights
            synthetic_breadth = (
                rsi_component * 0.30 +
                volume_component * 0.25 +
                momentum_component * 0.25 +
                volatility_component * 0.20
            )
            
            # Normalize to 0-1 range (market breadth percentage)
            synthetic_breadth = (synthetic_breadth + 1) / 2
            
            # Smooth with rolling average
            synthetic_breadth = synthetic_breadth.rolling(3).mean()
            
            return synthetic_breadth.fillna(0.5)
            
        except Exception as e:
            logger.warning(f"Synthetic market breadth calculation failed: {e}")
            return pd.Series(0.5, index=dataframe.index)

    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend strength to avoid entering against strong trends
        """
        # Linear regression slope
        def calc_slope(series, period=10):
            """Calculate linear regression slope"""
            if len(series) < period:
                return 0
            x = np.arange(period)
            y = series.iloc[-period:].values
            if np.isnan(y).any():
                return 0
            slope = np.polyfit(x, y, 1)[0]
            return slope
        
        # Calculate trend strength using multiple timeframes
        df['slope_5'] = df['close'].rolling(5).apply(lambda x: calc_slope(x, 5), raw=False)
        df['slope_10'] = df['close'].rolling(10).apply(lambda x: calc_slope(x, 10), raw=False)
        df['slope_20'] = df['close'].rolling(20).apply(lambda x: calc_slope(x, 20), raw=False)
        
        # Normalize slopes by price
        df['trend_strength_5'] = df['slope_5'] / df['close'] * 100
        df['trend_strength_10'] = df['slope_10'] / df['close'] * 100
        df['trend_strength_20'] = df['slope_20'] / df['close'] * 100
        
        # Combined trend strength
        df['trend_strength'] = (df['trend_strength_5'] + df['trend_strength_10'] + df['trend_strength_20']) / 3
        
        # Trend classification
        strong_up_threshold = self.trend_strength_threshold.value
        strong_down_threshold = -self.trend_strength_threshold.value
        
        df['strong_uptrend'] = df['trend_strength'] > strong_up_threshold
        df['strong_downtrend'] = df['trend_strength'] < strong_down_threshold
        df['ranging'] = (df['trend_strength'].abs() < strong_up_threshold * 0.5)
        
        return df
    @property
    def protections(self):
        prot = [{"method": "CooldownPeriod", "stop_duration_candles": self.cooldown_lookback.value}]
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 72,
                "trade_limit": 2,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False,
            })
        return prot

    def informative_pairs(self):
        """
        Define additional pairs for correlation analysis
        """
        pairs = []
        
        # Add BTC for correlation analysis (if not already trading BTC)
        if self.timeframe:
            pairs.append(("BTC/USDT:USDT", self.timeframe))
            
        # Add major market indicators
        pairs.extend([
            ("BTC/USDT:USDT", self.timeframe),
            ("ETH/USDT:USDT", self.timeframe),
            ("BNB/USDT:USDT", self.timeframe),
        ])
        
        return pairs


    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        
        if dataframe.empty or 'atr' not in dataframe.columns:
            return self.stoploss  # Use strategy stoploss (-0.15) as fallback
        
        atr = dataframe["atr"].iat[-1]
        if pd.isna(atr) or atr <= 0:
            return self.stoploss  # Fallback to -0.15
        
        atr_percent = atr / current_rate
        
        # Profit-based multiplier adjustment
        if current_profit > 0.15:
            multiplier = 1.0
        elif current_profit > 0.08:
            multiplier = 1.2
        elif current_profit > 0.03:
            multiplier = 1.4
        else:
            multiplier = 1.6
        
        calculated_stoploss = -(atr_percent * multiplier * self.atr_stoploss_multiplier.value)
        
        # Initialize trailing_offset
        trailing_offset = 0.0
        
        # Enhanced trailing logic with multiple profit levels
        if current_profit > 0.01:  # Start trailing at 1% profit instead of 3%
            if current_profit > self.trailing_stop_positive_offset:  # 0.03 (3% profit)
                # Full trailing at 3%+ profit
                trailing_offset = max(0, current_profit - self.trailing_stop_positive)  # Trail 0.5% below peak
            elif current_profit > 0.02:  # 2% profit
                # Moderate trailing at 2-3% profit
                trailing_offset = max(0, current_profit - 0.01)  # Trail 1% below peak
            else:  # 1-2% profit
                # Minimal trailing at 1-2% profit
                trailing_offset = max(0, current_profit - 0.015)  # Trail 1.5% below peak
            
            # Apply trailing adjustment to calculated stoploss
            if trailing_offset > 0:
                calculated_stoploss = min(calculated_stoploss, -trailing_offset)  # Trail up in profit
        
        final_stoploss = max(
            min(calculated_stoploss, self.atr_stoploss_ceiling.value),
            self.atr_stoploss_maximum.value
        )
        
        logger.info(f"{pair} Custom SL: {final_stoploss:.3f} | ATR: {atr:.6f} | "
                   f"Profit: {current_profit:.3f} | Trailing: {trailing_offset:.3f}")
        return final_stoploss


    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float,
                 max_leverage: float, side: str, **kwargs) -> float:
        window_size = self.leverage_window_size.value
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if len(dataframe) < window_size:
            logger.warning(
                f"{pair} Not enough data ({len(dataframe)} candles) to calculate dynamic leverage (requires {window_size}). Using proposed: {proposed_leverage}")
            return proposed_leverage
        close_prices_series = dataframe["close"].tail(window_size)
        high_prices_series = dataframe["high"].tail(window_size)
        low_prices_series = dataframe["low"].tail(window_size)
        base_leverage = self.leverage_base.value
        rsi_array = ta.RSI(close_prices_series, timeperiod=14)
        atr_array = ta.ATR(high_prices_series, low_prices_series, close_prices_series, timeperiod=14)
        sma_array = ta.SMA(close_prices_series, timeperiod=20)
        macd_output = ta.MACD(close_prices_series, fastperiod=12, slowperiod=26, signalperiod=9)

        current_rsi = rsi_array[-1] if rsi_array.size > 0 and not np.isnan(rsi_array[-1]) else 50.0
        current_atr = atr_array[-1] if atr_array.size > 0 and not np.isnan(atr_array[-1]) else 0.0
        current_sma = sma_array[-1] if sma_array.size > 0 and not np.isnan(sma_array[-1]) else current_rate
        current_macd_hist = 0.0

        if isinstance(macd_output, pd.DataFrame):
            if not macd_output.empty and 'macdhist' in macd_output.columns:
                valid_macdhist_series = macd_output['macdhist'].dropna()
                if not valid_macdhist_series.empty:
                    current_macd_hist = valid_macdhist_series.iloc[-1]

        # Apply rules based on indicators
        if side == "long":
            if current_rsi < self.leverage_rsi_low.value:
                base_leverage *= self.leverage_long_increase_factor.value
            elif current_rsi > self.leverage_rsi_high.value:
                base_leverage *= self.leverage_long_decrease_factor.value

            if current_atr > 0 and current_rate > 0:
                if (current_atr / current_rate) > self.leverage_atr_threshold_pct.value:
                    base_leverage *= self.leverage_volatility_decrease_factor.value

            if current_macd_hist > 0:
                base_leverage *= self.leverage_long_increase_factor.value

            if current_sma > 0 and current_rate < current_sma:
                base_leverage *= self.leverage_long_decrease_factor.value

        adjusted_leverage = round(max(1.0, min(base_leverage, max_leverage)), 2)
        logger.info(
            f"{pair} Dynamic Leverage: {adjusted_leverage:.2f} (Base: {base_leverage:.2f}, RSI: {current_rsi:.2f}, "
            f"ATR: {current_atr:.4f}, MACD Hist: {current_macd_hist:.4f}, SMA: {current_sma:.4f})")
        return adjusted_leverage

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        ULTIMATE indicator calculations with advanced market analysis
        """
        # === EXTERNAL DATA INTEGRATION ===
        try:
            # Add BTC data for correlation analysis using informative pairs
            if metadata['pair'] != 'BTC/USDT:USDT':
                btc_info = self.dp.get_pair_dataframe('BTC/USDT:USDT', self.timeframe)
                if not btc_info.empty and len(btc_info) >= len(dataframe):
                    # Take only the last N rows to match our dataframe length
                    btc_close_data = btc_info['close'].tail(len(dataframe)).reset_index(drop=True)
                    dataframe['btc_close'] = btc_close_data.values
                    logger.info(f"{metadata['pair']} BTC correlation data added successfully")
                else:
                    # Fallback: use current pair data
                    dataframe['btc_close'] = dataframe['close']
                    logger.warning(f"{metadata['pair']} BTC data unavailable, using pair data as fallback")
            else:
                # For BTC pairs, use own data
                dataframe['btc_close'] = dataframe['close']
        except Exception as e:
            logger.warning(f"{metadata['pair']} BTC data integration failed: {e}")
            dataframe['btc_close'] = dataframe['close']  # Safe fallback
        
        # === STANDARD INDICATORS ===
        dataframe["ema50"] = ta.EMA(dataframe["close"], timeperiod=50)
        dataframe["ema100"] = ta.EMA(dataframe["close"], timeperiod=100) # Neu hinzufÃ¼gen
        dataframe["rsi"] = ta.RSI(dataframe["close"])
        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=10)
        
        # === SYNTHETIC MARKET BREADTH CALCULATION ===
        try:
            # Calculate synthetic market breadth using multiple indicators
            # (after RSI and ATR are available)
            dataframe['market_breadth'] = self.calculate_synthetic_market_breadth(dataframe)
            logger.info(f"{metadata['pair']} Synthetic market breadth calculated")
        except Exception as e:
            logger.warning(f"{metadata['pair']} Market breadth calculation failed: {e}")
            dataframe['market_breadth'] = 0.5  # Neutral fallback
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)
        dataframe["DI_values"] = dataframe["plus_di"] - dataframe["minus_di"]
        dataframe["DI_cutoff"] = 0

        # === EXTREMA DETECTION ===
        extrema_order = self.indicator_extrema_order.value
        dataframe["maxima"] = (
            dataframe["close"] == dataframe["close"].shift(1).rolling(window=extrema_order).max()
        ).astype(int)
        dataframe["minima"] = (
            dataframe["close"] == dataframe["close"].shift(1).rolling(window=extrema_order).min()
        ).astype(int)

        dataframe["s_extrema"] = 0
        dataframe.loc[dataframe["minima"] == 1, "s_extrema"] = -1
        dataframe.loc[dataframe["maxima"] == 1, "s_extrema"] = 1

        # === HEIKIN-ASHI ===
        dataframe["ha_close"] = (dataframe["open"] + dataframe["high"] + dataframe["low"] + dataframe["close"]) / 4

        # === ROLLING EXTREMA ===
        dataframe["minh2"], dataframe["maxh2"] = calculate_minima_maxima(dataframe, self.h2.value)
        dataframe["minh1"], dataframe["maxh1"] = calculate_minima_maxima(dataframe, self.h1.value)
        dataframe["minh0"], dataframe["maxh0"] = calculate_minima_maxima(dataframe, self.h0.value)
        dataframe["mincp"], dataframe["maxcp"] = calculate_minima_maxima(dataframe, self.cp.value)

        # === MURREY MATH LEVELS ===
        mml_window = self.indicator_mml_window.value
        murrey_levels = self.calculate_rolling_murrey_math_levels_optimized(dataframe, window_size=mml_window)
        
        for level_name in MML_LEVEL_NAMES:
            if level_name in murrey_levels:
                dataframe[level_name] = murrey_levels[level_name]
            else:
                dataframe[level_name] = dataframe["close"]

        # === MML OSCILLATOR ===
        mml_4_8 = dataframe.get("[4/8]P")
        mml_plus_3_8 = dataframe.get("[+3/8]P")
        mml_minus_3_8 = dataframe.get("[-3/8]P")
        
        if mml_4_8 is not None and mml_plus_3_8 is not None and mml_minus_3_8 is not None:
            osc_denominator = (mml_plus_3_8 - mml_minus_3_8).replace(0, np.nan)
            dataframe["mmlextreme_oscillator"] = 100 * ((dataframe["close"] - mml_4_8) / osc_denominator)
        else:
            dataframe["mmlextreme_oscillator"] = np.nan

        # === DI CATCH ===
        dataframe["DI_catch"] = np.where(dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1)

        # === ROLLING THRESHOLDS ===
        rolling_window_threshold = self.indicator_rolling_window_threshold.value
        dataframe["minima_sort_threshold"] = dataframe["close"].rolling(
            window=rolling_window_threshold, min_periods=1
        ).min()
        dataframe["maxima_sort_threshold"] = dataframe["close"].rolling(
            window=rolling_window_threshold, min_periods=1
        ).max()

        # === EXTREMA CHECKS ===
        rolling_check_window = self.indicator_rolling_check_window.value
        dataframe["minima_check"] = (
            dataframe["minima"].rolling(window=rolling_check_window, min_periods=1).sum() == 0
        ).astype(int)
        dataframe["maxima_check"] = (
            dataframe["maxima"].rolling(window=rolling_check_window, min_periods=1).sum() == 0
        ).astype(int)

        # === VOLATILITY INDICATORS ===
        dataframe["volatility_range"] = dataframe["high"] - dataframe["low"]
        dataframe["avg_volatility"] = dataframe["volatility_range"].rolling(window=50).mean()
        dataframe["avg_volume"] = dataframe["volume"].rolling(window=50).mean()

        # === TREND STRENGTH INDICATORS ===
        # Use enhanced Wavelet+FFT method with fallback
        try:
            # Advanced wavelet & FFT method
            dataframe = calculate_advanced_trend_strength_with_wavelets(dataframe)
            
            # Use advanced trend strength as primary
            dataframe['trend_strength'] = dataframe['trend_strength_cycle_adjusted']
            dataframe['strong_uptrend'] = dataframe['strong_uptrend_advanced']
            dataframe['strong_downtrend'] = dataframe['strong_downtrend_advanced']
            dataframe['ranging'] = dataframe['ranging_advanced']
            
            logger.info(f"{metadata['pair']} Using advanced Wavelet+FFT trend analysis")
            
        except Exception as e:
            # Fallback to original enhanced method if advanced fails
            logger.warning(
                f"{metadata['pair']} Wavelet/FFT analysis failed: {e}. "
                "Using enhanced method."
            )
            
            def calc_slope(series, period):
                """Enhanced slope calculation as fallback"""
                if len(series) < period:
                    return 0
                y = series.values[-period:]
                if np.isnan(y).any() or np.isinf(y).any():
                    return 0
                if np.all(y == y[0]):
                    return 0
                x = np.linspace(0, period-1, period)
                try:
                    coefficients = np.polyfit(x, y, 1)
                    slope = coefficients[0]
                    if np.isnan(slope) or np.isinf(slope):
                        return 0
                    max_reasonable_slope = np.std(y) / period
                    if abs(slope) > max_reasonable_slope * 10:
                        return np.sign(slope) * max_reasonable_slope * 10
                    return slope
                except Exception:
                    try:
                        simple_slope = (y[-1] - y[0]) / (period - 1)
                        return (simple_slope if not
                               (np.isnan(simple_slope) or np.isinf(simple_slope))
                               else 0)
                    except Exception:
                        return 0
            
            # Original slope calculations
            dataframe['slope_5'] = dataframe['close'].rolling(5).apply(
                lambda x: calc_slope(x, 5), raw=False
            )
            dataframe['slope_10'] = dataframe['close'].rolling(10).apply(
                lambda x: calc_slope(x, 10), raw=False
            )
            dataframe['slope_20'] = dataframe['close'].rolling(20).apply(
                lambda x: calc_slope(x, 20), raw=False
            )
            
            dataframe['trend_strength_5'] = dataframe['slope_5'] / dataframe['close'] * 100
            dataframe['trend_strength_10'] = dataframe['slope_10'] / dataframe['close'] * 100
            dataframe['trend_strength_20'] = dataframe['slope_20'] / dataframe['close'] * 100
            
            dataframe['trend_strength'] = (
                dataframe['trend_strength_5'] +
                dataframe['trend_strength_10'] +
                dataframe['trend_strength_20']
            ) / 3
            
            strong_threshold = 0.02
            dataframe['strong_uptrend'] = dataframe['trend_strength'] > strong_threshold
            dataframe['strong_downtrend'] = dataframe['trend_strength'] < -strong_threshold
            dataframe['ranging'] = dataframe['trend_strength'].abs() < (strong_threshold * 0.5)

        # === MOMENTUM INDICATORS ===
        dataframe['price_momentum'] = dataframe['close'].pct_change(3)
        dataframe['momentum_increasing'] = (
            dataframe['price_momentum'] > dataframe['price_momentum'].shift(1)
        )
        dataframe['momentum_decreasing'] = (
            dataframe['price_momentum'] < dataframe['price_momentum'].shift(1)
        )

        dataframe['volume_momentum'] = (
            dataframe['volume'].rolling(3).mean() /
            dataframe['volume'].rolling(20).mean()
        )

        dataframe['rsi_divergence_bull'] = (
            (dataframe['close'] < dataframe['close'].shift(5)) &
            (dataframe['rsi'] > dataframe['rsi'].shift(5))
        )
        dataframe['rsi_divergence_bear'] = (
            (dataframe['close'] > dataframe['close'].shift(5)) &
            (dataframe['rsi'] < dataframe['rsi'].shift(5))
        )

        # === CANDLE PATTERNS ===
        dataframe['green_candle'] = dataframe['close'] > dataframe['open']
        dataframe['red_candle'] = dataframe['close'] < dataframe['open']
        dataframe['consecutive_green'] = dataframe['green_candle'].rolling(3).sum()
        dataframe['consecutive_red'] = dataframe['red_candle'].rolling(3).sum()

        # Define strong_threshold for momentum calculations
        strong_threshold = 0.02

        dataframe['strong_up_momentum'] = (
            (dataframe['consecutive_green'] >= 3) &
            (dataframe['volume'] > dataframe['avg_volume']) &
            (dataframe['trend_strength'] > strong_threshold)
        )
        dataframe['strong_down_momentum'] = (
            (dataframe['consecutive_red'] >= 3) &
            (dataframe['volume'] > dataframe['avg_volume']) &
            (dataframe['trend_strength'] < -strong_threshold)
        )

        # === ADVANCED ANALYSIS MODULES ===
        
        # 1. CONFLUENCE ANALYSIS
        if self.confluence_enabled.value:
            dataframe = calculate_confluence_score(dataframe)
        else:
            dataframe['confluence_score'] = 0
        
        # 2. SMART VOLUME ANALYSIS
        if self.volume_analysis_enabled.value:
            dataframe = calculate_smart_volume(dataframe)
        else:
            dataframe['volume_pressure'] = 0
            dataframe['volume_strength'] = 1.0
            dataframe['money_flow_index'] = 50
        
        # 3. ADVANCED MOMENTUM
        if self.momentum_analysis_enabled.value:
            dataframe = calculate_advanced_momentum(dataframe)
        else:
            dataframe['momentum_quality'] = 0
            dataframe['momentum_acceleration'] = 0
        
        # 4. MARKET STRUCTURE
        if self.structure_analysis_enabled.value:
            dataframe = calculate_market_structure(dataframe)
        else:
            dataframe['structure_score'] = 0
            dataframe['structure_break_up'] = 0
        
        # 5. ADVANCED ENTRY SIGNALS
        dataframe = calculate_advanced_entry_signals(dataframe)

        # === ULTIMATE MARKET SCORE ===
        dataframe['ultimate_score'] = (
            dataframe['confluence_score'] * 0.25 +           # 25% confluence
            dataframe['volume_pressure'] * 0.2 +             # 20% volume pressure
            dataframe['momentum_quality'] * 0.2 +            # 20% momentum quality
            (dataframe['structure_score'] / 5) * 0.15 +      # 15% structure (normalized)
            (dataframe['signal_strength'] / 10) * 0.2        # 20% signal strength
        )

        if self.regime_change_enabled.value:
            
            # ===========================================
            # FLASH MOVE DETECTION
            # ===========================================
            
            flash_candles = self.flash_move_candles.value
            flash_threshold = self.flash_move_threshold.value
            
            # Schnelle Preisbewegungen
            dataframe['price_change_fast'] = dataframe['close'].pct_change(flash_candles)
            dataframe['flash_pump'] = dataframe['price_change_fast'] > flash_threshold
            dataframe['flash_dump'] = dataframe['price_change_fast'] < -flash_threshold
            dataframe['flash_move'] = dataframe['flash_pump'] | dataframe['flash_dump']
            
            # ===========================================
            # VOLUME SPIKE DETECTION
            # ===========================================
            
            volume_ma20 = dataframe['volume'].rolling(20).mean()
            volume_multiplier = self.volume_spike_multiplier.value
            dataframe['volume_spike'] = dataframe['volume'] > (volume_ma20 * volume_multiplier)
            
            # Volume + Bewegung kombiniert
            dataframe['volume_pump'] = dataframe['volume_spike'] & dataframe['flash_pump']
            dataframe['volume_dump'] = dataframe['volume_spike'] & dataframe['flash_dump']
            
            # ===========================================
            # MARKET SENTIMENT DETECTION
            # ===========================================
            
            # Market Breadth Change (falls vorhanden)
            if 'market_breadth' in dataframe.columns:
                dataframe['market_breadth_change'] = dataframe['market_breadth'].diff(3)
                sentiment_threshold = self.sentiment_shift_threshold.value
                dataframe['sentiment_shift_bull'] = dataframe['market_breadth_change'] > sentiment_threshold
                dataframe['sentiment_shift_bear'] = dataframe['market_breadth_change'] < -sentiment_threshold
            else:
                dataframe['sentiment_shift_bull'] = False
                dataframe['sentiment_shift_bear'] = False
            
            # ===========================================
            # BTC CORRELATION MONITORING
            # ===========================================
            
            # BTC Flash Moves (falls BTC Daten vorhanden)
            if 'btc_close' in dataframe.columns:
                dataframe['btc_change_fast'] = dataframe['btc_close'].pct_change(flash_candles)
                dataframe['btc_flash_pump'] = dataframe['btc_change_fast'] > flash_threshold
                dataframe['btc_flash_dump'] = dataframe['btc_change_fast'] < -flash_threshold
                
                # Correlation Break: BTC bewegt sich stark, Coin nicht
                pair_movement = dataframe['price_change_fast'].abs()
                btc_movement = dataframe['btc_change_fast'].abs()
                dataframe['correlation_break'] = (btc_movement > flash_threshold) & (pair_movement < flash_threshold * 0.4)
            else:
                dataframe['btc_flash_pump'] = False
                dataframe['btc_flash_dump'] = False
                dataframe['correlation_break'] = False
            
            # ===========================================
            # REGIME CHANGE SCORE
            # ===========================================
            
            # Kombiniere alle Signale
            regime_signals = [
                'flash_move', 'volume_spike', 
                'sentiment_shift_bull', 'sentiment_shift_bear',
                'btc_flash_pump', 'btc_flash_dump', 'correlation_break'
            ]
            
            dataframe['regime_change_score'] = 0
            for signal in regime_signals:
                if signal in dataframe.columns:
                    dataframe['regime_change_score'] += dataframe[signal].astype(int)
            
            # Normalisiere auf 0-1
            max_signals = len(regime_signals)
            dataframe['regime_change_intensity'] = dataframe['regime_change_score'] / max_signals
            
            # Alert Level
            sensitivity = self.regime_change_sensitivity.value
            dataframe['regime_alert'] = dataframe['regime_change_intensity'] >= sensitivity
            
        else:
            # Falls Regime Detection deaktiviert
            dataframe['flash_pump'] = False
            dataframe['flash_dump'] = False
            dataframe['volume_pump'] = False
            dataframe['volume_dump'] = False
            dataframe['regime_alert'] = False
            dataframe['regime_change_intensity'] = 0.0
        
        # === ADVANCED PREDICTIVE ANALYSIS ===
        try:
            pair = metadata.get('pair', 'UNKNOWN')
            dataframe = calculate_advanced_predictive_signals(dataframe, pair)
            dataframe = calculate_quantum_momentum_analysis(dataframe)
            dataframe = calculate_neural_pattern_recognition(dataframe)
            
            logger.info(f"{pair} Advanced predictive analysis completed")
        except Exception as e:
            logger.warning(f"Advanced predictive analysis failed: {e}")
            dataframe['ml_entry_probability'] = 0.5
            dataframe['ml_enhanced_score'] = dataframe.get('ultimate_score', 0.5)
            dataframe['ml_high_confidence'] = 0
            dataframe['quantum_momentum_coherence'] = 0.5
            dataframe['momentum_entanglement'] = 0.5
            dataframe['quantum_tunnel_up_prob'] = 0.5
            dataframe['neural_pattern_score'] = 0.5
        
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        ULTIMATE ENTRY LOGIC - Multi-factor confluence system
        """

        # ===========================================
        # INITIALIZE ENTRY COLUMNS
        # ===========================================
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""
        
        # === CORE CONDITIONS (must all be true) ===
        core_conditions = (
            # Basic trend and momentum
            (dataframe['close'] > dataframe['ema50']) &
            (dataframe['rsi'] > 25) & (dataframe['rsi'] < 75) &
            (dataframe['trend_strength'] > -0.02) &  # Not in strong downtrend
            
            # Volume confirmation
            (dataframe['volume'] > dataframe['avg_volume'] * 0.7) &  # Minimum volume
            
            # No recent distribution
            (dataframe['selling_pressure'] <= 4)
        )
        
        # === ADVANCED CONDITIONS (weighted scoring) ===
        
        # 1. CONFLUENCE CONDITIONS
        confluence_conditions = (
            (dataframe['confluence_score'] >= self.confluence_threshold.value) |
            (
                (dataframe['confluence_score'] >= (self.confluence_threshold.value - 1)) &
                (dataframe['near_support'] == 1) &
                (dataframe['volume_spike'] == 1)
            )
        )
        
        # 2. VOLUME CONDITIONS
        volume_conditions = (
            (dataframe['volume_pressure'] >= self.volume_pressure_threshold.value) &
            (dataframe['volume_strength'] > self.volume_strength_threshold.value) &
            (dataframe['money_flow_index'] > 45)
        ) if self.require_volume_confirmation.value else True
        
        # 3. MOMENTUM CONDITIONS
        momentum_conditions = (
            (dataframe['momentum_quality'] >= self.momentum_quality_threshold.value) &
            (dataframe['momentum_acceleration'] > -0.01) &
            (dataframe['momentum_consistency'] >= 2)
        ) if self.require_momentum_confirmation.value else True
        
        # 4. STRUCTURE CONDITIONS
        structure_conditions = (
            (dataframe['structure_score'] >= self.structure_score_threshold.value) &
            (dataframe['bullish_structure'] > dataframe['bearish_structure'])
        ) if self.require_structure_confirmation.value else True
        
        # === ORIGINAL EXTREMA CONDITIONS ===
        extrema_conditions = (
            # Minima conditions
            (
                (dataframe["minima"] == 1) &
                (dataframe["minima_check"] == 1) &
                (dataframe["close"] <= dataframe["minima_sort_threshold"] * 1.02) &
                (dataframe["DI_catch"] == 1)
            ) |
            # Alternative: MML level conditions
            (
                (dataframe["close"] <= dataframe["[0/8]P"] * 1.01) |
                (dataframe["close"] <= dataframe["[2/8]P"] * 1.01) |
                (dataframe["close"] <= dataframe["[4/8]P"] * 1.01)
            ) |
            # Rolling extrema conditions
            (
                (dataframe["close"] <= dataframe["minh2"] * 1.015) |
                (dataframe["close"] <= dataframe["minh1"] * 1.015) |
                (dataframe["close"] <= dataframe["minh0"] * 1.015)
            )
        )
        
        # === QUALITY FILTERS ===
        quality_filters = (
            # Ultimate score threshold
            (dataframe['ultimate_score'] > self.ultimate_score_threshold.value) &
            
            # Signal strength
            (dataframe['signal_strength'] >= 5) &
            
            # No extreme RSI
            (dataframe['rsi'] < 80) &
            
            # Positive momentum
            (dataframe['price_momentum'] > -0.02) &
            
            # Volume not declining
            (dataframe['volume_trend'] == 1) |
            (dataframe['volume_strength'] > 1.1)
        )
        
        # === RISK MANAGEMENT CONDITIONS ===
        risk_conditions = (
            # No consecutive red candles
            (dataframe['consecutive_red'] <= 2) &
            
            # ATR not too high (volatility control)
            (dataframe['atr'] < dataframe['close'] * 0.05) &  # Max 5% ATR
            
            # No strong bearish momentum
            (dataframe['strong_down_momentum'] == 0) &
            
            # RSI divergence protection
            (dataframe['rsi_divergence_bear'] == 0)
        )
        
        # === MARKET REGIME ADAPTATIONS ===
        
        # In ranging markets, require stronger confluence
        ranging_market_adjustment = (
            (~dataframe['ranging']) |  # Not ranging, OR
            (dataframe['confluence_score'] >= (self.confluence_threshold.value + 1))  # Higher confluence if ranging
        )
        
        # In strong trends, allow more aggressive entries
        trend_market_adjustment = (
            (~dataframe['strong_uptrend']) |  # Not in strong uptrend, OR
            (dataframe['volume_pressure'] >= 1)  # Lower volume requirement in strong trends
        )
        
        # === FINAL ENTRY CONDITION COMBINATIONS ===
        
        # HIGH QUALITY ENTRIES (all advanced filters)
        high_quality_entry = (
            core_conditions &
            confluence_conditions &
            volume_conditions &
            momentum_conditions &
            structure_conditions &
            extrema_conditions &
            quality_filters &
            risk_conditions &
            ranging_market_adjustment &
            trend_market_adjustment
        )
        
        # MEDIUM QUALITY ENTRIES (relaxed requirements)
        medium_quality_entry = (
            core_conditions &
            extrema_conditions &
            (
                # Either strong confluence OR strong momentum OR strong volume
                confluence_conditions |
                (dataframe['momentum_quality'] >= 4) |
                (dataframe['volume_pressure'] >= 3)
            ) &
            # Basic quality filters
            (dataframe['ultimate_score'] > (self.ultimate_score_threshold.value * 0.7)) &
            (dataframe['signal_strength'] >= 3) &
            risk_conditions
        )
        
        # BACKUP ENTRIES (original logic only)
        backup_entry = (
            core_conditions &
            extrema_conditions &
            (dataframe['volume'] > dataframe['avg_volume']) &
            (dataframe['trend_strength'] > 0)
        )
        
        # === ADDITIONAL ENTRY SIGNALS FOR SPECIFIC SCENARIOS ===
        
        # Momentum breakout entries
        momentum_breakout = (
            (dataframe['structure_break_up'] == 1) &
            (dataframe['volume_strength'] > 1.5) &
            (dataframe['momentum_quality'] >= 4) &
            (dataframe['close'] > dataframe['ema50']) &
            (dataframe['rsi'] < 75) &
            risk_conditions
        )
        
        # Volume spike reversals
        volume_reversal = (
            (dataframe['volume_spike'] == 1) &
            (dataframe['rsi_oversold'] == 1) &
            (dataframe['near_support'] == 1) &
            (dataframe['buying_pressure'] >= 2) &
            (dataframe['close'] > dataframe['open']) &  # Green candle
            risk_conditions
        )
        
        # === SHORT LOGIC ===
        
        # CORE SHORT CONDITIONS (mirror of long)
        core_short_conditions = (
            (dataframe['close'] < dataframe['ema50']) &         # Below EMA
            (dataframe['rsi'] > 30) & (dataframe['rsi'] < 70) & # Not extreme
            (dataframe['trend_strength'] < 0.01) &              # Downtrend or neutral
            (dataframe['volume'] > dataframe['avg_volume'] * 0.8) &
            (dataframe['buying_pressure'] <= 3)                 # Not too much buying
        )
        
        # SHORT CONFLUENCE (inverted)
        short_confluence_conditions = (
            (dataframe['confluence_score'] >= self.confluence_threshold.value) |
            (
                (dataframe['confluence_score'] >= (self.confluence_threshold.value - 1)) &
                (dataframe['near_resistance'] == 1) &           # Near resistance for SHORT
                (dataframe['volume_spike'] == 1)
            )
        )
        
        # SHORT VOLUME CONDITIONS
        short_volume_conditions = (
            (dataframe['volume_pressure'] <= -self.volume_pressure_threshold.value) &  # Negative pressure
            (dataframe['volume_strength'] > self.volume_strength_threshold.value) &
            (dataframe['money_flow_index'] < 55)                # Money flowing out
        ) if self.require_volume_confirmation.value else True
        
        # SHORT MOMENTUM CONDITIONS  
        short_momentum_conditions = (
            (dataframe['momentum_quality'] <= -2) &             # Negative momentum quality
            (dataframe['momentum_acceleration'] < 0.01) &       # Decelerating
            (dataframe['momentum_consistency'] <= 1)            # Inconsistent upward momentum
        ) if self.require_momentum_confirmation.value else True
        
        # SHORT STRUCTURE CONDITIONS
        short_structure_conditions = (
            (dataframe['structure_score'] <= -self.structure_score_threshold.value) &  # Negative structure
            (dataframe['bearish_structure'] > dataframe['bullish_structure'])          # More bearish signals
        ) if self.require_structure_confirmation.value else True
        
        # SHORT EXTREMA CONDITIONS (inverted)
        short_extrema_conditions = (
            # Maxima conditions (resistance for shorts)
            (
                (dataframe["maxima"] == 1) &
                (dataframe["maxima_check"] == 1) &
                (dataframe["close"] >= dataframe["maxima_sort_threshold"] * 0.98) &
                (dataframe["DI_catch"] == 1)
            ) |
            # MML resistance levels
            (
                (dataframe["close"] >= dataframe["[6/8]P"] * 0.99) |
                (dataframe["close"] >= dataframe["[8/8]P"] * 0.99) |
                (dataframe["close"] >= dataframe["[+1/8]P"] * 0.99)
            ) |
            # Rolling maxima
            (
                (dataframe["close"] >= dataframe["maxh2"] * 0.985) |
                (dataframe["close"] >= dataframe["maxh1"] * 0.985) |
                (dataframe["close"] >= dataframe["maxh0"] * 0.985)
            )
        )
        
        # SHORT QUALITY FILTERS
        short_quality_filters = (
            (dataframe['ultimate_score'] < (1 - self.ultimate_score_threshold.value)) &  # Low score for shorts
            (dataframe['signal_strength'] <= -3) &              # Negative signal strength
            (dataframe['rsi'] > 20) &                           # Not oversold
            (dataframe['price_momentum'] < 0.02) &              # Negative momentum
            (dataframe['volume_trend'] == 0) |                  # Volume declining
            (dataframe['volume_strength'] < 0.9)                # Weak volume
        )
        
        # SHORT RISK CONDITIONS
        short_risk_conditions = (
            (dataframe['consecutive_green'] <= 2) &             # Not too many green candles
            (dataframe['atr'] < dataframe['close'] * 0.05) &    # Volatility control
            (dataframe['strong_up_momentum'] == 0) &            # No strong bullish momentum
            (dataframe['rsi_divergence_bull'] == 0)             # No bullish divergence
        )
        
        # === SHORT ENTRY COMBINATIONS ===
        
        # HIGH QUALITY SHORT ENTRIES
        high_quality_short = (
            core_short_conditions &
            short_confluence_conditions &
            short_volume_conditions &
            short_momentum_conditions &
            short_structure_conditions &
            short_extrema_conditions &
            short_quality_filters &
            short_risk_conditions
        )
        
        # MEDIUM QUALITY SHORT ENTRIES
        medium_quality_short = (
            core_short_conditions &
            short_extrema_conditions &
            (
                short_confluence_conditions |
                (dataframe['momentum_quality'] <= -3) |
                (dataframe['volume_pressure'] <= -3)
            ) &
            short_risk_conditions &
            ~(
                (dataframe['close'] > dataframe['ema50']) |  # Exclude if above EMA50 (long condition)
                (dataframe['trend_strength'] > 0.01) |       # Exclude if in uptrend
                (dataframe['rsi'] < 30) |                   # Exclude if oversold (long bias)
                (dataframe['ultimate_score'] > self.ultimate_score_threshold.value)  # Exclude high-quality long setups
            )
        )
        
        # === ENTRY PRIORITY SYSTEM ===
        # Initialize entry type column
        dataframe['entry_type'] = 0
        
        # === APPLY LONG ENTRIES ===
        # HIGH QUALITY LONG
        dataframe.loc[high_quality_entry, "enter_long"] = 1
        dataframe.loc[high_quality_entry, 'entry_type'] = 3
        dataframe.loc[high_quality_entry, "enter_tag"] = "high_quality_long"
        
        # MEDIUM QUALITY LONG  
        dataframe.loc[medium_quality_entry & ~high_quality_entry, "enter_long"] = 1
        dataframe.loc[medium_quality_entry & ~high_quality_entry, 'entry_type'] = 2
        dataframe.loc[medium_quality_entry & ~high_quality_entry, "enter_tag"] = "medium_quality_long"
        
        # BACKUP LONG
        dataframe.loc[backup_entry & ~(high_quality_entry | medium_quality_entry), "enter_long"] = 1
        dataframe.loc[backup_entry & ~(high_quality_entry | medium_quality_entry), 'entry_type'] = 1  
        dataframe.loc[backup_entry & ~(high_quality_entry | medium_quality_entry), "enter_tag"] = "backup_long"
        
        # BREAKOUT LONG
        dataframe.loc[momentum_breakout, "enter_long"] = 1
        dataframe.loc[momentum_breakout, 'entry_type'] = 4
        dataframe.loc[momentum_breakout, "enter_tag"] = "breakout_long"
        
        # REVERSAL LONG
        dataframe.loc[volume_reversal, "enter_long"] = 1  
        dataframe.loc[volume_reversal, 'entry_type'] = 5
        dataframe.loc[volume_reversal, "enter_tag"] = "reversal_long"
        
        # === APPLY SHORT ENTRIES ===
        dataframe.loc[high_quality_short, "enter_short"] = 1
        dataframe.loc[high_quality_short, 'entry_type'] = 6     # High quality short
        dataframe.loc[high_quality_short, "enter_tag"] = "high_quality_short"
        
        dataframe.loc[
            medium_quality_short & ~high_quality_short, 
            "enter_short"
        ] = 1
        dataframe.loc[
            medium_quality_short & ~high_quality_short, 
            'entry_type'
        ] = 7  # Medium quality short
        dataframe.loc[
            medium_quality_short & ~high_quality_short, 
            "enter_tag"
        ] = "medium_quality_short"
        
        # === ENHANCED ML ENTRY SIGNALS WITH DYNAMIC POSITION SIZING ===
        
        # Helper function to safely check column existence and get conditions
        def safe_column_condition(df, col_name, operator, threshold, default_condition=True):
            """Safely create condition from dataframe column"""
            if col_name in df.columns and df[col_name] is not None:
                col_data = df[col_name]
                if operator == '>':
                    return col_data > threshold
                elif operator == '<':
                    return col_data < threshold
                elif operator == '>=':
                    return col_data >= threshold
                elif operator == '<=':
                    return col_data <= threshold
                elif operator == '==':
                    return col_data == threshold
            # Return default condition if column doesn't exist
            if isinstance(default_condition, bool):
                return pd.Series([default_condition] * len(df), index=df.index)
            else:
                return default_condition
        
        # ML Ultra Confidence Entries (Highest precision - 95%+ threshold)
        ml_ultra_long = (
            safe_column_condition(dataframe, 'ml_incremental_prediction', '>', 0.95, False) &
            (dataframe['ml_ultra_confidence'] == 1) &
            (dataframe['ml_entry_probability'] > 0.9) &
            safe_column_condition(dataframe, 'ml_model_agreement', '>', 0.85, True) &
            safe_column_condition(dataframe, 'quantum_momentum_coherence', '>', 0.8, True) &
            safe_column_condition(dataframe, 'momentum_entanglement', '>', 0.75, True) &
            safe_column_condition(dataframe, 'quantum_tunnel_up_prob', '>', 0.85, True) &
            safe_column_condition(dataframe, 'neural_pattern_score', '>', 0.9, True) &
            (dataframe['ml_enhanced_score'] > 0.85) &
            safe_column_condition(dataframe, 'pattern_prediction_confidence', '>', 0.8, True) &
            core_conditions &
            risk_conditions
        )
        
        # ML Enhanced High Confidence Entries (85%+ threshold)
        ml_enhanced_long = (
            safe_column_condition(dataframe, 'ml_incremental_prediction', '>', 0.85, False) &
            (dataframe['ml_high_confidence'] == 1) &
            (dataframe['ml_entry_probability'] > 0.8) &
            safe_column_condition(dataframe, 'ml_model_agreement', '>', 0.75, True) &
            safe_column_condition(dataframe, 'quantum_momentum_coherence', '>', 0.7, True) &
            safe_column_condition(dataframe, 'neural_pattern_score', '>', 0.8, True) &
            (dataframe['ml_enhanced_score'] > 0.8) &
            (dataframe['ultimate_score'] > self.ultimate_score_threshold.value) &
            core_conditions &
            risk_conditions &
            ~ml_ultra_long  # Not already covered by ultra precision
        )
        
        # ML Consensus Entries (Multiple models + incremental agree)
        ml_consensus_long = (
            safe_column_condition(dataframe, 'ml_incremental_prediction', '>', 0.75, False) &
            (dataframe['ml_entry_probability'] > 0.75) &
            safe_column_condition(dataframe, 'ml_model_agreement', '>', 0.8, True) &
            safe_column_condition(dataframe, 'quantum_momentum_coherence', '>', 0.65, True) &
            safe_column_condition(dataframe, 'neural_pattern_score', '>', 0.75, True) &
            (dataframe['volume_strength'] > 1.2) &
            (dataframe['trend_strength'] > 0.01) &
            core_conditions &
            risk_conditions &
            ~(ml_ultra_long | ml_enhanced_long)  # Not already covered
        )
        
        # ML Incremental Breakout (Focus on incremental learning signals)
        ml_incremental_breakout = (
            safe_column_condition(dataframe, 'ml_incremental_prediction', '>', 0.8, False) &
            safe_column_condition(dataframe, 'suggested_position_size', '>', 0.03, False) &  # High confidence size
            (dataframe['structure_break_up'] == 1) &
            safe_column_condition(dataframe, 'quantum_tunnel_up_prob', '>', 0.75, True) &
            safe_column_condition(dataframe, 'momentum_entanglement', '>', 0.7, True) &
            (dataframe['volume_strength'] > 1.4) &
            (dataframe['close'] > dataframe['ema50']) &
            (dataframe['trend_strength'] > 0.015) &
            risk_conditions &
            ~(ml_ultra_long | ml_enhanced_long | ml_consensus_long)
        )
        
        # ML Dynamic Position Entries (Adaptive to market conditions)
        ml_dynamic_position = (
            safe_column_condition(dataframe, 'ml_incremental_prediction', '>', 0.7, False) &
            safe_column_condition(dataframe, 'suggested_position_size', '>', 0.025, False) &
            (dataframe['ml_entry_probability'] > 0.7) &
            safe_column_condition(dataframe, 'ml_model_agreement', '>', 0.7, True) &
            (dataframe['ultimate_score'] > self.ultimate_score_threshold.value * 0.8) &
            (dataframe['volume_strength'] > 1.1) &
            extrema_conditions &  # Still require technical setup
            core_conditions &
            risk_conditions &
            ~(ml_ultra_long | ml_enhanced_long | ml_consensus_long | ml_incremental_breakout)
        )
        
        # Apply Enhanced ML Entry Signals with Position Sizing
        dataframe.loc[ml_ultra_long, "enter_long"] = 1
        dataframe.loc[ml_ultra_long, 'entry_type'] = 13
        dataframe.loc[ml_ultra_long, "enter_tag"] = "ml_ultra_95_dynamic"
        
        dataframe.loc[ml_enhanced_long, "enter_long"] = 1
        dataframe.loc[ml_enhanced_long, 'entry_type'] = 14
        dataframe.loc[ml_enhanced_long, "enter_tag"] = "ml_enhanced_85_dynamic"
        
        dataframe.loc[ml_consensus_long, "enter_long"] = 1
        dataframe.loc[ml_consensus_long, 'entry_type'] = 15
        dataframe.loc[ml_consensus_long, "enter_tag"] = "ml_consensus_dynamic"
        
        dataframe.loc[ml_incremental_breakout, "enter_long"] = 1
        dataframe.loc[ml_incremental_breakout, 'entry_type'] = 16
        dataframe.loc[ml_incremental_breakout, "enter_tag"] = "ml_incremental_breakout"
        
        dataframe.loc[ml_dynamic_position, "enter_long"] = 1
        dataframe.loc[ml_dynamic_position, 'entry_type'] = 17
        dataframe.loc[ml_dynamic_position, "enter_tag"] = "ml_dynamic_position"

        # === ENTRY DEBUGGING WITH ENHANCED METRICS ===
        # Log entry signals for major pairs
        if metadata['pair'] in ['BTC/USDT:USDT', 'ETH/USDT:USDT']:
            recent_entries = (dataframe['enter_long'].tail(5).sum() + 
                            dataframe['enter_short'].tail(5).sum())
            if recent_entries > 0:
                entry_type = dataframe['entry_type'].iloc[-1]
                entry_types = {
                    0: 'None', 1: 'Backup', 2: 'Medium', 3: 'High', 4: 'Breakout', 5: 'Reversal',
                    6: 'High Short', 7: 'Medium Short', 8: 'ML Ultra 90%', 9: 'ML Enhanced 80%',
                    10: 'ML Consensus', 11: 'ML Quantum', 12: 'ML Pattern',
                    13: 'ML Ultra 95% Dynamic', 14: 'ML Enhanced 85% Dynamic', 
                    15: 'ML Consensus Dynamic', 16: 'ML Incremental Breakout', 17: 'ML Dynamic Position'
                }
                logger.info(f"{metadata['pair']} 🎯 ENHANCED ENTRY SIGNAL - "
                           f"Type: {entry_types.get(entry_type, 'Unknown')}")
                logger.info(f"  📊 Ultimate Score: {dataframe['ultimate_score'].iloc[-1]:.3f}")
                logger.info(f"  🤖 ML Probability: {dataframe['ml_entry_probability'].iloc[-1]:.3f}")
                
                # Enhanced incremental ML metrics
                if 'ml_incremental_prediction' in dataframe.columns:
                    logger.info(f"  🔄 Incremental ML: {dataframe['ml_incremental_prediction'].iloc[-1]:.3f}")
                if 'suggested_position_size' in dataframe.columns:
                    logger.info(f"  💰 Dynamic Size: {dataframe['suggested_position_size'].iloc[-1]:.4f}")
                
                logger.info(f"  🔮 Quantum Coherence: "
                           f"{dataframe['quantum_momentum_coherence'].iloc[-1]:.3f}")
                logger.info(f"  🧠 Neural Pattern: {dataframe['neural_pattern_score'].iloc[-1]:.3f}")
                if 'ml_model_agreement' in dataframe.columns:
                    logger.info(f"  🤝 Model Agreement: "
                               f"{dataframe['ml_model_agreement'].iloc[-1]:.3f}")
        
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        UNIFIED EXIT SYSTEM - Choose between Custom MML Exits or Simple Opposite Signal Exits
        """
        # ===========================================
        # INITIALIZE EXIT COLUMNS
        # ===========================================
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""
        
        # ===========================================
        # CHOOSE EXIT SYSTEM
        # ===========================================
        if self.use_custom_exits_advanced:
            # Use Alex's Advanced MML-based Exit System
            return self._populate_custom_exits_advanced(dataframe, metadata)
        else:
            # Use Simple Opposite Signal Exit System
            return self._populate_simple_exits(dataframe, metadata)
    
    def _populate_custom_exits_advanced(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        ALEX'S ADVANCED MML-BASED EXIT SYSTEM
        Profit-protecting exit strategy with better signal coordination
        """
        
        # ===========================================
        # MML MARKET STRUCTURE FOR EXITS
        # ===========================================
        
        # Bullish/Bearish structure (same as entry)
        bullish_mml = (
            (df["close"] > df["[6/8]P"]) |
            ((df["close"] > df["[4/8]P"]) & (df["close"].shift(5) < df["[4/8]P"].shift(5)))
        )
        
        bearish_mml = (
            (df["close"] < df["[2/8]P"]) |
            ((df["close"] < df["[4/8]P"]) & (df["close"].shift(5) > df["[4/8]P"].shift(5)))
        )
        
        # MML resistance/support levels for exits
        at_resistance = (
            (df["high"] >= df["[6/8]P"]) |  # At 75%
            (df["high"] >= df["[7/8]P"]) |  # At 87.5%
            (df["high"] >= df["[8/8]P"])    # At 100%
        )
        
        at_support = (
            (df["low"] <= df["[2/8]P"]) |   # At 25%
            (df["low"] <= df["[1/8]P"]) |   # At 12.5%
            (df["low"] <= df["[0/8]P"])     # At 0%
        )
        
        # ===========================================
        # LONG EXIT SIGNALS (ADVANCED MML SYSTEM)
        # ===========================================
        
        # 1. Profit-Taking Exits
        long_exit_resistance_profit = (
            at_resistance &
            (df["close"] < df["high"]) &  # Failed to close at high
            (df["rsi"] > 65) &  # Overbought
            (df["maxima"] == 1) &  # Local top
            (df["volume"] > df["volume"].rolling(10).mean())
        )
        
        long_exit_extreme_overbought = (
            (df["close"] > df["[7/8]P"]) &
            (df["rsi"] > 75) &
            (df["close"] < df["close"].shift(1)) &  # Price turning down
            (df["maxima"] == 1)
        )
        
        long_exit_volume_exhaustion = (
            at_resistance &
            (df["volume"] < df["volume"].rolling(20).mean() * 0.6) &  # Tightened from 0.8
            (df["rsi"] > 70) &
            (df["close"] < df["close"].shift(1)) &
            (df["close"] < df["close"].rolling(3).mean())  # Added price confirmation
        )
        
        # 2. Structure Breakdown (Improved with strong filters)
        long_exit_structure_breakdown = (
            (df["close"] < df["[4/8]P"]) &
            (df["close"].shift(1) >= df["[4/8]P"].shift(1)) &
            bullish_mml.shift(1) &
            (df["close"] < df["[4/8]P"] * 0.995) &
            (df["close"] < df["close"].shift(1)) &
            (df["close"] < df["close"].shift(2)) &
            (df["rsi"] < 45) &  # Tightened from 50
            (df["volume"] > df["volume"].rolling(15).mean() * 2.0) &  # Increased from 1.5
            (df["close"] < df["open"]) &
            (df["low"] < df["low"].shift(1)) &
            (df["close"] < df["close"].rolling(3).mean()) &
            (df["momentum_quality"] < 0)  # Added momentum check
        )
        
        # 3. Momentum Divergence
        long_exit_momentum_divergence = (
            at_resistance &
            (df["rsi"] < df["rsi"].shift(1)) &  # RSI falling
            (df["rsi"].shift(1) < df["rsi"].shift(2)) &  # RSI was falling
            (df["rsi"] < df["rsi"].shift(3)) &  # 3-candle RSI decline
            (df["close"] >= df["close"].shift(1)) &  # Price still up/flat
            (df["maxima"] == 1) &
            (df["rsi"] > 60)  # Only in overbought territory
        )
        
        # 4. Range Exit
        long_exit_range = (
            (df["close"] >= df["[2/8]P"]) &
            (df["close"] <= df["[6/8]P"]) &  # In range
            (df["high"] >= df["[6/8]P"]) &  # HIGH touched 75%, not close
            (df["close"] < df["[6/8]P"] * 0.995) &  # But closed below
            (df["rsi"] > 65) &  # More conservative RSI
            (df["maxima"] == 1) &
            (df["volume"] > df["volume"].rolling(10).mean() * 1.2)  # Volume confirmation
        )
        
        # 5. Emergency Exit
        long_exit_emergency = (
            (df["close"] < df["[0/8]P"]) &
            (df["rsi"] < 20) &  # Changed from 15
            (df["volume"] > df["volume"].rolling(20).mean() * 2.5) &  # Reduced from 3
            (df["close"] < df["close"].shift(1)) &
            (df["close"] < df["close"].shift(2)) &
            (df["close"] < df["open"])
        ) if self.use_emergency_exits else pd.Series([False] * len(df), index=df.index)
        
        # Combine all Long Exit signals
        any_long_exit = (
            long_exit_resistance_profit |
            long_exit_extreme_overbought |
            long_exit_volume_exhaustion |
            long_exit_structure_breakdown |
            long_exit_momentum_divergence |
            long_exit_range |
            long_exit_emergency
        )
        
        # ===========================================
        # SHORT EXIT SIGNALS (if enabled)
        # ===========================================
        
        if self.can_short:
            # 1. Profit-Taking Exits
            short_exit_support_profit = (
                at_support &
                (df["close"] > df["low"]) &  # Failed to close at low
                (df["rsi"] < 35) &  # Oversold
                (df["minima"] == 1) &  # Local bottom
                (df["volume"] > df["volume"].rolling(10).mean())
            )
            
            short_exit_extreme_oversold = (
                (df["close"] < df["[1/8]P"]) &
                (df["rsi"] < 25) &
                (df["close"] > df["close"].shift(1)) &  # Price turning up
                (df["minima"] == 1)
            )
            
            short_exit_volume_exhaustion = (
                at_support &
                (df["volume"] < df["volume"].rolling(20).mean() * 0.6) &  # Tightened from 0.8
                (df["rsi"] < 30) &
                (df["close"] > df["close"].shift(1)) &
                (df["close"] > df["close"].rolling(3).mean())  # Added price confirmation
            )
            
            # 2. Structure Breakout
            short_exit_structure_breakout = (
                (df["close"] > df["[4/8]P"]) &
                (df["close"].shift(1) <= df["[4/8]P"].shift(1)) &
                bearish_mml.shift(1) &
                (df["close"] > df["[4/8]P"] * 1.005) &
                (df["close"] > df["close"].shift(1)) &
                (df["close"] > df["close"].shift(2)) &
                (df["rsi"] > 55) &  # Tightened from 50
                (df["volume"] > df["volume"].rolling(15).mean() * 2.0) &  # Increased from 1.5
                (df["close"] > df["open"]) &
                (df["high"] > df["high"].shift(1)) &
                (df["momentum_quality"] > 0)  # Added momentum check
            )
            
            # 3. Momentum Divergence
            short_exit_momentum_divergence = (
                at_support &
                (df["rsi"] > df["rsi"].shift(1)) &  # RSI rising
                (df["rsi"].shift(1) > df["rsi"].shift(2)) &  # RSI was rising
                (df["rsi"] > df["rsi"].shift(3)) &  # 3-candle RSI rise
                (df["close"] <= df["close"].shift(1)) &  # Price still down/flat
                (df["minima"] == 1) &
                (df["rsi"] < 40)  # Only in oversold territory
            )
            
            # 4. Range Exit
            short_exit_range = (
                (df["close"] >= df["[2/8]P"]) &
                (df["close"] <= df["[6/8]P"]) &  # In range
                (df["low"] <= df["[2/8]P"]) &  # LOW touched 25%
                (df["close"] > df["[2/8]P"] * 1.005) &  # But closed above
                (df["rsi"] < 35) &  # More conservative RSI
                (df["minima"] == 1) &
                (df["volume"] > df["volume"].rolling(10).mean() * 1.2)  # Volume confirmation
            )
            
            # 5. Emergency Exit
            short_exit_emergency = (
                (df["close"] > df["[8/8]P"]) &
                (df["rsi"] > 80) &  # Changed from 85
                (df["volume"] > df["volume"].rolling(20).mean() * 2.5) &  # Reduced from 3
                (df["close"] > df["close"].shift(1)) &
                (df["close"] > df["close"].shift(2)) &
                (df["close"] > df["open"])
            ) if self.use_emergency_exits else pd.Series([False] * len(df), index=df.index)
            
            # Combine all Short Exit signals
            any_short_exit = (
                short_exit_support_profit |
                short_exit_extreme_oversold |
                short_exit_volume_exhaustion |
                short_exit_structure_breakout |
                short_exit_momentum_divergence |
                short_exit_range |
                short_exit_emergency
            )
        else:
            any_short_exit = pd.Series([False] * len(df), index=df.index)
        
        # ===========================================
        # COORDINATION WITH ENTRY SIGNALS
        # ===========================================
        
        # If we have new Entry signals, they override Exit signals
        has_long_entry = "enter_long" in df.columns and (df["enter_long"] == 1).any()
        has_short_entry = "enter_short" in df.columns and (df["enter_short"] == 1).any()
        
        if has_long_entry:
            long_entry_mask = df["enter_long"] == 1
            any_long_exit = any_long_exit & (~long_entry_mask)
        
        if has_short_entry and self.can_short:
            short_entry_mask = df["enter_short"] == 1
            any_short_exit = any_short_exit & (~short_entry_mask)
        
        # ===========================================
        # SET FINAL EXIT SIGNALS AND TAGS
        # ===========================================
        
        # Long Exits
        df.loc[any_long_exit, "exit_long"] = 1
        
        # Tags for Long Exits (Priority: Emergency > Structure > Profit)
        df.loc[any_long_exit & long_exit_emergency, "exit_tag"] = "MML_Emergency_Long_Exit"
        df.loc[any_long_exit & long_exit_structure_breakdown & (df["exit_tag"] == ""), "exit_tag"] = "MML_Structure_Breakdown_Confirmed"
        df.loc[any_long_exit & long_exit_resistance_profit & (df["exit_tag"] == ""), "exit_tag"] = "MML_Resistance_Profit"
        df.loc[any_long_exit & long_exit_extreme_overbought & (df["exit_tag"] == ""), "exit_tag"] = "MML_Extreme_Overbought"
        df.loc[any_long_exit & long_exit_volume_exhaustion & (df["exit_tag"] == ""), "exit_tag"] = "MML_Volume_Exhaustion_Long"
        df.loc[any_long_exit & long_exit_momentum_divergence & (df["exit_tag"] == ""), "exit_tag"] = "MML_Momentum_Divergence_Long"
        df.loc[any_long_exit & long_exit_range & (df["exit_tag"] == ""), "exit_tag"] = "MML_Range_Exit_Long"
        
        # Short Exits
        if self.can_short:
            df.loc[any_short_exit, "exit_short"] = 1
            
            # Tags for Short Exits (Priority: Emergency > Structure > Profit)
            df.loc[any_short_exit & short_exit_emergency, "exit_tag"] = "MML_Emergency_Short_Exit"
            df.loc[any_short_exit & short_exit_structure_breakout & (df["exit_tag"] == ""), "exit_tag"] = "MML_Structure_Breakout_Confirmed"
            df.loc[any_short_exit & short_exit_support_profit & (df["exit_tag"] == ""), "exit_tag"] = "MML_Support_Profit"
            df.loc[any_short_exit & short_exit_extreme_oversold & (df["exit_tag"] == ""), "exit_tag"] = "MML_Extreme_Oversold"
            df.loc[any_short_exit & short_exit_volume_exhaustion & (df["exit_tag"] == ""), "exit_tag"] = "MML_Volume_Exhaustion_Short"
            df.loc[any_short_exit & short_exit_momentum_divergence & (df["exit_tag"] == ""), "exit_tag"] = "MML_Momentum_Divergence_Short"
            df.loc[any_short_exit & short_exit_range & (df["exit_tag"] == ""), "exit_tag"] = "MML_Range_Exit_Short"
        
        return df
    
    def _populate_simple_exits(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        SIMPLE OPPOSITE SIGNAL EXIT SYSTEM - SYNTAX FIXED
        """
        
        # Exit LONG when any SHORT signal appears
        long_exit_on_short = (dataframe["enter_short"] == 1)
        
        # Exit SHORT when any LONG signal appears  
        short_exit_on_long = (dataframe["enter_long"] == 1)
        
        # Emergency exits (if enabled)
        if self.use_emergency_exits:
            emergency_long_exit = (
                (dataframe['rsi'] > 85) &
                (dataframe['volume'] > dataframe['avg_volume'] * 3) &
                (dataframe['close'] < dataframe['open']) &
                (dataframe['close'] < dataframe['low'].shift(1))
            ) | (
                (dataframe.get('structure_break_down', 0) == 1) &
                (dataframe['volume'] > dataframe['avg_volume'] * 2.5) &
                (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 2)
            )
            
            emergency_short_exit = (
                (dataframe['rsi'] < 15) &
                (dataframe['volume'] > dataframe['avg_volume'] * 3) &
                (dataframe['close'] > dataframe['open']) &
                (dataframe['close'] > dataframe['high'].shift(1))
            ) | (
                (dataframe.get('structure_break_up', 0) == 1) &
                (dataframe['volume'] > dataframe['avg_volume'] * 2.5) &
                (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 2)
            )
        else:
            emergency_long_exit = pd.Series([False] * len(dataframe), index=dataframe.index)
            emergency_short_exit = pd.Series([False] * len(dataframe), index=dataframe.index)
        
        # Apply exits
        dataframe.loc[long_exit_on_short, "exit_long"] = 1
        dataframe.loc[long_exit_on_short, "exit_tag"] = "trend_reversal"
        
        dataframe.loc[short_exit_on_long, "exit_short"] = 1
        dataframe.loc[short_exit_on_long, "exit_tag"] = "trend_reversal"
        
        # Emergency exits
        dataframe.loc[emergency_long_exit & ~long_exit_on_short, "exit_long"] = 1
        dataframe.loc[emergency_long_exit & ~long_exit_on_short, "exit_tag"] = "emergency_exit"
        
        dataframe.loc[emergency_short_exit & ~short_exit_on_long, "exit_short"] = 1
        dataframe.loc[emergency_short_exit & ~short_exit_on_long, "exit_tag"] = "emergency_exit"
        
        # DEBUGGING (FIXED THE ERROR HERE)
        if metadata['pair'] in ['BTC/USDT:USDT', 'ETH/USDT:USDT']:
            recent_exits = dataframe['exit_long'].tail(5).sum() + dataframe['exit_short'].tail(5).sum()
            if recent_exits > 0:
                exit_tag = dataframe['exit_tag'].iloc[-1]
                logger.info(f"{metadata['pair']} EXIT SIGNAL - Tag: {exit_tag}")
                # âœ… FIXED: Use the correct attribute name
                logger.info(f"  Exit System: {'Custom MML' if self.use_custom_exits_advanced else 'Simple Opposite'}")
                logger.info(f"  RSI: {dataframe['rsi'].iloc[-1]:.1f}")
        
        return dataframe
  
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float, rate: float,
                          time_in_force: str, exit_reason: str, current_time: datetime, **kwargs) -> bool:
        current_profit_ratio = trade.calc_profit_ratio(rate)
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600  # Hours
        
        always_allow = [
            "stoploss", "stop_loss", "custom_stoploss",
            "roi", "trend_reversal", "emergency_exit"
        ]
        
        # Allow regime protection exits (icons)
        if any(char in exit_reason for char in ["⚡", "🔊", "🌊", "🎯", "₿"]):
            return True
        
        # Allow known good exits
        if exit_reason in always_allow:
            return True
        
        # FIXED: Previously blocked trailing stops if profit <= 0.
        # Now configurable & allows controlled negative trailing exits (prevents deeper drawdowns).
        if exit_reason in ["trailing_stop_loss", "trailing_stop"]:
            # If enabled, allow exit when profit >= configured minimal threshold
            if self.allow_trailing_exit_when_negative.value:
                if current_profit_ratio >= self.trailing_exit_min_profit.value:
                    logger.info(f"{pair} Allow trailing exit (thr={self.trailing_exit_min_profit.value:.3f}) "
                                f"Profit: {current_profit_ratio:.2%} Reason: {exit_reason}")
                    return True
                else:
                    logger.info(f"{pair} Blocking trailing exit below min threshold "
                                f"(profit {current_profit_ratio:.2%} < {self.trailing_exit_min_profit.value:.2%})")
                    return False
            else:
                # Legacy behaviour (only positive)
                if current_profit_ratio > 0:
                    logger.info(f"{pair} Allow trailing exit (legacy >0). Profit: {current_profit_ratio:.2%}")
                    return True
                logger.info(f"{pair} Blocking trailing exit (legacy rule). Profit: {current_profit_ratio:.2%}")
                return False
        
        # Time based safety exit (still keeps strategy responsive)
        if trade_duration > 4:
            logger.info(f"{pair} Forcing timed exit (4h). Profit: {current_profit_ratio:.2%}")
            return True
        
        return True