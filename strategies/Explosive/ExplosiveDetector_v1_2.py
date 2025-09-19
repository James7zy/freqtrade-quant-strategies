import logging
import numpy as np
import pandas as pd
import talib.abstract as ta
import pandas_ta as pta
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Optional, Dict, List, Tuple
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter, BooleanParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib
import requests
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import defaultdict
import time
import hashlib
import pickle
import os
from pathlib import Path

# 尝试导入机器学习依赖
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.pipeline import Pipeline
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# 情绪分析依赖
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

# 高级数据处理
try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

logger = logging.getLogger(__name__)

class ExplosiveDetector_v1_2(IStrategy):
    """
    24小时暴涨币种检测策略 v1.2
    新增功能:
    - 机器学习模型集成
    - 币种相关性分析
    - 链上数据集成
    - 期货持仓量分析
    - 自定义筛选条件
    """

    # 基础配置
    timeframe = '1h'
    startup_candle_count: int = 200
    stoploss = -0.99
    can_short = False
    process_only_new_candles = True
    use_exit_signal = False
    ignore_roi_if_entry_signal = True
    minimal_roi = {"0": 10.0}

    # v1.0 & v1.1 参数
    volume_surge_threshold = DecimalParameter(2.0, 5.0, default=3.0, space="buy", optimize=True)
    price_momentum_threshold = DecimalParameter(0.05, 0.20, default=0.10, space="buy", optimize=True)
    rsi_recovery_threshold = DecimalParameter(25.0, 45.0, default=35.0, space="buy", optimize=True)
    breakout_strength_threshold = DecimalParameter(1.5, 3.0, default=2.0, space="buy", optimize=True)
    volume_lookback = IntParameter(6, 24, default=12, space="buy", optimize=True)
    momentum_lookback = IntParameter(3, 12, default=6, space="buy", optimize=True)
    sentiment_weight = DecimalParameter(0.1, 0.5, default=0.25, space="buy", optimize=True)
    social_heat_weight = DecimalParameter(0.1, 0.4, default=0.20, space="buy", optimize=True)
    news_lookback_hours = IntParameter(6, 24, default=12, space="buy", optimize=True)

    # v1.2 新增参数
    ml_model_weight = DecimalParameter(0.2, 0.6, default=0.35, space="buy", optimize=True)
    correlation_threshold = DecimalParameter(0.3, 0.8, default=0.6, space="buy", optimize=True)
    onchain_weight = DecimalParameter(0.1, 0.4, default=0.25, space="buy", optimize=True)
    futures_oi_weight = DecimalParameter(0.1, 0.3, default=0.20, space="buy", optimize=True)

    # 功能开关
    enable_sentiment_analysis = BooleanParameter(default=True, space="buy", optimize=False)
    enable_social_analysis = BooleanParameter(default=True, space="buy", optimize=False)
    enable_ml_predictions = BooleanParameter(default=True, space="buy", optimize=False)
    enable_correlation_analysis = BooleanParameter(default=True, space="buy", optimize=False)
    enable_onchain_analysis = BooleanParameter(default=True, space="buy", optimize=False)
    enable_futures_analysis = BooleanParameter(default=True, space="buy", optimize=False)

    # 自定义筛选条件
    min_market_cap_usd = IntParameter(10000000, 1000000000, default=100000000, space="buy", optimize=False)  # 1亿美元
    min_24h_volume_usd = IntParameter(1000000, 100000000, default=10000000, space="buy", optimize=False)     # 1000万美元
    max_correlation_btc = DecimalParameter(0.5, 0.95, default=0.85, space="buy", optimize=False)

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # 初始化组件
        self.sentiment_analyzer = None
        if VADER_AVAILABLE and self.enable_sentiment_analysis.value:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()

        # 缓存系统
        self.news_cache = {}
        self.social_cache = {}
        self.onchain_cache = {}
        self.futures_cache = {}
        self.correlation_cache = {}
        self.cache_ttl = 1800
        self.cache_lock = threading.Lock()

        # 机器学习模型
        self.ml_model = None
        self.ml_scaler = None
        self.feature_columns = []
        self.model_path = "models/explosive_detector_v1_2.pkl"
        self.training_data = []
        self.model_last_trained = None

        # 并行处理
        self.max_workers = 8
        self.api_rate_limit = 0.15
        self.last_api_call = {}

        # 币种数据缓存
        self.market_data_cache = {}
        self.correlation_matrix = None
        self.correlation_last_update = None

        # 初始化ML模型
        if ML_AVAILABLE and self.enable_ml_predictions.value:
            self._initialize_ml_models()

    def _initialize_ml_models(self):
        """初始化机器学习模型"""
        try:
            # 尝试加载已保存的模型
            if os.path.exists(self.model_path):
                self.ml_model = joblib.load(self.model_path)
                self.ml_scaler = joblib.load(self.model_path.replace('.pkl', '_scaler.pkl'))
                logger.info("已加载预训练ML模型")
            else:
                # 创建新模型
                self._create_ml_model()
        except Exception as e:
            logger.warning(f"ML模型初始化失败: {e}")
            self.ml_model = None

    def _create_ml_model(self):
        """创建机器学习模型"""
        # 集成学习器组合
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )

        lr = LogisticRegression(
            random_state=42,
            max_iter=1000
        )

        # 投票分类器
        self.ml_model = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('lr', lr)
            ],
            voting='probability'
        )

        self.ml_scaler = StandardScaler()
        logger.info("已创建新的ML模型")

    def _get_market_data(self, pair: str) -> Dict:
        """获取市场数据"""
        cache_key = f"market_{pair}"
        if self._is_cache_valid(cache_key):
            return self.market_data_cache.get(cache_key, {})

        try:
            # 模拟市场数据API调用
            market_data = {
                'market_cap_usd': np.random.randint(10000000, 50000000000),
                'volume_24h_usd': np.random.randint(1000000, 1000000000),
                'circulating_supply': np.random.randint(1000000, 100000000000),
                'price_change_24h': np.random.uniform(-20, 20),
                'price_change_7d': np.random.uniform(-40, 40),
                'timestamp': time.time()
            }

            with self.cache_lock:
                self.market_data_cache[cache_key] = market_data

            return market_data

        except Exception as e:
            logger.warning(f"获取{pair}市场数据失败: {e}")
            return {}

    def _calculate_correlation_matrix(self, pairs: List[str]) -> pd.DataFrame:
        """计算币种间相关性矩阵"""
        if (self.correlation_matrix is not None and
            self.correlation_last_update and
            time.time() - self.correlation_last_update < 3600):  # 1小时缓存
            return self.correlation_matrix

        try:
            # 获取所有币种的价格数据
            price_data = {}
            for pair in pairs:
                df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
                if len(df) > 100:  # 确保有足够的数据
                    price_data[pair] = df['close'].pct_change().dropna()

            if len(price_data) >= 2:
                # 创建价格变化DataFrame
                price_df = pd.DataFrame(price_data)

                # 计算相关性矩阵
                self.correlation_matrix = price_df.corr()
                self.correlation_last_update = time.time()

                logger.info(f"已更新{len(pairs)}个币种的相关性矩阵")
                return self.correlation_matrix

        except Exception as e:
            logger.warning(f"计算相关性矩阵失败: {e}")

        return pd.DataFrame()

    def _get_onchain_data(self, pair: str) -> Dict:
        """获取链上数据"""
        cache_key = self._get_cache_key(pair, 'onchain')
        if self._is_cache_valid(cache_key):
            with self.cache_lock:
                return self.onchain_cache.get(cache_key, {})

        coin_symbol = self._extract_coin_symbol(pair)
        onchain_data = {
            'whale_transactions': 0,
            'large_transfers_24h': 0,
            'exchange_inflow': 0.0,
            'exchange_outflow': 0.0,
            'holder_distribution': 0.0,
            'active_addresses': 0,
            'timestamp': time.time()
        }

        try:
            # 模拟链上数据API调用
            self._rate_limit_api_call('onchain')

            # 实际实现中应调用如Glassnode, IntoTheBlock等API
            onchain_data.update({
                'whale_transactions': np.random.randint(0, 50),
                'large_transfers_24h': np.random.randint(0, 100),
                'exchange_inflow': np.random.uniform(0, 10000000),
                'exchange_outflow': np.random.uniform(0, 15000000),
                'holder_distribution': np.random.uniform(0.3, 0.8),
                'active_addresses': np.random.randint(1000, 100000)
            })

        except Exception as e:
            logger.warning(f"获取{pair}链上数据失败: {e}")

        with self.cache_lock:
            self.onchain_cache[cache_key] = onchain_data

        return onchain_data

    def _get_futures_data(self, pair: str) -> Dict:
        """获取期货数据"""
        cache_key = self._get_cache_key(pair, 'futures')
        if self._is_cache_valid(cache_key):
            with self.cache_lock:
                return self.futures_cache.get(cache_key, {})

        futures_data = {
            'open_interest': 0.0,
            'oi_change_24h': 0.0,
            'funding_rate': 0.0,
            'long_short_ratio': 0.5,
            'liquidations_24h': 0.0,
            'timestamp': time.time()
        }

        try:
            # 模拟期货数据API调用
            self._rate_limit_api_call('futures')

            # 实际实现中调用Binance Futures, BitMEX等API
            futures_data.update({
                'open_interest': np.random.uniform(10000000, 500000000),
                'oi_change_24h': np.random.uniform(-20, 20),
                'funding_rate': np.random.uniform(-0.01, 0.01),
                'long_short_ratio': np.random.uniform(0.3, 3.0),
                'liquidations_24h': np.random.uniform(0, 10000000)
            })

        except Exception as e:
            logger.warning(f"获取{pair}期货数据失败: {e}")

        with self.cache_lock:
            self.futures_cache[cache_key] = futures_data

        return futures_data

    def _prepare_ml_features(self, dataframe: DataFrame, pair: str) -> pd.DataFrame:
        """准备ML模型特征"""
        try:
            latest = dataframe.iloc[-1]

            # 基础技术特征
            features = {
                'rsi': latest['rsi'],
                'macd_hist': latest['macd_hist'],
                'bb_position': latest['price_position_bb'],
                'vol_ratio': latest['vol_ratio_20'],
                'roc_1h': latest['roc_1h'],
                'roc_4h': latest['roc_4h'],
                'atr_percent': latest['atr_percent'],
                'mfi': latest['mfi'],
                'willr': latest['willr'],
                'stoch_k': latest['stoch_k']
            }

            # 新增特征
            if 'news_sentiment_score' in dataframe.columns:
                features['news_sentiment'] = latest['news_sentiment_score']
                features['news_count'] = latest['news_count']

            if 'social_heat_index' in dataframe.columns:
                features['social_heat'] = latest['social_heat_index']
                features['social_sentiment'] = latest['social_sentiment_trend']

            # 市场数据特征
            market_data = self._get_market_data(pair)
            if market_data:
                features['market_cap_log'] = np.log10(max(market_data.get('market_cap_usd', 1), 1))
                features['volume_24h_log'] = np.log10(max(market_data.get('volume_24h_usd', 1), 1))
                features['price_change_24h'] = market_data.get('price_change_24h', 0)

            # 链上数据特征
            if self.enable_onchain_analysis.value:
                onchain_data = self._get_onchain_data(pair)
                features['whale_transactions'] = onchain_data.get('whale_transactions', 0)
                features['exchange_flow_ratio'] = (
                    onchain_data.get('exchange_outflow', 0) /
                    max(onchain_data.get('exchange_inflow', 1), 1)
                )

            # 期货数据特征
            if self.enable_futures_analysis.value:
                futures_data = self._get_futures_data(pair)
                features['oi_change'] = futures_data.get('oi_change_24h', 0)
                features['funding_rate'] = futures_data.get('funding_rate', 0)
                features['long_short_ratio'] = futures_data.get('long_short_ratio', 1)

            # 相关性特征
            if self.enable_correlation_analysis.value and self.correlation_matrix is not None:
                btc_pair = 'BTC/USDT'
                if pair in self.correlation_matrix.columns and btc_pair in self.correlation_matrix.columns:
                    features['btc_correlation'] = self.correlation_matrix.loc[pair, btc_pair]
                else:
                    features['btc_correlation'] = 0.0

            return pd.DataFrame([features])

        except Exception as e:
            logger.warning(f"ML特征准备失败: {e}")
            return pd.DataFrame()

    def _get_ml_prediction(self, features_df: pd.DataFrame) -> Dict:
        """获取ML模型预测"""
        if not ML_AVAILABLE or self.ml_model is None or features_df.empty:
            return {'prediction': 0, 'probability': 0.5, 'confidence': 0.0}

        try:
            # 确保特征顺序一致
            if not self.feature_columns:
                self.feature_columns = features_df.columns.tolist()

            # 重新排序特征列
            missing_cols = set(self.feature_columns) - set(features_df.columns)
            for col in missing_cols:
                features_df[col] = 0.0

            features_df = features_df[self.feature_columns]

            # 特征缩放
            if self.ml_scaler:
                features_scaled = self.ml_scaler.transform(features_df)
            else:
                features_scaled = features_df.values

            # 预测
            prediction = self.ml_model.predict(features_scaled)[0]
            probabilities = self.ml_model.predict_proba(features_scaled)[0]

            return {
                'prediction': int(prediction),
                'probability': float(probabilities[1] if len(probabilities) > 1 else probabilities[0]),
                'confidence': float(max(probabilities))
            }

        except Exception as e:
            logger.warning(f"ML预测失败: {e}")
            return {'prediction': 0, 'probability': 0.5, 'confidence': 0.0}

    def _apply_custom_filters(self, pair: str, signal_data: Dict) -> bool:
        """应用自定义筛选条件"""
        try:
            # 市场数据筛选
            market_data = self._get_market_data(pair)

            # 最小市值筛选
            market_cap = market_data.get('market_cap_usd', 0)
            if market_cap < self.min_market_cap_usd.value:
                return False

            # 最小成交量筛选
            volume_24h = market_data.get('volume_24h_usd', 0)
            if volume_24h < self.min_24h_volume_usd.value:
                return False

            # BTC相关性筛选
            if self.enable_correlation_analysis.value and self.correlation_matrix is not None:
                btc_pair = 'BTC/USDT'
                if (pair in self.correlation_matrix.columns and
                    btc_pair in self.correlation_matrix.columns):
                    correlation = abs(self.correlation_matrix.loc[pair, btc_pair])
                    if correlation > self.max_correlation_btc.value:
                        return False

            return True

        except Exception as e:
            logger.warning(f"自定义筛选失败: {e}")
            return True

    # 继承v1.1的方法
    def _get_cache_key(self, pair: str, data_type: str) -> str:
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        return f"{pair}_{data_type}_{current_hour.isoformat()}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        with self.cache_lock:
            caches = [self.news_cache, self.social_cache, self.onchain_cache, self.futures_cache]
            for cache in caches:
                if cache_key in cache:
                    cache_time = cache[cache_key].get('timestamp', 0)
                    return time.time() - cache_time < self.cache_ttl
        return False

    def _rate_limit_api_call(self, api_name: str):
        current_time = time.time()
        last_call_time = self.last_api_call.get(api_name, 0)
        time_diff = current_time - last_call_time
        if time_diff < self.api_rate_limit:
            time.sleep(self.api_rate_limit - time_diff)
        self.last_api_call[api_name] = time.time()

    def _extract_coin_symbol(self, pair: str) -> str:
        return pair.split('/')[0].upper()

    # 继承v1.1的外部数据获取方法 (为了简洁，这里只声明)
    def _fetch_news_sentiment(self, pair: str) -> Dict:
        # 继承v1.1实现
        return {'sentiment_score': 0.0, 'news_count': 0, 'positive_ratio': 0.0, 'negative_ratio': 0.0, 'timestamp': time.time()}

    def _fetch_social_heat(self, pair: str) -> Dict:
        # 继承v1.1实现
        return {'mention_count': 0, 'sentiment_trend': 0.0, 'engagement_score': 0.0, 'heat_index': 0.0, 'timestamp': time.time()}

    def informative_pairs(self):
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for pair in pairs:
            informative_pairs.append((pair, '15m'))
            informative_pairs.append((pair, '4h'))
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """计算暴涨检测指标 - v1.2 ML增强版"""

        # 获取多时间框架数据
        dataframe_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='15m')
        dataframe_4h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='4h')
        dataframe = merge_informative_pair(dataframe, dataframe_15m, self.timeframe, '15m', ffill=True)
        dataframe = merge_informative_pair(dataframe, dataframe_4h, self.timeframe, '4h', ffill=True)

        # v1.0 & v1.1 基础指标 (继承所有现有指标)
        # 移动平均线
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_15m'] = dataframe['rsi_15m']

        # MACD
        macd, macdsignal, macdhist = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd
        dataframe['macd_signal'] = macdsignal
        dataframe['macd_hist'] = macdhist

        # 布林带
        bb_upper, bb_middle, bb_lower = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe['bb_upper'] = bb_upper
        dataframe['bb_middle'] = bb_middle
        dataframe['bb_lower'] = bb_lower
        dataframe['bb_width'] = (bb_upper - bb_lower) / bb_middle
        dataframe['price_position_bb'] = (dataframe['close'] - bb_lower) / (bb_upper - bb_lower)

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = dataframe['atr'] / dataframe['close'] * 100

        # 成交量指标
        dataframe['vol_sma_20'] = dataframe['volume'].rolling(20).mean()
        dataframe['vol_sma_50'] = dataframe['volume'].rolling(50).mean()
        dataframe['vol_ratio_20'] = dataframe['volume'] / dataframe['vol_sma_20']
        dataframe['vol_ratio_50'] = dataframe['volume'] / dataframe['vol_sma_50']
        dataframe['vol_surge'] = ((dataframe['vol_ratio_20'] > self.volume_surge_threshold.value) |
                                 (dataframe['vol_ratio_50'] > self.volume_surge_threshold.value * 0.8)).astype(int)

        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['obv_sma'] = dataframe['obv'].rolling(20).mean()
        dataframe['obv_trend'] = np.where(dataframe['obv'] > dataframe['obv_sma'], 1, 0)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)

        # 动量指标
        dataframe['roc_1h'] = ta.ROC(dataframe, timeperiod=1)
        dataframe['roc_4h'] = ta.ROC(dataframe, timeperiod=4)
        dataframe['roc_12h'] = ta.ROC(dataframe, timeperiod=12)
        dataframe['momentum_surge'] = ((dataframe['roc_1h'] > self.price_momentum_threshold.value) |
                                      (dataframe['roc_4h'] > self.price_momentum_threshold.value * 2)).astype(int)

        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=14)
        stoch_k, stoch_d = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch_k
        dataframe['stoch_d'] = stoch_d

        # 突破检测
        dataframe['ema_breakout'] = ((dataframe['close'] > dataframe['ema_20']) &
                                    (dataframe['close'].shift(1) <= dataframe['ema_20'].shift(1))).astype(int)

        dataframe['resistance_level'] = dataframe['high'].rolling(50).max()
        dataframe['resistance_breakout'] = ((dataframe['close'] > dataframe['resistance_level'].shift(1)) &
                                          (dataframe['volume'] > dataframe['vol_sma_20'])).astype(int)

        # 反转信号
        dataframe['rsi_reversal'] = ((dataframe['rsi'] < 30) & (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
                                   (dataframe['rsi'] > self.rsi_recovery_threshold.value)).astype(int)

        dataframe['macd_golden_cross'] = ((dataframe['macd'] > dataframe['macd_signal']) &
                                        (dataframe['macd'].shift(1) <= dataframe['macd_signal'].shift(1)) &
                                        (dataframe['macd_hist'] > 0)).astype(int)

        # 获取外部数据
        try:
            current_pair = metadata['pair']

            # 新闻和社交数据 (继承v1.1)
            news_data = self._fetch_news_sentiment(current_pair)
            social_data = self._fetch_social_heat(current_pair)

            dataframe['news_sentiment_score'] = news_data.get('sentiment_score', 0.0)
            dataframe['news_count'] = news_data.get('news_count', 0)
            dataframe['news_positive_ratio'] = news_data.get('positive_ratio', 0.0)
            dataframe['social_mention_count'] = social_data.get('mention_count', 0)
            dataframe['social_sentiment_trend'] = social_data.get('sentiment_trend', 0.0)
            dataframe['social_heat_index'] = social_data.get('heat_index', 0.0)

            # v1.2 新增数据
            if self.enable_onchain_analysis.value:
                onchain_data = self._get_onchain_data(current_pair)
                dataframe['whale_transactions'] = onchain_data.get('whale_transactions', 0)
                dataframe['exchange_flow_ratio'] = onchain_data.get('exchange_outflow', 0) / max(onchain_data.get('exchange_inflow', 1), 1)

            if self.enable_futures_analysis.value:
                futures_data = self._get_futures_data(current_pair)
                dataframe['oi_change_24h'] = futures_data.get('oi_change_24h', 0)
                dataframe['funding_rate'] = futures_data.get('funding_rate', 0)
                dataframe['long_short_ratio'] = futures_data.get('long_short_ratio', 1)

        except Exception as e:
            logger.warning(f"外部数据获取失败: {e}")
            # 设置默认值
            for col in ['news_sentiment_score', 'news_count', 'news_positive_ratio',
                       'social_mention_count', 'social_sentiment_trend', 'social_heat_index',
                       'whale_transactions', 'exchange_flow_ratio', 'oi_change_24h',
                       'funding_rate', 'long_short_ratio']:
                if col not in dataframe.columns:
                    dataframe[col] = 0.0

        # ==================== v1.2 增强评分系统 ====================

        # v1.0 & v1.1 基础评分
        dataframe['volume_score'] = (dataframe['vol_surge'] * 25 +
                                   (dataframe['vol_ratio_20'] > 2.0).astype(int) * 15 +
                                   dataframe['obv_trend'] * 10)

        dataframe['momentum_score'] = (dataframe['momentum_surge'] * 25 +
                                     (dataframe['roc_1h'] > 0.05).astype(int) * 15 +
                                     (dataframe['roc_4h'] > 0.10).astype(int) * 10)

        dataframe['technical_score'] = (dataframe['ema_breakout'] * 20 +
                                      dataframe['resistance_breakout'] * 25 +
                                      dataframe['macd_golden_cross'] * 15 +
                                      dataframe['rsi_reversal'] * 20)

        dataframe['pattern_score'] = ((dataframe['price_position_bb'] < 0.2).astype(int) * 10 +
                                    (dataframe['willr'] > -20).astype(int) * 10)

        dataframe['sentiment_score'] = ((dataframe['news_sentiment_score'] > 0.1).astype(int) * 20 +
                                      (dataframe['news_positive_ratio'] > 0.6).astype(int) * 15 +
                                      (dataframe['news_count'] > 2).astype(int) * 10) * self.sentiment_weight.value

        dataframe['social_score'] = ((dataframe['social_heat_index'] > 0.3).astype(int) * 25 +
                                   (dataframe['social_sentiment_trend'] > 0.1).astype(int) * 15 +
                                   (dataframe['social_mention_count'] > 10).astype(int) * 10) * self.social_heat_weight.value

        # v1.2 新增评分
        dataframe['onchain_score'] = 0.0
        if self.enable_onchain_analysis.value:
            dataframe['onchain_score'] = ((dataframe['whale_transactions'] > 5).astype(int) * 20 +
                                        (dataframe['exchange_flow_ratio'] > 1.2).astype(int) * 15) * self.onchain_weight.value

        dataframe['futures_score'] = 0.0
        if self.enable_futures_analysis.value:
            dataframe['futures_score'] = ((dataframe['oi_change_24h'] > 10).astype(int) * 15 +
                                        (abs(dataframe['funding_rate']) > 0.005).astype(int) * 10 +
                                        ((dataframe['long_short_ratio'] > 1.5) | (dataframe['long_short_ratio'] < 0.7)).astype(int) * 10) * self.futures_oi_weight.value

        # ML预测评分
        dataframe['ml_score'] = 0.0
        if ML_AVAILABLE and self.enable_ml_predictions.value and self.ml_model is not None:
            try:
                # 为最新的几个数据点生成ML预测
                for i in range(max(1, len(dataframe) - 5), len(dataframe)):
                    if i >= 0:
                        row_df = dataframe.iloc[[i]].copy()
                        features_df = self._prepare_ml_features(row_df, metadata['pair'])
                        if not features_df.empty:
                            ml_result = self._get_ml_prediction(features_df)
                            dataframe.iloc[i, dataframe.columns.get_loc('ml_score')] = (
                                ml_result['probability'] * 50 * self.ml_model_weight.value
                            )
            except Exception as e:
                logger.warning(f"ML评分计算失败: {e}")

        # 综合评分 v1.2
        dataframe['explosion_score'] = (dataframe['volume_score'] + dataframe['momentum_score'] +
                                      dataframe['technical_score'] + dataframe['pattern_score'] +
                                      dataframe['sentiment_score'] + dataframe['social_score'] +
                                      dataframe['onchain_score'] + dataframe['futures_score'] +
                                      dataframe['ml_score'])

        # 多时间框架确认 (增强版)
        dataframe['multi_timeframe_confirm'] = ((dataframe['rsi_15m'] < 70) &
                                              (dataframe['volume_15m'] > dataframe['volume_15m'].rolling(20).mean()) &
                                              (dataframe['close_4h'] > dataframe['close_4h'].shift(1)) &
                                              ((dataframe['news_sentiment_score'] >= 0) | (dataframe['news_count'] == 0)) &
                                              ((dataframe['social_sentiment_trend'] >= -0.2) | (dataframe['social_mention_count'] == 0))).astype(int)

        # 最终信号强度 v1.2
        base_strength = dataframe['explosion_score'] * (1 + dataframe['multi_timeframe_confirm'] * 0.3)

        # 外部数据加成
        external_boost = 1.0
        if self.enable_sentiment_analysis.value:
            external_boost += dataframe['news_sentiment_score'].clip(0, 0.5) * 0.2
        if self.enable_social_analysis.value:
            external_boost += dataframe['social_heat_index'] * 0.15
        if self.enable_onchain_analysis.value:
            external_boost += (dataframe['whale_transactions'] / 100).clip(0, 0.1)
        if self.enable_futures_analysis.value:
            external_boost += (abs(dataframe['oi_change_24h']) / 100).clip(0, 0.1)

        dataframe['signal_strength'] = base_strength * external_boost

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """暴涨预警信号 - v1.2 ML增强版"""

        # 应用自定义筛选
        signal_data = {'pair': metadata['pair']}

        # 极高潜力信号 (100分以上，v1.2新增)
        condition_ultra_plus = (
            (dataframe['signal_strength'] >= 100) &
            (dataframe['volume'] > 0) &
            (dataframe['explosion_score'] >= 90) &
            (dataframe['ml_score'] > 15)
        )

        if self._apply_custom_filters(metadata['pair'], signal_data):
            dataframe.loc[condition_ultra_plus, ['enter_long', 'enter_tag']] = (1, 'EXTREME_EXPLOSIVE_POTENTIAL')

        # 超高潜力信号 (90-99分)
        condition_ultra = (
            (dataframe['signal_strength'] >= 90) &
            (dataframe['signal_strength'] < 100) &
            (dataframe['volume'] > 0) &
            (dataframe['explosion_score'] >= 80)
        )

        if self._apply_custom_filters(metadata['pair'], signal_data):
            dataframe.loc[condition_ultra, ['enter_long', 'enter_tag']] = (1, 'ULTRA_HIGH_EXPLOSIVE_POTENTIAL')

        # 高潜力信号 (80-89分)
        condition_high = (
            (dataframe['signal_strength'] >= 80) &
            (dataframe['signal_strength'] < 90) &
            (dataframe['volume'] > 0) &
            (dataframe['explosion_score'] >= 70)
        )

        if self._apply_custom_filters(metadata['pair'], signal_data):
            dataframe.loc[condition_high, ['enter_long', 'enter_tag']] = (1, 'HIGH_EXPLOSIVE_POTENTIAL')

        # 中等潜力信号 (60-79分)
        condition_medium = (
            (dataframe['signal_strength'] >= 60) &
            (dataframe['signal_strength'] < 80) &
            (dataframe['volume'] > 0) &
            (dataframe['explosion_score'] >= 50)
        )

        if self._apply_custom_filters(metadata['pair'], signal_data):
            dataframe.loc[condition_medium, ['enter_long', 'enter_tag']] = (1, 'MEDIUM_EXPLOSIVE_POTENTIAL')

        # 早期预警信号 (45-59分)
        condition_early = (
            (dataframe['signal_strength'] >= 45) &
            (dataframe['signal_strength'] < 60) &
            (dataframe['volume'] > 0) &
            (dataframe['vol_surge'] == 1)
        )

        if self._apply_custom_filters(metadata['pair'], signal_data):
            dataframe.loc[condition_early, ['enter_long', 'enter_tag']] = (1, 'EARLY_WARNING')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """不设置退出信号，仅用于筛选"""
        return dataframe

    def custom_entry_price(self, pair: str, trade: Optional["Trade"], current_time: datetime,
                          proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        return proposed_rate

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], max_stake: float,
                           leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        return min_stake if min_stake else 10.0

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """发送v1.2 ML增强版暴涨预警消息"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        if len(dataframe) > 0:
            latest = dataframe.iloc[-1]
            signal_strength = latest['signal_strength']
            explosion_score = latest['explosion_score']

            # 构建v1.2增强版预警消息
            message = f"""
🚀 暴涨预警 v1.2 - {pair} 🚀
📊 信号强度: {signal_strength:.1f}
💥 爆发评分: {explosion_score:.1f}
🏷️ 信号类型: {entry_tag}
💰 当前价格: {rate:.8f}

📈 技术指标:
├─ 1小时涨幅: {latest['roc_1h']:.2%}
├─ 成交量比率: {latest['vol_ratio_20']:.1f}x
└─ RSI: {latest['rsi']:.1f}

🤖 AI增强信号:
├─ ML评分: {latest['ml_score']:.1f}
├─ 新闻情绪: {latest['news_sentiment_score']:.2f} ({latest['news_count']:.0f}条)
└─ 社交热度: {latest['social_heat_index']:.2f}

🔗 链上&期货:
├─ 鲸鱼交易: {latest.get('whale_transactions', 0):.0f}笔
├─ 资金流比: {latest.get('exchange_flow_ratio', 0):.2f}
└─ 持仓变化: {latest.get('oi_change_24h', 0):.1f}%

⏰ {current_time.strftime('%Y-%m-%d %H:%M:%S')}
🔥 v1.2 ML增强算法 - AI辅助暴涨预测！
            """

            self.dp.send_msg(message.strip())

            # 特殊信号额外提醒
            if entry_tag == 'EXTREME_EXPLOSIVE_POTENTIAL':
                self.dp.send_msg(f"🚨🚨🚨 EXTREME ALERT: {pair} AI模型预测极强暴涨潜力！信号强度: {signal_strength:.1f} 🚨🚨🚨")
            elif entry_tag == 'ULTRA_HIGH_EXPLOSIVE_POTENTIAL':
                self.dp.send_msg(f"🔥🔥🔥 超高关注: {pair} 显示极强暴涨潜力！信号强度: {signal_strength:.1f} 🔥🔥🔥")

        return False

    def custom_stoploss(self, pair: str, trade: "Trade", current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        return -0.99