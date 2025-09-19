import logging
import numpy as np
import pandas as pd
import talib.abstract as ta
import pandas_ta as pta
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Optional, Dict, List
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

# 尝试导入可选依赖
try:
    import vaderSentiment
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

try:
    import textblob
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

logger = logging.getLogger(__name__)

class ExplosiveDetector_v1_1(IStrategy):
    """
    24小时暴涨币种检测策略 v1.1
    新增功能:
    - 新闻情绪分析
    - 社交媒体热度指标
    - 多币种并行处理优化
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

    # v1.0 检测参数
    volume_surge_threshold = DecimalParameter(2.0, 5.0, default=3.0, space="buy", optimize=True)
    price_momentum_threshold = DecimalParameter(0.05, 0.20, default=0.10, space="buy", optimize=True)
    rsi_recovery_threshold = DecimalParameter(25.0, 45.0, default=35.0, space="buy", optimize=True)
    breakout_strength_threshold = DecimalParameter(1.5, 3.0, default=2.0, space="buy", optimize=True)
    volume_lookback = IntParameter(6, 24, default=12, space="buy", optimize=True)
    momentum_lookback = IntParameter(3, 12, default=6, space="buy", optimize=True)

    # v1.1 新增参数
    sentiment_weight = DecimalParameter(0.1, 0.5, default=0.25, space="buy", optimize=True)
    social_heat_weight = DecimalParameter(0.1, 0.4, default=0.20, space="buy", optimize=True)
    news_lookback_hours = IntParameter(6, 24, default=12, space="buy", optimize=True)
    enable_sentiment_analysis = BooleanParameter(default=True, space="buy", optimize=False)
    enable_social_analysis = BooleanParameter(default=True, space="buy", optimize=False)

    def __init__(self, config: dict) -> None:
        super().__init__(config)

        # 情绪分析器初始化
        self.sentiment_analyzer = None
        if VADER_AVAILABLE and self.enable_sentiment_analysis.value:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER情绪分析器已启用")

        # 缓存系统
        self.news_cache = {}
        self.social_cache = {}
        self.cache_ttl = 1800  # 30分钟缓存
        self.cache_lock = threading.Lock()

        # 并行处理配置
        self.max_workers = 5
        self.api_rate_limit = 0.2  # 200ms between API calls
        self.last_api_call = {}

    def informative_pairs(self):
        """多时间框架数据"""
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for pair in pairs:
            informative_pairs.append((pair, '15m'))
            informative_pairs.append((pair, '4h'))
        return informative_pairs

    def _get_cache_key(self, pair: str, data_type: str) -> str:
        """生成缓存键"""
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        return f"{pair}_{data_type}_{current_hour.isoformat()}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """检查缓存是否有效"""
        with self.cache_lock:
            if cache_key in self.news_cache:
                cache_time = self.news_cache[cache_key].get('timestamp', 0)
                return time.time() - cache_time < self.cache_ttl
            if cache_key in self.social_cache:
                cache_time = self.social_cache[cache_key].get('timestamp', 0)
                return time.time() - cache_time < self.cache_ttl
        return False

    def _rate_limit_api_call(self, api_name: str):
        """API调用频率限制"""
        current_time = time.time()
        last_call_time = self.last_api_call.get(api_name, 0)
        time_diff = current_time - last_call_time

        if time_diff < self.api_rate_limit:
            time.sleep(self.api_rate_limit - time_diff)

        self.last_api_call[api_name] = time.time()

    def _extract_coin_symbol(self, pair: str) -> str:
        """从交易对中提取币种符号"""
        return pair.split('/')[0].upper()

    def _fetch_news_sentiment(self, pair: str) -> Dict:
        """获取新闻情绪分析数据"""
        cache_key = self._get_cache_key(pair, 'news')

        # 检查缓存
        if self._is_cache_valid(cache_key):
            with self.cache_lock:
                return self.news_cache[cache_key]

        coin_symbol = self._extract_coin_symbol(pair)
        sentiment_data = {
            'sentiment_score': 0.0,
            'news_count': 0,
            'positive_ratio': 0.0,
            'negative_ratio': 0.0,
            'timestamp': time.time()
        }

        try:
            # 模拟新闻API调用 (实际使用时替换为真实API)
            self._rate_limit_api_call('news')

            # 这里应该是真实的新闻API调用
            # 示例: CryptoPanic, NewsAPI, CoinDesk API等
            news_data = self._mock_news_api(coin_symbol)

            if news_data and self.sentiment_analyzer:
                sentiments = []
                positive_count = 0
                negative_count = 0
                total_count = len(news_data)

                for article in news_data:
                    title = article.get('title', '')
                    content = article.get('content', '')
                    text = f"{title} {content}"

                    # VADER情绪分析
                    scores = self.sentiment_analyzer.polarity_scores(text)
                    compound_score = scores['compound']
                    sentiments.append(compound_score)

                    if compound_score > 0.1:
                        positive_count += 1
                    elif compound_score < -0.1:
                        negative_count += 1

                if sentiments:
                    sentiment_data.update({
                        'sentiment_score': np.mean(sentiments),
                        'news_count': total_count,
                        'positive_ratio': positive_count / total_count if total_count > 0 else 0,
                        'negative_ratio': negative_count / total_count if total_count > 0 else 0
                    })

        except Exception as e:
            logger.warning(f"获取{pair}新闻情绪数据失败: {e}")

        # 更新缓存
        with self.cache_lock:
            self.news_cache[cache_key] = sentiment_data

        return sentiment_data

    def _fetch_social_heat(self, pair: str) -> Dict:
        """获取社交媒体热度数据"""
        cache_key = self._get_cache_key(pair, 'social')

        # 检查缓存
        if self._is_cache_valid(cache_key):
            with self.cache_lock:
                return self.social_cache[cache_key]

        coin_symbol = self._extract_coin_symbol(pair)
        heat_data = {
            'mention_count': 0,
            'sentiment_trend': 0.0,
            'engagement_score': 0.0,
            'heat_index': 0.0,
            'timestamp': time.time()
        }

        try:
            # 模拟社交媒体API调用
            self._rate_limit_api_call('social')

            # 这里应该是真实的社交媒体API调用
            # 示例: Twitter API, Reddit API, Telegram群组分析等
            social_data = self._mock_social_api(coin_symbol)

            if social_data:
                mentions = social_data.get('mentions', [])
                if mentions:
                    # 计算热度指标
                    mention_count = len(mentions)
                    recent_mentions = [m for m in mentions if self._is_recent_mention(m)]

                    # 情绪趋势分析
                    if self.sentiment_analyzer:
                        sentiments = []
                        engagement_scores = []

                        for mention in recent_mentions:
                            text = mention.get('text', '')
                            scores = self.sentiment_analyzer.polarity_scores(text)
                            sentiments.append(scores['compound'])

                            # 简单的参与度评分
                            likes = mention.get('likes', 0)
                            retweets = mention.get('retweets', 0)
                            comments = mention.get('comments', 0)
                            engagement = likes + retweets * 2 + comments * 3
                            engagement_scores.append(engagement)

                        if sentiments:
                            sentiment_trend = np.mean(sentiments)
                            avg_engagement = np.mean(engagement_scores) if engagement_scores else 0

                            # 综合热度指数
                            heat_index = (
                                min(mention_count / 100, 1.0) * 0.4 +  # 提及次数标准化
                                (sentiment_trend + 1) / 2 * 0.3 +      # 情绪趋势标准化
                                min(avg_engagement / 1000, 1.0) * 0.3  # 参与度标准化
                            )

                            heat_data.update({
                                'mention_count': mention_count,
                                'sentiment_trend': sentiment_trend,
                                'engagement_score': avg_engagement,
                                'heat_index': heat_index
                            })

        except Exception as e:
            logger.warning(f"获取{pair}社交热度数据失败: {e}")

        # 更新缓存
        with self.cache_lock:
            self.social_cache[cache_key] = heat_data

        return heat_data

    def _mock_news_api(self, coin_symbol: str) -> List[Dict]:
        """模拟新闻API返回数据"""
        # 实际使用时替换为真实API调用
        return [
            {
                'title': f'{coin_symbol} shows strong momentum',
                'content': f'Recent analysis suggests {coin_symbol} may continue upward trend',
                'timestamp': time.time() - 3600
            },
            {
                'title': f'{coin_symbol} partnership announcement',
                'content': f'{coin_symbol} announces strategic partnership',
                'timestamp': time.time() - 7200
            }
        ]

    def _mock_social_api(self, coin_symbol: str) -> Dict:
        """模拟社交媒体API返回数据"""
        # 实际使用时替换为真实API调用
        return {
            'mentions': [
                {
                    'text': f'{coin_symbol} looking bullish today!',
                    'likes': 25,
                    'retweets': 8,
                    'comments': 5,
                    'timestamp': time.time() - 1800
                },
                {
                    'text': f'Just bought more {coin_symbol}, expecting big moves',
                    'likes': 15,
                    'retweets': 3,
                    'comments': 2,
                    'timestamp': time.time() - 3600
                }
            ]
        }

    def _is_recent_mention(self, mention: Dict) -> bool:
        """检查提及是否为最近时间"""
        mention_time = mention.get('timestamp', 0)
        current_time = time.time()
        return current_time - mention_time < (self.news_lookback_hours.value * 3600)

    def _parallel_fetch_external_data(self, pairs: List[str]) -> Dict:
        """并行获取外部数据"""
        external_data = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交新闻情绪分析任务
            news_futures = {}
            social_futures = {}

            for pair in pairs:
                if self.enable_sentiment_analysis.value:
                    future = executor.submit(self._fetch_news_sentiment, pair)
                    news_futures[future] = pair

                if self.enable_social_analysis.value:
                    future = executor.submit(self._fetch_social_heat, pair)
                    social_futures[future] = pair

            # 收集新闻情绪结果
            for future in as_completed(news_futures):
                pair = news_futures[future]
                try:
                    result = future.result(timeout=10)
                    if pair not in external_data:
                        external_data[pair] = {}
                    external_data[pair]['news_sentiment'] = result
                except Exception as e:
                    logger.warning(f"获取{pair}新闻数据超时: {e}")
                    external_data[pair] = external_data.get(pair, {})
                    external_data[pair]['news_sentiment'] = {
                        'sentiment_score': 0.0, 'news_count': 0,
                        'positive_ratio': 0.0, 'negative_ratio': 0.0
                    }

            # 收集社交热度结果
            for future in as_completed(social_futures):
                pair = social_futures[future]
                try:
                    result = future.result(timeout=10)
                    if pair not in external_data:
                        external_data[pair] = {}
                    external_data[pair]['social_heat'] = result
                except Exception as e:
                    logger.warning(f"获取{pair}社交数据超时: {e}")
                    external_data[pair] = external_data.get(pair, {})
                    external_data[pair]['social_heat'] = {
                        'mention_count': 0, 'sentiment_trend': 0.0,
                        'engagement_score': 0.0, 'heat_index': 0.0
                    }

        return external_data

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """计算暴涨检测指标 - v1.1增强版"""

        # 获取多时间框架数据
        dataframe_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='15m')
        dataframe_4h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='4h')

        # 合并多时间框架数据
        dataframe = merge_informative_pair(dataframe, dataframe_15m, self.timeframe, '15m', ffill=True)
        dataframe = merge_informative_pair(dataframe, dataframe_4h, self.timeframe, '4h', ffill=True)

        # ==================== v1.0 基础指标 ====================

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

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = dataframe['atr'] / dataframe['close'] * 100

        # 成交量指标
        dataframe['vol_sma_20'] = dataframe['volume'].rolling(20).mean()
        dataframe['vol_sma_50'] = dataframe['volume'].rolling(50).mean()
        dataframe['vol_ratio_20'] = dataframe['volume'] / dataframe['vol_sma_20']
        dataframe['vol_ratio_50'] = dataframe['volume'] / dataframe['vol_sma_50']

        dataframe['vol_surge'] = (
            (dataframe['vol_ratio_20'] > self.volume_surge_threshold.value) |
            (dataframe['vol_ratio_50'] > self.volume_surge_threshold.value * 0.8)
        ).astype(int)

        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['obv_sma'] = dataframe['obv'].rolling(20).mean()
        dataframe['obv_trend'] = np.where(dataframe['obv'] > dataframe['obv_sma'], 1, 0)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)

        # 动量指标
        dataframe['roc_1h'] = ta.ROC(dataframe, timeperiod=1)
        dataframe['roc_4h'] = ta.ROC(dataframe, timeperiod=4)
        dataframe['roc_12h'] = ta.ROC(dataframe, timeperiod=12)

        dataframe['momentum_surge'] = (
            (dataframe['roc_1h'] > self.price_momentum_threshold.value) |
            (dataframe['roc_4h'] > self.price_momentum_threshold.value * 2)
        ).astype(int)

        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=14)
        stoch_k, stoch_d = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch_k
        dataframe['stoch_d'] = stoch_d

        # 突破检测
        dataframe['price_position_bb'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])

        dataframe['ema_breakout'] = (
            (dataframe['close'] > dataframe['ema_20']) &
            (dataframe['close'].shift(1) <= dataframe['ema_20'].shift(1))
        ).astype(int)

        dataframe['resistance_level'] = dataframe['high'].rolling(50).max()
        dataframe['resistance_breakout'] = (
            (dataframe['close'] > dataframe['resistance_level'].shift(1)) &
            (dataframe['volume'] > dataframe['vol_sma_20'])
        ).astype(int)

        # 反转信号
        dataframe['rsi_reversal'] = (
            (dataframe['rsi'] < 30) &
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
            (dataframe['rsi'] > self.rsi_recovery_threshold.value)
        ).astype(int)

        dataframe['macd_golden_cross'] = (
            (dataframe['macd'] > dataframe['macd_signal']) &
            (dataframe['macd'].shift(1) <= dataframe['macd_signal'].shift(1)) &
            (dataframe['macd_hist'] > 0)
        ).astype(int)

        dataframe['hammer_pattern'] = (
            (dataframe['close'] > dataframe['open']) &
            ((dataframe['low'] - dataframe[['open', 'close']].min(axis=1)) /
             (dataframe['high'] - dataframe['low']) > 0.6) &
            ((dataframe[['open', 'close']].max(axis=1) - dataframe['high']) /
             (dataframe['high'] - dataframe['low']) < 0.1)
        ).astype(int)

        # ==================== v1.1 新增指标 ====================

        # 获取外部数据 (新闻情绪 + 社交热度)
        try:
            current_pair = metadata['pair']
            external_data = self._parallel_fetch_external_data([current_pair])
            pair_data = external_data.get(current_pair, {})

            # 新闻情绪指标
            news_data = pair_data.get('news_sentiment', {})
            dataframe['news_sentiment_score'] = news_data.get('sentiment_score', 0.0)
            dataframe['news_count'] = news_data.get('news_count', 0)
            dataframe['news_positive_ratio'] = news_data.get('positive_ratio', 0.0)

            # 社交热度指标
            social_data = pair_data.get('social_heat', {})
            dataframe['social_mention_count'] = social_data.get('mention_count', 0)
            dataframe['social_sentiment_trend'] = social_data.get('sentiment_trend', 0.0)
            dataframe['social_heat_index'] = social_data.get('heat_index', 0.0)

        except Exception as e:
            logger.warning(f"获取{current_pair}外部数据失败: {e}")
            # 设置默认值
            dataframe['news_sentiment_score'] = 0.0
            dataframe['news_count'] = 0
            dataframe['news_positive_ratio'] = 0.0
            dataframe['social_mention_count'] = 0
            dataframe['social_sentiment_trend'] = 0.0
            dataframe['social_heat_index'] = 0.0

        # ==================== v1.1 增强评分系统 ====================

        # v1.0 基础评分
        dataframe['volume_score'] = (
            dataframe['vol_surge'] * 25 +
            (dataframe['vol_ratio_20'] > 2.0).astype(int) * 15 +
            dataframe['obv_trend'] * 10
        )

        dataframe['momentum_score'] = (
            dataframe['momentum_surge'] * 25 +
            (dataframe['roc_1h'] > 0.05).astype(int) * 15 +
            (dataframe['roc_4h'] > 0.10).astype(int) * 10
        )

        dataframe['technical_score'] = (
            dataframe['ema_breakout'] * 20 +
            dataframe['resistance_breakout'] * 25 +
            dataframe['macd_golden_cross'] * 15 +
            dataframe['rsi_reversal'] * 20
        )

        dataframe['pattern_score'] = (
            dataframe['hammer_pattern'] * 15 +
            (dataframe['price_position_bb'] < 0.2).astype(int) * 10 +
            (dataframe['willr'] > -20).astype(int) * 10
        )

        # v1.1 新增评分
        dataframe['sentiment_score'] = (
            (dataframe['news_sentiment_score'] > 0.1).astype(int) * 20 +
            (dataframe['news_positive_ratio'] > 0.6).astype(int) * 15 +
            (dataframe['news_count'] > 2).astype(int) * 10
        ) * self.sentiment_weight.value

        dataframe['social_score'] = (
            (dataframe['social_heat_index'] > 0.3).astype(int) * 25 +
            (dataframe['social_sentiment_trend'] > 0.1).astype(int) * 15 +
            (dataframe['social_mention_count'] > 10).astype(int) * 10
        ) * self.social_heat_weight.value

        # 综合评分 (v1.1)
        dataframe['explosion_score_v1'] = (
            dataframe['volume_score'] +
            dataframe['momentum_score'] +
            dataframe['technical_score'] +
            dataframe['pattern_score']
        )

        dataframe['explosion_score'] = (
            dataframe['explosion_score_v1'] +
            dataframe['sentiment_score'] +
            dataframe['social_score']
        )

        # 多时间框架确认 (增强版)
        dataframe['multi_timeframe_confirm'] = (
            (dataframe['rsi_15m'] < 70) &
            (dataframe['volume_15m'] > dataframe['volume_15m'].rolling(20).mean()) &
            (dataframe['close_4h'] > dataframe['close_4h'].shift(1)) &
            # v1.1 新增确认条件
            ((dataframe['news_sentiment_score'] >= 0) | (dataframe['news_count'] == 0)) &
            ((dataframe['social_sentiment_trend'] >= -0.2) | (dataframe['social_mention_count'] == 0))
        ).astype(int)

        # 最终信号强度 (v1.1)
        base_strength = dataframe['explosion_score'] * (1 + dataframe['multi_timeframe_confirm'] * 0.3)

        # 外部数据加成
        external_boost = 1.0
        if self.enable_sentiment_analysis.value:
            external_boost += dataframe['news_sentiment_score'].clip(0, 0.5) * 0.2
        if self.enable_social_analysis.value:
            external_boost += dataframe['social_heat_index'] * 0.15

        dataframe['signal_strength'] = base_strength * external_boost

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """暴涨预警信号 - v1.1增强版"""

        # 超高潜力信号 (90分以上，v1.1新增)
        dataframe.loc[
            (
                (dataframe['signal_strength'] >= 90) &
                (dataframe['volume'] > 0) &
                (dataframe['explosion_score'] >= 80) &
                (
                    (dataframe['news_sentiment_score'] > 0.2) |
                    (dataframe['social_heat_index'] > 0.4)
                )
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'ULTRA_HIGH_EXPLOSIVE_POTENTIAL')

        # 高潜力信号 (80-89分)
        dataframe.loc[
            (
                (dataframe['signal_strength'] >= 80) &
                (dataframe['signal_strength'] < 90) &
                (dataframe['volume'] > 0) &
                (dataframe['explosion_score'] >= 70)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'HIGH_EXPLOSIVE_POTENTIAL')

        # 中等潜力信号 (60-79分)
        dataframe.loc[
            (
                (dataframe['signal_strength'] >= 60) &
                (dataframe['signal_strength'] < 80) &
                (dataframe['volume'] > 0) &
                (dataframe['explosion_score'] >= 50)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'MEDIUM_EXPLOSIVE_POTENTIAL')

        # 早期预警信号 (45-59分)
        dataframe.loc[
            (
                (dataframe['signal_strength'] >= 45) &
                (dataframe['signal_strength'] < 60) &
                (dataframe['volume'] > 0) &
                (dataframe['vol_surge'] == 1)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'EARLY_WARNING')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """不设置退出信号，仅用于筛选"""
        return dataframe

    def custom_entry_price(self, pair: str, trade: Optional["Trade"],
                          current_time: datetime, proposed_rate: float,
                          entry_tag: Optional[str], side: str, **kwargs) -> float:
        """自定义入场价格（实际不交易）"""
        return proposed_rate

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                           proposed_stake: float, min_stake: Optional[float], max_stake: float,
                           leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """最小仓位（实际不交易）"""
        return min_stake if min_stake else 10.0

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                           time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                           side: str, **kwargs) -> bool:
        """发送增强版暴涨预警消息"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        if len(dataframe) > 0:
            latest = dataframe.iloc[-1]
            signal_strength = latest['signal_strength']
            explosion_score = latest['explosion_score']

            # 构建v1.1增强版预警消息
            message = f"""
🚀 暴涨预警 v1.1 - {pair} 🚀
📊 信号强度: {signal_strength:.1f}
💥 爆发评分: {explosion_score:.1f}
🏷️ 信号类型: {entry_tag}
💰 当前价格: {rate:.8f}

📈 技术指标:
├─ 1小时涨幅: {latest['roc_1h']:.2%}
├─ 成交量比率: {latest['vol_ratio_20']:.1f}x
└─ RSI: {latest['rsi']:.1f}

📰 外部信号:
├─ 新闻情绪: {latest['news_sentiment_score']:.2f} ({latest['news_count']:.0f}条)
├─ 社交热度: {latest['social_heat_index']:.2f}
└─ 社交情绪: {latest['social_sentiment_trend']:.2f}

⏰ {current_time.strftime('%Y-%m-%d %H:%M:%S')}
🔥 v1.1增强算法检测 - 建议关注24小时内价格变动！
            """

            # 发送消息
            self.dp.send_msg(message.strip())

            # 特殊信号额外提醒
            if entry_tag == 'ULTRA_HIGH_EXPLOSIVE_POTENTIAL':
                self.dp.send_msg(f"🔥🔥🔥 超高关注: {pair} 显示极强暴涨潜力！信号强度: {signal_strength:.1f} 🔥🔥🔥")
            elif entry_tag == 'HIGH_EXPLOSIVE_POTENTIAL':
                self.dp.send_msg(f"⚠️ 高度关注: {pair} 显示极强暴涨潜力！信号强度: {signal_strength:.1f}")

        # 不实际进行交易
        return False

    def custom_stoploss(self, pair: str, trade: "Trade", current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """不使用止损"""
        return -0.99