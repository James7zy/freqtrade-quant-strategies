import logging
import numpy as np
import pandas as pd
import talib.abstract as ta
import pandas_ta as pta
from datetime import datetime, timedelta
from pandas import DataFrame
from typing import Optional
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair, DecimalParameter, IntParameter, BooleanParameter
import freqtrade.vendor.qtpylib.indicators as qtpylib

logger = logging.getLogger(__name__)

class ExplosiveDetector_v1(IStrategy):
    """
    24小时暴涨币种检测策略
    专门用于筛选和预警可能在24小时内出现暴涨的币种
    """

    # 基础配置
    timeframe = '1h'  # 主时间框架：1小时
    startup_candle_count: int = 200
    stoploss = -0.99  # 不实际交易，仅用于筛选
    can_short = False

    # 不进行实际交易，仅用于筛选
    process_only_new_candles = True
    use_exit_signal = False
    ignore_roi_if_entry_signal = True

    # ROI设置（不实际使用）
    minimal_roi = {"0": 10.0}

    # 检测参数（可优化）
    volume_surge_threshold = DecimalParameter(2.0, 5.0, default=3.0, space="buy", optimize=True)
    price_momentum_threshold = DecimalParameter(0.05, 0.20, default=0.10, space="buy", optimize=True)
    rsi_recovery_threshold = DecimalParameter(25.0, 45.0, default=35.0, space="buy", optimize=True)
    breakout_strength_threshold = DecimalParameter(1.5, 3.0, default=2.0, space="buy", optimize=True)

    # 时间窗口参数
    volume_lookback = IntParameter(6, 24, default=12, space="buy", optimize=True)
    momentum_lookback = IntParameter(3, 12, default=6, space="buy", optimize=True)

    def informative_pairs(self):
        """多时间框架数据"""
        pairs = self.dp.current_whitelist()
        informative_pairs = []
        for pair in pairs:
            informative_pairs.append((pair, '15m'))  # 15分钟用于短期确认
            informative_pairs.append((pair, '4h'))   # 4小时用于趋势确认
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """计算暴涨检测指标"""

        # 获取多时间框架数据
        dataframe_15m = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='15m')
        dataframe_4h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='4h')

        # 合并多时间框架数据
        dataframe = merge_informative_pair(dataframe, dataframe_15m, self.timeframe, '15m', ffill=True)
        dataframe = merge_informative_pair(dataframe, dataframe_4h, self.timeframe, '4h', ffill=True)

        # ==================== 基础技术指标 ====================

        # 移动平均线
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_15m'] = dataframe['rsi_15m']  # 15分钟RSI

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

        # ATR (平均真实波幅)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = dataframe['atr'] / dataframe['close'] * 100

        # ==================== 成交量指标 ====================

        # 成交量移动平均
        dataframe['vol_sma_20'] = dataframe['volume'].rolling(20).mean()
        dataframe['vol_sma_50'] = dataframe['volume'].rolling(50).mean()

        # 成交量比率
        dataframe['vol_ratio_20'] = dataframe['volume'] / dataframe['vol_sma_20']
        dataframe['vol_ratio_50'] = dataframe['volume'] / dataframe['vol_sma_50']

        # 成交量突增检测
        dataframe['vol_surge'] = (
            (dataframe['vol_ratio_20'] > self.volume_surge_threshold.value) |
            (dataframe['vol_ratio_50'] > self.volume_surge_threshold.value * 0.8)
        ).astype(int)

        # OBV (能量潮)
        dataframe['obv'] = ta.OBV(dataframe)
        dataframe['obv_sma'] = dataframe['obv'].rolling(20).mean()
        dataframe['obv_trend'] = np.where(dataframe['obv'] > dataframe['obv_sma'], 1, 0)

        # MFI (资金流量指数)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)

        # ==================== 动量指标 ====================

        # ROC (变化率)
        dataframe['roc_1h'] = ta.ROC(dataframe, timeperiod=1)
        dataframe['roc_4h'] = ta.ROC(dataframe, timeperiod=4)
        dataframe['roc_12h'] = ta.ROC(dataframe, timeperiod=12)

        # 动量突破检测
        dataframe['momentum_surge'] = (
            (dataframe['roc_1h'] > self.price_momentum_threshold.value) |
            (dataframe['roc_4h'] > self.price_momentum_threshold.value * 2)
        ).astype(int)

        # Williams %R
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=14)

        # Stochastic
        stoch_k, stoch_d = ta.STOCH(dataframe)
        dataframe['stoch_k'] = stoch_k
        dataframe['stoch_d'] = stoch_d

        # ==================== 突破检测 ====================

        # 价格位置
        dataframe['price_position_bb'] = (dataframe['close'] - dataframe['bb_lower']) / (dataframe['bb_upper'] - dataframe['bb_lower'])

        # EMA突破
        dataframe['ema_breakout'] = (
            (dataframe['close'] > dataframe['ema_20']) &
            (dataframe['close'].shift(1) <= dataframe['ema_20'].shift(1))
        ).astype(int)

        # 阻力位突破检测
        dataframe['resistance_level'] = dataframe['high'].rolling(50).max()
        dataframe['resistance_breakout'] = (
            (dataframe['close'] > dataframe['resistance_level'].shift(1)) &
            (dataframe['volume'] > dataframe['vol_sma_20'])
        ).astype(int)

        # ==================== 反转信号检测 ====================

        # RSI反转
        dataframe['rsi_reversal'] = (
            (dataframe['rsi'] < 30) &  # 之前超卖
            (dataframe['rsi'] > dataframe['rsi'].shift(1)) &  # 开始上升
            (dataframe['rsi'] > self.rsi_recovery_threshold.value)  # 超过恢复阈值
        ).astype(int)

        # MACD金叉
        dataframe['macd_golden_cross'] = (
            (dataframe['macd'] > dataframe['macd_signal']) &
            (dataframe['macd'].shift(1) <= dataframe['macd_signal'].shift(1)) &
            (dataframe['macd_hist'] > 0)
        ).astype(int)

        # 锤子线形态检测
        dataframe['hammer_pattern'] = (
            (dataframe['close'] > dataframe['open']) &  # 阳线
            ((dataframe['low'] - dataframe[['open', 'close']].min(axis=1)) /
             (dataframe['high'] - dataframe['low']) > 0.6) &  # 长下影线
            ((dataframe[['open', 'close']].max(axis=1) - dataframe['high']) /
             (dataframe['high'] - dataframe['low']) < 0.1)  # 无上影线或很短
        ).astype(int)

        # ==================== 综合爆发力评分 ====================

        # 计算各项指标得分
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
            (dataframe['price_position_bb'] < 0.2).astype(int) * 10 +  # 布林带下轨附近
            (dataframe['willr'] > -20).astype(int) * 10  # Williams %R 超买区域
        )

        # 总评分 (0-100)
        dataframe['explosion_score'] = (
            dataframe['volume_score'] +
            dataframe['momentum_score'] +
            dataframe['technical_score'] +
            dataframe['pattern_score']
        )

        # 多时间框架确认
        dataframe['multi_timeframe_confirm'] = (
            (dataframe['rsi_15m'] < 70) &  # 15分钟不超买
            (dataframe['volume_15m'] > dataframe['volume_15m'].rolling(20).mean()) &  # 15分钟成交量放大
            (dataframe['close_4h'] > dataframe['close_4h'].shift(1))  # 4小时上升趋势
        ).astype(int)

        # 最终信号强度
        dataframe['signal_strength'] = dataframe['explosion_score'] * (1 + dataframe['multi_timeframe_confirm'] * 0.3)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """暴涨预警信号"""

        # 高潜力暴涨信号 (80分以上)
        dataframe.loc[
            (
                (dataframe['signal_strength'] >= 80) &
                (dataframe['volume'] > 0) &
                (dataframe['explosion_score'] >= 70)
            ),
            ['enter_long', 'enter_tag']
        ] = (1, 'HIGH_EXPLOSIVE_POTENTIAL')

        # 中等潜力暴涨信号 (60-79分)
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
        """发送暴涨预警消息"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        if len(dataframe) > 0:
            latest = dataframe.iloc[-1]
            signal_strength = latest['signal_strength']
            explosion_score = latest['explosion_score']

            # 构建详细的预警消息
            message = f"""
🚀 暴涨预警 - {pair} 🚀
📊 信号强度: {signal_strength:.1f}
💥 爆发评分: {explosion_score:.1f}
🏷️ 信号类型: {entry_tag}
💰 当前价格: {rate:.8f}
📈 1小时涨幅: {latest['roc_1h']:.2%}
📊 成交量比率: {latest['vol_ratio_20']:.1f}x
📉 RSI: {latest['rsi']:.1f}
⏰ 时间: {current_time.strftime('%Y-%m-%d %H:%M:%S')}

🔥 建议关注24小时内价格变动！
            """

            # 发送消息
            self.dp.send_msg(message.strip())

            # 高潜力币种额外提醒
            if entry_tag == 'HIGH_EXPLOSIVE_POTENTIAL':
                self.dp.send_msg(f"⚠️ 高度关注: {pair} 显示极强暴涨潜力！信号强度: {signal_strength:.1f}")

        # 不实际进行交易
        return False

    def custom_stoploss(self, pair: str, trade: "Trade", current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """不使用止损"""
        return -0.99