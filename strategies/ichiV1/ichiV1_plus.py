# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import pandas as pd  # noqa
import numpy as np
pd.options.mode.chained_assignment = None  # default='warn'
import technical.indicators as ftt
from functools import reduce
from datetime import datetime, timedelta
from freqtrade.strategy import merge_informative_pair
from freqtrade.strategy import stoploss_from_open


class ichiV1_plus(IStrategy):

    # NOTE: settings as of the 25th july 21
    # Buy hyperspace params:
    buy_params = {
        "buy_trend_above_senkou_level": 1,
        "buy_trend_bullish_level": 6,
        "buy_fan_magnitude_shift_value": 3,
        "buy_min_fan_magnitude_gain": 1.002 # NOTE: Good value (Win% ~70%), alot of trades
        #"buy_min_fan_magnitude_gain": 1.008 # NOTE: Very save value (Win% ~90%), only the biggest moves 1.008,
    }

    # Sell hyperspace params:
    # 增强的卖出参数配置
    sell_params = {
        # 基础趋势指标
        "sell_trend_indicator": "trend_close_2h",
        "sell_short_trend": "trend_close_5m",
        
        # 震荡市场过滤参数
        "adx_threshold": 25,  # ADX阈值，低于此值视为震荡市场
        "bb_width_percentile": 30,  # 布林带宽度百分位数阈值
        
        # 确认指标阈值
        "rsi_overbought": 70,  # RSI超买阈值
        "volume_confirmation": 1.2,  # 成交量确认倍数
        "trend_consistency_min": 0.3,  # 趋势一致性最小值
        
        # 分级卖出阈值
        "partial_sell_ratio": 0.4,  # 部分卖出比例
        "strong_sell_confirmation": 3,  # 强卖出信号确认数量
    }

    # ROI table:
    minimal_roi = {
        "0": 0.059,
        "10": 0.037,
        "41": 0.012,
        "114": 0
    }

    # Stoploss:
    stoploss = -0.275

    # Optimal timeframe for the strategy
    timeframe = '15m'

    startup_candle_count = 96
    process_only_new_candles = False

    trailing_stop = False
    #trailing_stop_positive = 0.002
    #trailing_stop_positive_offset = 0.025
    #trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    plot_config = {
        'main_plot': {
            # fill area between senkou_a and senkou_b
            'senkou_a': {
                'color': 'green', #optional
                'fill_to': 'senkou_b',
                'fill_label': 'Ichimoku Cloud', #optional
                'fill_color': 'rgba(255,76,46,0.2)', #optional
            },
            # plot senkou_b, too. Not only the area to it.
            'senkou_b': {},
            'trend_close_5m': {'color': '#FF5733'},
            'trend_close_15m': {'color': '#FF8333'},
            'trend_close_30m': {'color': '#FFB533'},
            'trend_close_1h': {'color': '#FFE633'},
            'trend_close_2h': {'color': '#E3FF33'},
            'trend_close_4h': {'color': '#C4FF33'},
            'trend_close_6h': {'color': '#61FF33'},
            'trend_close_8h': {'color': '#33FF7D'}
        },
        'subplots': {
            'fan_magnitude': {
                'fan_magnitude': {}
            },
            'fan_magnitude_gain': {
                'fan_magnitude_gain': {}
            }
        }
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['open'] = heikinashi['open']
        #dataframe['close'] = heikinashi['close']
        dataframe['high'] = heikinashi['high']
        dataframe['low'] = heikinashi['low']

        dataframe['trend_close_5m'] = dataframe['close']
        dataframe['trend_close_15m'] = ta.EMA(dataframe['close'], timeperiod=3)
        dataframe['trend_close_30m'] = ta.EMA(dataframe['close'], timeperiod=6)
        dataframe['trend_close_1h'] = ta.EMA(dataframe['close'], timeperiod=12)
        dataframe['trend_close_2h'] = ta.EMA(dataframe['close'], timeperiod=24)
        dataframe['trend_close_4h'] = ta.EMA(dataframe['close'], timeperiod=48)
        dataframe['trend_close_6h'] = ta.EMA(dataframe['close'], timeperiod=72)
        dataframe['trend_close_8h'] = ta.EMA(dataframe['close'], timeperiod=96)

        dataframe['trend_open_5m'] = dataframe['open']
        dataframe['trend_open_15m'] = ta.EMA(dataframe['open'], timeperiod=3)
        dataframe['trend_open_30m'] = ta.EMA(dataframe['open'], timeperiod=6)
        dataframe['trend_open_1h'] = ta.EMA(dataframe['open'], timeperiod=12)
        dataframe['trend_open_2h'] = ta.EMA(dataframe['open'], timeperiod=24)
        dataframe['trend_open_4h'] = ta.EMA(dataframe['open'], timeperiod=48)
        dataframe['trend_open_6h'] = ta.EMA(dataframe['open'], timeperiod=72)
        dataframe['trend_open_8h'] = ta.EMA(dataframe['open'], timeperiod=96)

        dataframe['fan_magnitude'] = (dataframe['trend_close_1h'] / dataframe['trend_close_8h'])
        dataframe['fan_magnitude_gain'] = dataframe['fan_magnitude'] / dataframe['fan_magnitude'].shift(1)

        # 震荡市场识别指标
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['atr'] = ta.ATR(dataframe)
        dataframe['atr_pct'] = (dataframe['atr'] / dataframe['close']) * 100
        
        # 布林带用于波动性分析
        bollinger = qtpylib.bollinger_bands(dataframe['close'], window=20, stds=2)
        dataframe['bb_upper'] = bollinger['upper']
        dataframe['bb_lower'] = bollinger['lower']
        dataframe['bb_width'] = ((dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['close']) * 100
        
        # 趋势一致性评分 (多时间框架趋势方向一致性)
        trend_directions = []
        timeframes = ['5m', '15m', '30m', '1h', '2h', '4h']
        for tf in timeframes:
            trend_col = f'trend_close_{tf}'
            if trend_col in dataframe.columns:
                trend_directions.append((dataframe[trend_col] > dataframe[trend_col].shift(1)).astype(int))
        
        if trend_directions:
            dataframe['trend_consistency'] = sum(trend_directions) / len(trend_directions)
        else:
            dataframe['trend_consistency'] = 0.5
            
        # RSI用于超买确认
        dataframe['rsi'] = ta.RSI(dataframe)
        
        # 成交量相关指标
        dataframe['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # 震荡市场标识 (ADX < 25 且 BB宽度较小)
        dataframe['is_ranging'] = (
            (dataframe['adx'] < 25) & 
            (dataframe['bb_width'] < dataframe['bb_width'].rolling(50).quantile(0.3))
        )

        ichimoku = ftt.ichimoku(dataframe, conversion_line_period=20, base_line_periods=60, laggin_span=120, displacement=30)
        dataframe['chikou_span'] = ichimoku['chikou_span']
        dataframe['tenkan_sen'] = ichimoku['tenkan_sen']
        dataframe['kijun_sen'] = ichimoku['kijun_sen']
        dataframe['senkou_a'] = ichimoku['senkou_span_a']
        dataframe['senkou_b'] = ichimoku['senkou_span_b']
        dataframe['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        dataframe['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        dataframe['cloud_green'] = ichimoku['cloud_green']
        dataframe['cloud_red'] = ichimoku['cloud_red']

        dataframe['atr'] = ta.ATR(dataframe)

        return dataframe


    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []

        # Trending market
        if self.buy_params['buy_trend_above_senkou_level'] >= 1:
            conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_5m'] > dataframe['senkou_b'])

        if self.buy_params['buy_trend_above_senkou_level'] >= 2:
            conditions.append(dataframe['trend_close_15m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_15m'] > dataframe['senkou_b'])

        if self.buy_params['buy_trend_above_senkou_level'] >= 3:
            conditions.append(dataframe['trend_close_30m'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_30m'] > dataframe['senkou_b'])

        if self.buy_params['buy_trend_above_senkou_level'] >= 4:
            conditions.append(dataframe['trend_close_1h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_1h'] > dataframe['senkou_b'])

        if self.buy_params['buy_trend_above_senkou_level'] >= 5:
            conditions.append(dataframe['trend_close_2h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_2h'] > dataframe['senkou_b'])

        if self.buy_params['buy_trend_above_senkou_level'] >= 6:
            conditions.append(dataframe['trend_close_4h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_4h'] > dataframe['senkou_b'])

        if self.buy_params['buy_trend_above_senkou_level'] >= 7:
            conditions.append(dataframe['trend_close_6h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_6h'] > dataframe['senkou_b'])

        if self.buy_params['buy_trend_above_senkou_level'] >= 8:
            conditions.append(dataframe['trend_close_8h'] > dataframe['senkou_a'])
            conditions.append(dataframe['trend_close_8h'] > dataframe['senkou_b'])

        # Trends bullish
        if self.buy_params['buy_trend_bullish_level'] >= 1:
            conditions.append(dataframe['trend_close_5m'] > dataframe['trend_open_5m'])

        if self.buy_params['buy_trend_bullish_level'] >= 2:
            conditions.append(dataframe['trend_close_15m'] > dataframe['trend_open_15m'])

        if self.buy_params['buy_trend_bullish_level'] >= 3:
            conditions.append(dataframe['trend_close_30m'] > dataframe['trend_open_30m'])

        if self.buy_params['buy_trend_bullish_level'] >= 4:
            conditions.append(dataframe['trend_close_1h'] > dataframe['trend_open_1h'])

        if self.buy_params['buy_trend_bullish_level'] >= 5:
            conditions.append(dataframe['trend_close_2h'] > dataframe['trend_open_2h'])

        if self.buy_params['buy_trend_bullish_level'] >= 6:
            conditions.append(dataframe['trend_close_4h'] > dataframe['trend_open_4h'])

        if self.buy_params['buy_trend_bullish_level'] >= 7:
            conditions.append(dataframe['trend_close_6h'] > dataframe['trend_open_6h'])

        if self.buy_params['buy_trend_bullish_level'] >= 8:
            conditions.append(dataframe['trend_close_8h'] > dataframe['trend_open_8h'])

        # Trends magnitude
        conditions.append(dataframe['fan_magnitude_gain'] >= self.buy_params['buy_min_fan_magnitude_gain'])
        conditions.append(dataframe['fan_magnitude'] > 1)

        for x in range(self.buy_params['buy_fan_magnitude_shift_value']):
            conditions.append(dataframe['fan_magnitude'].shift(x+1) < dataframe['fan_magnitude'])

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe


    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # 初始化卖出信号列
        dataframe['sell'] = 0.0

        # ============ 基础趋势穿越条件 ============
        basic_sell_signal = qtpylib.crossed_below(
            dataframe[self.sell_params['sell_short_trend']], 
            dataframe[self.sell_params['sell_trend_indicator']]
        )

        # ============ 确认指标收集 ============
        confirmations = []

        # 1. RSI超买确认
        rsi_confirmation = dataframe['rsi'] > self.sell_params['rsi_overbought']
        confirmations.append(rsi_confirmation)

        # 2. 成交量确认（放量下跌）
        volume_confirmation = dataframe['volume_ratio'] > self.sell_params['volume_confirmation']
        confirmations.append(volume_confirmation)

        # 3. 一目均衡表确认（价格跌破转换线）
        ichimoku_confirmation = dataframe['close'] < dataframe['tenkan_sen']
        confirmations.append(ichimoku_confirmation)

        # 4. 趋势一致性恶化确认
        trend_deterioration = dataframe['trend_consistency'] < self.sell_params['trend_consistency_min']
        confirmations.append(trend_deterioration)

        # 5. 云图跌破确认
        cloud_break = (dataframe['close'] < dataframe['senkou_a']) & (dataframe['close'] < dataframe['senkou_b'])
        confirmations.append(cloud_break)

        # 计算确认信号数量
        confirmation_count = sum([conf.astype(int) for conf in confirmations])

        # ============ 震荡市场保护机制 ============
        # 在震荡市场中提高卖出门槛，减少频繁交易
        ranging_market = dataframe['is_ranging']

        # ============ 分级卖出逻辑 ============

        # 部分卖出条件（震荡市场中只进行部分卖出）
        partial_sell_conditions = (
            basic_sell_signal &
            (confirmation_count >= 1) &
            ranging_market &
            (dataframe['adx'] < self.sell_params['adx_threshold'])
        )

        # 强势卖出条件（趋势市场或多重确认）
        strong_sell_conditions = (
            basic_sell_signal &
            (
                # 趋势市场中的确认卖出
                ((~ranging_market) & (confirmation_count >= 2)) |
                # 或者多重确认的强势卖出
                (confirmation_count >= self.sell_params['strong_sell_confirmation'])
            )
        )

        # 紧急卖出条件（多重负面信号同时出现）
        emergency_sell_conditions = (
            basic_sell_signal &
            (confirmation_count >= 4) &
            (dataframe['rsi'] > 75) &  # 严重超买
            cloud_break &
            (dataframe['close'] < dataframe['bb_lower'])  # 跌破布林带下轨
        )

        # ============ 应用卖出信号 ============

        # 部分卖出（40%仓位）
        dataframe.loc[partial_sell_conditions, 'sell'] = self.sell_params['partial_sell_ratio']

        # 强势卖出（70%仓位）
        dataframe.loc[strong_sell_conditions, 'sell'] = 0.7

        # 紧急全部卖出（100%仓位）
        dataframe.loc[emergency_sell_conditions, 'sell'] = 1.0

        # ============ 额外的市场环境适应性调整 ============

        # 如果扇形幅度急剧恶化，增强卖出信号
        fan_deterioration = (
            (dataframe['fan_magnitude'] < 0.98) &  # 短期趋势弱于长期趋势
            (dataframe['fan_magnitude_gain'] < 0.995)  # 且持续恶化
        )

        # 扇形恶化时的额外卖出
        fan_sell_conditions = basic_sell_signal & fan_deterioration & (confirmation_count >= 1)
        dataframe.loc[fan_sell_conditions, 'sell'] = np.maximum(
            dataframe['sell'], 0.6  # 至少卖出60%
        )

        return dataframe
