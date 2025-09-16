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
from datetime import datetime


class ichiV1_plus(IStrategy):
    #can_short = True
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
    #minimal_roi = {
    #    "0": 0.059,
    #    "10": 0.037,
    #    "41": 0.012,
    #    "115": 0
    #}

    # ============= 新 ROI 阶梯（快速锁定型） =============
    # 设计依据：你实盘的中位持仓 ~30min；>60min 后收益率快速衰减。
    # 策略：前 1 小时逐步降低要求，180~240 分钟后基本撤退，防止资金囚禁。
    # 可在后续根据波动性（ATR%）动态切换不同模板（fast/balanced/trend）。
    minimal_roi = {
        "0":   0.030,   # 首阶段：抓动量
        "15":  0.024,   # 15 分钟后没到 3% -> 降一点防回吐
        "30":  0.018,   # 进入震荡保护
        "60":  0.012,   # 1 小时后还能给 1.2%
        "90":  0.008,
        "120": 0.005,
        "180": 0.003,   # 3 小时后只要覆盖手续费即可
        "240": 0.000    # 4 小时仍不行 -> 保本退出
    }

    # ============= 止损配置 =============
    # 原 -25.5% 过深，放任尾部风险；改为较紧的基础止损。
    # 建议：核心固定止损 + 结构/时间/动态 ATR 收紧（后续可加 custom_stoploss）。
    stoploss = -0.12  # 基础硬止损（可回测 -0.10/-0.12/-0.14 三档择优）

    # 预留：动态止损调节参数（可在 custom_stoploss 中引用）
    dyn_stop_params = {
        'atr_period': 14,
        'atr_mult_initial': 3.0,      # 初始宽（进入后 0~15min）
        'atr_mult_trend': 2.2,        # 当趋势一致性高时（trend_consistency>0.7）
        'atr_mult_decay': 1.8,        # 持仓 >90min 收紧
        'time_tighten_min': 60,       # 60 分钟后开始考虑收紧
        'min_stop': -0.055            # 动态收紧的底线（避免过度紧）
    }

    # Optimal timeframe for the strategy
    timeframe = '15m'

    startup_candle_count = 96
    process_only_new_candles = False

    # ============= 追踪止盈优化 =============
    # 调整：降低启动门槛 offset（3% -> 2.2%），细化正向锁定（1% -> 0.9%）。
    # 目的：让 1.5%~2.8% 的常见强势波段不全部回吐。
    trailing_stop = True
    trailing_stop_positive = 0.009             # 启用后允许最大回撤 0.9%
    trailing_stop_positive_offset = 0.022      # 先达到 2.2% 才启用（防止震荡提前触发）
    trailing_only_offset_is_reached = True

    # 扩展：可在 custom_exit / custom_stoploss 中实现“分批止盈 + 回撤强化”
    partial_exit_params = {
        'enable': True,
        'first_take_profit': 0.025,   # 浮盈 ≥2.5% 触发第一次部分减仓
        'first_pct': 0.5,             # 减仓 50%
        'second_take_profit': 0.05,   # 剩余仓位如果继续拉升到 5%
        'second_trail_offset': 0.015  # 第二阶段更紧追踪
    }

    # 超时退出：超过 X 分钟仍未达到最低阈值（如 <0.4%）主动退出释放资金
    timeout_exit_params = {
        'enable': True,
        'check_min': 90,      # 90 分钟
        'profit_floor': 0.004 # 若 <0.4% 且无趋势改善信号则退出
    }

    # 峰值回撤退出：用于在获得一定利润后，价格出现较深回撤时锁定收益
    drawdown_exit_params = {
        'enable': True,
        'min_profit': 0.03,     # 仅当曾经浮盈 ≥3% 时才启动回撤监控
        'drawdown_pct': 0.015   # 从峰值回撤 ≥1.5%（绝对利润值）则触发退出
    }

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    # 启用自定义动态止损逻辑（custom_stoploss 方法才会被调用）
    use_custom_stoploss = True
    # 启用仓位调整（DCA / 部分减仓 等功能需要）
    position_adjustment_enable = True

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

    # 固定杠杆模式：直接使用常量倍数
    fixed_leverage: float = 2.0

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

    # =============================================================
    # 可选增强：custom_exit 钩子（需 freqtrade 支持版本）。
    # 实现思路：
    # 1. 如果 partial_exit_params.enable:
    #       - 检查当前浮盈 profit_ratio >= first_take_profit 且 还未记录第一次减仓 -> 返回部分卖出 (通过 'sell' 标签 或使用 custom_exit_info)
    #       - 第二次同理；可将 trailing_stop_positive 动态下调。
    # 2. 如果 timeout_exit_params.enable:
    #       - 持仓分钟数 > check_min 且 profit_ratio < profit_floor 且 trend_consistency < 阈值 -> 直接给出 exit。
    # 3. 动态 ATR 止损：根据 dyn_stop_params 计算 ATR * 对应倍数，若当前跌破 open_rate*(1-动态止损) 则退出。
    # 下面仅放置占位，不直接启用，以免与现有卖出逻辑冲突；需要启用时取消注释并结合策略回测调整。
    # -------------------------------------------------------------
    # def custom_exit(self, pair: str, trade, current_time: datetime, current_rate: float,
    #                 current_profit: float, **kwargs):
    #     # 示例：超时退出
    #     if self.timeout_exit_params['enable']:
    #         age_min = (current_time - trade.open_date_utc).total_seconds() / 60
    #         if age_min > self.timeout_exit_params['check_min'] and \
    #            current_profit < self.timeout_exit_params['profit_floor']:
    #             return ("timeout_exit", "time_based")
    #     # 示例：第一次部分获利（需检查 trade.nr_of_successful_exits 等属性 / position size）
    #     # if self.partial_exit_params['enable'] and current_profit >= self.partial_exit_params['first_take_profit']:
    #     #     return ("partial_1", "part_take")
    #     return None

    # =============================================================
    # 动态止损: custom_stoploss
    # 逻辑层次：
    # 1) 基础硬止损 self.stoploss (-12%) 是底线
    # 2) 根据持仓时间与趋势一致性(trend_consistency)逐步收紧
    # 3) ATR * 不同倍数提供上限（更紧的止损限制）
    # 4) 当浮盈超过一定阈值，锁定一部分利润（抬高止损）
    # 返回值：距离开仓价的负比例(例如 -0.05)
    # 注意：需 freqtrade 配置中启用 use_custom_stoploss
    # =============================================================
    def custom_stoploss(self, pair: str, trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs):
        try:
            # 若 indicators 不可用（启动初期）直接返回原止损
            if 'trend_consistency' not in trade.open_trade_price:
                pass
        except Exception:
            pass

        # 计算持仓分钟
        age_min = (current_time - trade.open_date_utc).total_seconds() / 60

        # 基础最大允许损失
        base_stop = self.stoploss  # 例如 -0.12

        # 动态 ATR 估算：使用最近的 dataframe（freqtrade 会在 kwargs 里传递）
        dataframe = kwargs.get('dataframe')
        dyn_stop = None
        if dataframe is not None and len(dataframe) > 0 and 'atr' in dataframe.columns:
            atr = float(dataframe['atr'].iloc[-1])
            last_close = float(dataframe['close'].iloc[-1])
            atr_pct = atr / last_close if last_close else 0
            # 选取倍数
            mult = self.dyn_stop_params['atr_mult_initial']
            # 趋势一致性（使用最近行）
            trend_consistency = dataframe['trend_consistency'].iloc[-1] if 'trend_consistency' in dataframe.columns else 0.5
            if trend_consistency > 0.7:
                mult = self.dyn_stop_params['atr_mult_trend']
            if age_min > self.dyn_stop_params['time_tighten_min']:
                mult = min(mult, self.dyn_stop_params['atr_mult_decay'])
            dyn_stop_level = - atr_pct * mult
            # 限制不超过 min_stop（例如 -5.5%）
            dyn_stop = max(dyn_stop_level, self.dyn_stop_params['min_stop'])

        # 时间收紧：随着时间推移，最大亏损容忍下降
        time_stop = base_stop
        if age_min > 180:
            time_stop = max(time_stop, -0.06)
        elif age_min > 120:
            time_stop = max(time_stop, -0.075)
        elif age_min > 60:
            time_stop = max(time_stop, -0.09)

        # 浮盈保护：当已获得一定利润，抬高止损（盈转亏保护）
        profit_protect_stop = None
        if current_profit > 0.05:       # >5%
            profit_protect_stop = current_profit * 0.4 * -1  # 保留 60% 浮盈
        elif current_profit > 0.03:     # >3%
            profit_protect_stop = -0.015
        elif current_profit > 0.02:     # >2%
            profit_protect_stop = -0.02

        # 汇总候选止损（取“最不允许亏得多”的，即较大的那个）
        candidates = [base_stop]
        if dyn_stop is not None:
            candidates.append(dyn_stop)
        candidates.append(time_stop)
        if profit_protect_stop is not None:
            candidates.append(profit_protect_stop)

        final_stop = max(candidates)  # 例如在 -0.12 与 -0.055 之间取 -0.055

        # 如果已经触及盈利阈值并且 trailing 已启动，可再略收紧
        if current_profit > 0.06:
            final_stop = max(final_stop, -0.025)

        return final_stop

    # =============================================================
    # 自定义退出：超时 + 部分减仓
    # 注意：部分减仓功能需 freqtrade 版本支持 position adjustments。
    # 如果你的版本不支持 partial exits，你可以只返回一次性退出。
    # 返回格式: (exit_reason, tag) 或 None
    # =============================================================
    def custom_exit(self, pair: str, trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        # 记录并更新峰值浮盈，用于回撤退出逻辑
        if not hasattr(trade, 'user_data') or trade.user_data is None:
            trade.user_data = {}
        peak_profit = trade.user_data.get('peak_profit', current_profit)
        if current_profit > peak_profit:
            peak_profit = current_profit
            trade.user_data['peak_profit'] = peak_profit

        # 超时退出
        if self.timeout_exit_params['enable']:
            age_min = (current_time - trade.open_date_utc).total_seconds() / 60
            if age_min > self.timeout_exit_params['check_min'] and \
               current_profit < self.timeout_exit_params['profit_floor']:
                return ("timeout_exit", "time_based")

        # 峰值回撤退出：当盈利达到阈值后出现显著回撤
        if self.drawdown_exit_params['enable']:
            if peak_profit >= self.drawdown_exit_params['min_profit'] and \
               (peak_profit - current_profit) >= self.drawdown_exit_params['drawdown_pct'] and \
               current_profit > 0:  # 仍是盈利区间才锁定
                return ("drawdown_exit", "peak_retrace")

        # 部分减仓逻辑（示例）
        if self.partial_exit_params['enable']:
            # 使用 trade.user_data 记录阶段
            stage = None
            if hasattr(trade, 'user_data') and isinstance(trade.user_data, dict):
                stage = trade.user_data.get('partial_stage')
            else:
                trade.user_data = {}

            # 第一次部分减仓
            if current_profit >= self.partial_exit_params['first_take_profit'] and stage is None:
                trade.user_data['partial_stage'] = 1
                # 返回一个标签 - freqtrade 将按策略卖出(需要在配置中允许部分平仓)
                return ("partial_1", f"part_{self.partial_exit_params['first_pct']}")

            # 第二次部分减仓（更高目标）
            if current_profit >= self.partial_exit_params['second_take_profit'] and stage == 1:
                trade.user_data['partial_stage'] = 2
                return ("partial_2", "trail_tight")

        return None

    # =============================================================
    # 固定杠杆：仅返回设定或配置覆盖的 fixed_leverage
    # -------------------------------------------------------------
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str, side: str, **kwargs) -> float:
        if hasattr(self, 'config'):
            sp = self.config.get('strategy_parameters', {}) or {}
            cfg_val = sp.get('fixed_leverage')
            if cfg_val is not None:
                try:
                    self.fixed_leverage = float(cfg_val)
                except Exception:
                    pass
        return float(max(1.0, min(self.fixed_leverage, max_leverage)))

    # =============================================================
    # 仓位调整：用于实现部分减仓 (partial take profit)
    # 返回正数 -> 增加仓位 (DCA)；负数 -> 减仓；None -> 不调整
    # =============================================================
    def adjust_trade_position(self, trade, current_time: datetime, current_rate: float,
                               current_profit: float, min_stake: float, max_stake: float,
                               current_entry_rate: float, current_exit_rate: float,
                               current_entry_profit: float, current_exit_profit: float, **kwargs):
        if not self.partial_exit_params['enable']:
            return None

        # 初始化 user_data
        if not hasattr(trade, 'user_data') or trade.user_data is None:
            trade.user_data = {}
        stage = trade.user_data.get('partial_stage')

        # 当前持仓数量（以 amount 计算）
        current_amount = trade.amount
        if current_amount is None or current_amount <= 0:
            return None

        # 已经执行的部分退出次数（freqtrade 记录成功退出订单数）
    # exits_done = trade.nr_of_successful_exits if hasattr(trade, 'nr_of_successful_exits') else 0

        # 第一阶段部分减仓
        if stage is None and current_profit >= self.partial_exit_params['first_take_profit']:
            reduce_amt = current_amount * self.partial_exit_params['first_pct']
            trade.user_data['partial_stage'] = 1
            trade.user_data['peak_profit'] = max(trade.user_data.get('peak_profit', current_profit), current_profit)
            # 返回负数表示减少仓位
            return -reduce_amt

        # 第二阶段：达到第二目标 -> 清仓（或可选择再留一小部分）
        if stage == 1 and current_profit >= self.partial_exit_params['second_take_profit']:
            # 这里选择全部卖出剩余仓位，你也可以改成只再卖出一半
            trade.user_data['partial_stage'] = 2
            trade.user_data['peak_profit'] = max(trade.user_data.get('peak_profit', current_profit), current_profit)
            return -current_amount  # 剩余全平

        return None
