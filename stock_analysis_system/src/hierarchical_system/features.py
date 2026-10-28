from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


LAYER_FEATURES = {
    "micro": [
        "close", "return_1d", "return_3d", "volatility_5d", "price_volatility",
        "volume_ratio_5", "breakout_5d", "amplitude", "turnover_rate", "turnover_change",
        "money_flow_proxy", "upper_shadow_ratio", "lower_shadow_ratio", "close_to_high", "close_to_low",
    ],
    "small": [
        "close", "return_1d", "return_5d", "return_10d", "volume_ma5_ma20", "amount_change_rate",
        "amount_ma5_ma20", "relative_strength_proxy", "breakout_10d", "volatility_10d",
        "trend_strength", "turnover_rate", "turnover_change", "money_flow_proxy", "amplitude",
        "close_to_high", "close_to_low", "gap_open", "price_acceleration", "volume_acceleration",
    ],
    "mid": [
        "close", "sma_5", "sma_20", "sma_ratio", "ema_12", "ema_26", "macd", "macd_signal",
        "rsi_14", "cci_14", "atr_14", "turnover_rate", "price_amplitude", "volatility_20d",
        "amount_ma5_ma20", "volume_ma5_ma20", "return_5d", "return_20d", "obv", "obv_change",
        "money_flow_proxy", "close_to_high", "close_to_low", "gap_open", "trend_strength",
    ],
    "large": [
        "close", "return_1d", "return_5d", "return_20d", "market_cap_ratio", "amount_market_cap_ratio",
        "stability_inverse", "volume_concentration", "volatility_20d", "volatility_60d", "beta_proxy",
        "amount_ma5_ma20", "volume_ma5_ma20", "turnover_rate", "turnover_change", "atr_14",
        "rsi_14", "macd", "macd_signal", "trend_strength",
    ],
}


@dataclass(frozen=True)
class LayerBound:
    min_yi: float | None
    max_yi: float | None


LAYER_BOUNDS = {
    "micro": LayerBound(None, 50),
    "small": LayerBound(50, 200),
    "mid": LayerBound(200, 1000),
    "large": LayerBound(1000, None),
}


def layer_from_market_cap_yi(total_mv_yi: float) -> str:
    if total_mv_yi < 50:
        return "micro"
    if total_mv_yi < 200:
        return "small"
    if total_mv_yi < 1000:
        return "mid"
    return "large"


def _safe_div(a: pd.Series | float, b: pd.Series | float) -> pd.Series | float:
    if isinstance(b, pd.Series):
        safe_b = b.replace(0, np.nan)
    else:
        safe_b = np.nan if b == 0 else b
    return a / safe_b


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _cci(df: pd.DataFrame, window: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(window).mean()
    md = (tp - ma).abs().rolling(window).mean()
    return (tp - ma) / (0.015 * md.replace(0, np.nan))


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["volume"]).cumsum()


def _ffill_pivot_values(values: pd.Series, flags: pd.Series) -> pd.Series:
    pivot_values = values.where(flags.astype(bool))
    return pivot_values.ffill()


def _build_feature_frame(index: pd.Index, columns: dict[str, pd.Series | float | str]) -> pd.DataFrame:
    normalized: dict[str, pd.Series] = {}
    for name, value in columns.items():
        if isinstance(value, pd.Series):
            normalized[name] = value
        else:
            normalized[name] = pd.Series(value, index=index)
    return pd.DataFrame(normalized, index=index)


def build_layer_features(price_df: pd.DataFrame, market_cap_yi: float) -> pd.DataFrame:
    df = price_df.copy().sort_values("date").reset_index(drop=True)

    def _numeric_series_or_default(column_name: str, default_value: float = 0.0) -> pd.Series:
        if column_name in df.columns:
            return pd.to_numeric(df[column_name], errors="coerce").fillna(default_value)
        return pd.Series(default_value, index=df.index, dtype=float)

    def _text_value_or_default(column_name: str, default_value: str = "") -> str:
        if column_name not in df.columns or df[column_name].isna().all():
            return default_value
        return str(df[column_name].dropna().iloc[0]).strip()

    def _scaled_log_feature(series: pd.Series | float, clip_upper: float) -> pd.Series:
        if not isinstance(series, pd.Series):
            series = pd.Series(series, index=df.index, dtype=float)
        clipped = pd.to_numeric(series, errors="coerce").clip(lower=0, upper=clip_upper).fillna(0.0)
        return np.log1p(clipped) / np.log1p(clip_upper)

    industry_name = _text_value_or_default("行业")
    market_name = _text_value_or_default("市场")
    listing_date_raw = _text_value_or_default("上市日期")
    listing_date = pd.to_datetime(listing_date_raw, errors="coerce") if listing_date_raw else pd.NaT

    tech_keywords = ("电子", "半导体", "通信", "计算机", "软件", "互联网", "元器件", "IT", "芯片", "人工智能")
    consumer_keywords = ("食品", "饮料", "家电", "零售", "商贸", "服饰", "旅游", "酒店", "美容", "乳业", "白酒", "消费")
    medicine_keywords = ("医药", "医疗", "生物", "制药", "器械")
    finance_keywords = ("银行", "证券", "保险", "金融", "信托")
    cycle_keywords = ("钢铁", "有色", "煤炭", "化工", "建材", "地产", "机械", "石油", "天然气", "航运", "交运")
    layer = layer_from_market_cap_yi(market_cap_yi)
    amount = df.get("amount", df["close"] * df["volume"])
    return_1d = df["close"].pct_change()
    return_3d = df["close"].pct_change(3)
    return_5d = df["close"].pct_change(5)
    return_10d = df["close"].pct_change(10)
    return_20d = df["close"].pct_change(20)
    volatility_5d = return_1d.rolling(5).std()
    volatility_10d = return_1d.rolling(10).std()
    volatility_20d = return_1d.rolling(20).std()
    volatility_60d = return_1d.rolling(60).std()
    price_volatility = _safe_div(df["high"] - df["low"], df["close"])
    volume_ma5 = df["volume"].rolling(5).mean()
    volume_ma20 = df["volume"].rolling(20).mean()
    volume_ratio_5 = _safe_div(df["volume"], volume_ma5)
    volume_ma5_ma20 = _safe_div(volume_ma5, volume_ma20)
    amount_ma5 = amount.rolling(5).mean()
    amount_ma20 = amount.rolling(20).mean()
    amount_ma5_ma20 = _safe_div(amount_ma5, amount_ma20)
    amount_change_rate = amount.pct_change(fill_method=None)
    amount_return_5d = amount.pct_change(5, fill_method=None)
    breakout_5d = (df["close"] > df["high"].shift(1).rolling(5).max()).astype(float)
    breakout_10d = (df["close"] > df["high"].shift(1).rolling(10).max()).astype(float)
    sma_5 = df["close"].rolling(5).mean()
    sma_20 = df["close"].rolling(20).mean()
    sma_ratio = _safe_div(sma_5, sma_20)
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    rsi_14 = _rsi(df["close"], 14)
    cci_14 = _cci(df, 14)
    atr_14 = _atr(df, 14)
    obv = _obv(df)
    obv_change = obv.pct_change(fill_method=None)
    amplitude = _safe_div(df["high"] - df["low"], df["open"])
    turnover_rate = df.get("turnover_rate", np.nan)
    pe_ttm = _scaled_log_feature(df.get("市盈率TTM", np.nan), 200.0)
    pb = _scaled_log_feature(df.get("市净率", np.nan), 20.0)
    ps_ttm = _scaled_log_feature(df.get("市销率TTM", np.nan), 30.0)
    dividend_ttm = _scaled_log_feature(df.get("股息率TTM", np.nan), 15.0)
    estimated_float_shares = _safe_div(amount, df["close"] * 0.7)
    turnover_rate_est = _safe_div(df["volume"], estimated_float_shares)
    turnover_change = turnover_rate.pct_change(fill_method=None) if "turnover_rate" in df else turnover_rate_est.pct_change(fill_method=None)
    money_flow_proxy = _safe_div(amount, market_cap_yi * 1e8)
    net_mf_amount = pd.to_numeric(df.get("净流入额", np.nan), errors="coerce")
    moneyflow_source_missing = pd.concat(
        [
            pd.to_numeric(df.get("净流入额", np.nan), errors="coerce"),
            pd.to_numeric(df.get("大单买入额", np.nan), errors="coerce"),
            pd.to_numeric(df.get("大单卖出额", np.nan), errors="coerce"),
            pd.to_numeric(df.get("特大单买入额", np.nan), errors="coerce"),
            pd.to_numeric(df.get("特大单卖出额", np.nan), errors="coerce"),
        ],
        axis=1,
    ).isna().all(axis=1)
    net_mf_amount_ratio = _safe_div(net_mf_amount, amount)
    net_lg_amount_ratio = _safe_div(
        pd.to_numeric(df.get("大单买入额", np.nan), errors="coerce")
        + pd.to_numeric(df.get("特大单买入额", np.nan), errors="coerce")
        - pd.to_numeric(df.get("大单卖出额", np.nan), errors="coerce")
        - pd.to_numeric(df.get("特大单卖出额", np.nan), errors="coerce"),
        amount,
    )
    net_elg_amount_ratio = _safe_div(
        pd.to_numeric(df.get("特大单买入额", np.nan), errors="coerce")
        - pd.to_numeric(df.get("特大单卖出额", np.nan), errors="coerce"),
        amount,
    )
    net_sm_amount_ratio = _safe_div(
        pd.to_numeric(df.get("小单买入额", np.nan), errors="coerce")
        + pd.to_numeric(df.get("中单买入额", np.nan), errors="coerce")
        - pd.to_numeric(df.get("小单卖出额", np.nan), errors="coerce")
        - pd.to_numeric(df.get("中单卖出额", np.nan), errors="coerce"),
        amount,
    )
    net_mf_amount_ratio = net_mf_amount_ratio.mask(moneyflow_source_missing, 0.0)
    net_lg_amount_ratio = net_lg_amount_ratio.mask(moneyflow_source_missing, 0.0)
    net_elg_amount_ratio = net_elg_amount_ratio.mask(moneyflow_source_missing, 0.0)
    net_sm_amount_ratio = net_sm_amount_ratio.mask(moneyflow_source_missing, 0.0)
    net_mf_ratio_5d = net_mf_amount_ratio.rolling(5).mean()
    net_lg_ratio_5d = net_lg_amount_ratio.rolling(5).mean()
    net_elg_ratio_5d = net_elg_amount_ratio.rolling(5).mean()
    net_mf_ratio_10d = net_mf_amount_ratio.rolling(10).mean()
    net_lg_ratio_10d = net_lg_amount_ratio.rolling(10).mean()
    net_elg_ratio_10d = net_elg_amount_ratio.rolling(10).mean()
    net_mf_ratio_20d = net_mf_amount_ratio.rolling(20).mean()
    mf_acceleration = net_mf_amount_ratio.diff()
    mf_change_rate_3d = net_mf_ratio_5d - net_mf_ratio_5d.shift(3)
    mf_change_rate_5d = net_mf_ratio_5d - net_mf_ratio_5d.shift(5)
    mf_std_20d = net_mf_amount_ratio.rolling(20).std()
    mf_zscore_20d = _safe_div(net_mf_amount_ratio - net_mf_ratio_20d, mf_std_20d)
    mf_return_divergence_5d = net_mf_ratio_5d - return_5d
    mf_return_confirmation_5d = net_mf_ratio_5d * return_5d
    lg_minus_mf_ratio = net_lg_amount_ratio - net_mf_amount_ratio
    elg_flow_bias = net_elg_amount_ratio - net_lg_amount_ratio
    inst_buy = _numeric_series_or_default("机构买入额")
    inst_sell = _numeric_series_or_default("机构卖出额")
    inst_net = _numeric_series_or_default("机构净买额")
    inst_count = _numeric_series_or_default("机构席位数")
    top_inst_buy_ratio = _safe_div(inst_buy, amount).fillna(0.0)
    top_inst_sell_ratio = _safe_div(inst_sell, amount).fillna(0.0)
    top_inst_net_buy_ratio = _safe_div(inst_net, amount).fillna(0.0)
    top_inst_active = (inst_count > 0).astype(float)
    top_inst_net_buy_5d = top_inst_net_buy_ratio.rolling(5).sum().fillna(0.0)
    top_inst_active_5d = top_inst_active.rolling(5).max().fillna(0.0)
    margin_rzye = _numeric_series_or_default("融资余额")
    margin_rqye = _numeric_series_or_default("融券余额")
    margin_rzmre = _numeric_series_or_default("融资买入额")
    margin_rzche = _numeric_series_or_default("融资偿还额")
    margin_rqyl = _numeric_series_or_default("融券余量")
    margin_rqchl = _numeric_series_or_default("融券偿还量")
    margin_rqmcl = _numeric_series_or_default("融券卖出量")
    margin_total = _numeric_series_or_default("融资融券余额")
    margin_rzye_ratio = _safe_div(margin_rzye, amount).fillna(0.0)
    margin_rzmre_ratio = _safe_div(margin_rzmre, amount).fillna(0.0)
    margin_rz_net_buy_ratio = _safe_div(margin_rzmre - margin_rzche, amount).fillna(0.0)
    margin_rqye_ratio = _safe_div(margin_rqye, amount).fillna(0.0)
    margin_rq_net_sell_ratio = _safe_div(margin_rqmcl - margin_rqchl, df["volume"]).fillna(0.0)
    margin_rqyl_ratio = _safe_div(margin_rqyl, df["volume"]).fillna(0.0)
    margin_total_balance_ratio = _safe_div(margin_total, amount).fillna(0.0)
    margin_rz_net_buy_5d = margin_rz_net_buy_ratio.rolling(5).sum().fillna(0.0)
    margin_rq_net_sell_5d = margin_rq_net_sell_ratio.rolling(5).sum().fillna(0.0)
    market_cap_ratio = _safe_div(df["close"], pd.Series(market_cap_yi, index=df.index, dtype=float))
    amount_market_cap_ratio = _safe_div(amount, market_cap_yi * 1e8)
    stability_inverse = 1 / volatility_20d.replace(0, np.nan)
    volume_concentration = _safe_div(df["volume"].rolling(20).std(), df["volume"].rolling(20).mean())
    relative_strength_proxy = return_20d - return_20d.rolling(20).mean()
    trend_strength = _safe_div(df["close"] - sma_20, sma_20)
    close_to_high = _safe_div(df["close"], df["high"].rolling(20).max())
    close_to_low = _safe_div(df["close"], df["low"].rolling(20).min())
    gap_open = _safe_div(df["open"] - df["close"].shift(1), df["close"].shift(1))
    return_vol_adj_5d = _safe_div(return_5d, volatility_20d)
    mf_volatility_adj = _safe_div(net_mf_amount_ratio, volatility_20d)
    price_amount_divergence_5d = return_5d - amount_return_5d
    body = (df["close"] - df["open"]).abs()
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    upper_shadow_ratio = _safe_div(df["high"] - df[["open", "close"]].max(axis=1), candle_range)
    lower_shadow_ratio = _safe_div(df[["open", "close"]].min(axis=1) - df["low"], candle_range)
    price_acceleration = return_5d.diff()
    volume_acceleration = volume_ratio_5.diff()
    beta_proxy = _safe_div(return_20d, volatility_20d)

    macd_hist = macd - macd_signal
    peak_flag = ((df["high"].shift(2) < df["high"].shift(1)) & (df["high"].shift(1) >= df["high"]))
    trough_flag = ((df["low"].shift(2) > df["low"].shift(1)) & (df["low"].shift(1) <= df["low"]))
    swing_peak_count_20 = peak_flag.astype(float).rolling(20).sum()
    swing_trough_count_20 = trough_flag.astype(float).rolling(20).sum()
    last_swing_high = _ffill_pivot_values(df["high"].shift(1), peak_flag)
    last_swing_low = _ffill_pivot_values(df["low"].shift(1), trough_flag)
    close_to_last_swing_high = _safe_div(df["close"] - last_swing_high, last_swing_high)
    close_to_last_swing_low = _safe_div(df["close"] - last_swing_low, last_swing_low)

    rolling_high_20 = df["high"].rolling(20).max()
    rolling_low_20 = df["low"].rolling(20).min()
    higher_high = (rolling_high_20 > rolling_high_20.shift(10)).astype(float)
    higher_low = (rolling_low_20 > rolling_low_20.shift(10)).astype(float)
    lower_high = (rolling_high_20 < rolling_high_20.shift(10)).astype(float)
    lower_low = (rolling_low_20 < rolling_low_20.shift(10)).astype(float)
    chan_trend_type_20 = np.where(
        (higher_high > 0) & (higher_low > 0),
        1.0,
        np.where((lower_high > 0) & (lower_low > 0), -1.0, 0.0),
    )
    chan_trend_strength_20 = (
        _safe_div(rolling_high_20 - rolling_high_20.shift(10), rolling_high_20.shift(10)).abs()
        + _safe_div(rolling_low_20 - rolling_low_20.shift(10), rolling_low_20.shift(10)).abs()
    ) / 2.0

    seg1_high = df["high"].rolling(5).max()
    seg1_low = df["low"].rolling(5).min()
    seg2_high = df["high"].shift(5).rolling(5).max()
    seg2_low = df["low"].shift(5).rolling(5).min()
    seg3_high = df["high"].shift(10).rolling(5).max()
    seg3_low = df["low"].shift(10).rolling(5).min()
    zhongshu_high = pd.concat([seg1_high, seg2_high, seg3_high], axis=1).min(axis=1)
    zhongshu_low = pd.concat([seg1_low, seg2_low, seg3_low], axis=1).max(axis=1)
    zhongshu_center = (zhongshu_high + zhongshu_low) / 2.0
    zhongshu_exists = zhongshu_high > zhongshu_low
    chan_zhongshu_exists_15 = zhongshu_exists.astype(float)
    chan_zhongshu_width_15 = _safe_div((zhongshu_high - zhongshu_low).clip(lower=0), df["close"])
    chan_price_vs_zhongshu_15 = _safe_div(df["close"] - zhongshu_center, df["close"]).where(zhongshu_exists)

    prior_price_high = df["high"].shift(1).rolling(10).max()
    prior_price_low = df["low"].shift(1).rolling(10).min()
    prior_hist_high = macd_hist.shift(1).rolling(10).max()
    prior_hist_low = macd_hist.shift(1).rolling(10).min()
    bearish_strength = (_safe_div(df["high"] - prior_price_high, prior_price_high)).clip(lower=0).fillna(0.0)
    bullish_strength = (_safe_div(prior_price_low - df["low"], prior_price_low)).clip(lower=0).fillna(0.0)
    hist_bear_gap = (_safe_div(prior_hist_high - macd_hist, prior_hist_high.abs() + 1e-6)).clip(lower=0).fillna(0.0)
    hist_bull_gap = (_safe_div(macd_hist - prior_hist_low, prior_hist_low.abs() + 1e-6)).clip(lower=0).fillna(0.0)
    chan_bearish_divergence_10 = ((df["high"] >= prior_price_high) & (macd_hist < prior_hist_high)).astype(float)
    chan_bullish_divergence_10 = ((df["low"] <= prior_price_low) & (macd_hist > prior_hist_low)).astype(float)
    chan_divergence_score_10 = (chan_bullish_divergence_10 * bullish_strength * hist_bull_gap) - (chan_bearish_divergence_10 * bearish_strength * hist_bear_gap)

    is_tech = float(any(keyword in industry_name for keyword in tech_keywords))
    is_consumer = float(any(keyword in industry_name for keyword in consumer_keywords))
    is_medicine = float(any(keyword in industry_name for keyword in medicine_keywords))
    is_finance = float(any(keyword in industry_name for keyword in finance_keywords))
    is_cycle = float(any(keyword in industry_name for keyword in cycle_keywords))
    is_kcb_cyb = float(any(keyword in market_name for keyword in ("科创板", "创业板")))
    if pd.isna(listing_date):
        listed_days = pd.Series(0.0, index=df.index)
    else:
        listed_days = (df["date"] - listing_date).dt.days.clip(lower=0).astype(float)
    listed_days = np.log1p(listed_days.clip(upper=8000.0)) / np.log1p(8000.0)

    feature_columns = _build_feature_frame(
        df.index,
        {
            "layer": layer,
            "market_cap_yi": market_cap_yi,
            "amount": amount,
            "return_1d": return_1d,
            "return_3d": return_3d,
            "return_5d": return_5d,
            "return_10d": return_10d,
            "return_20d": return_20d,
            "volatility_5d": volatility_5d,
            "volatility_10d": volatility_10d,
            "volatility_20d": volatility_20d,
            "volatility_60d": volatility_60d,
            "price_volatility": price_volatility,
            "volume_ma5": volume_ma5,
            "volume_ma20": volume_ma20,
            "volume_ratio_5": volume_ratio_5,
            "volume_ma5_ma20": volume_ma5_ma20,
            "amount_ma5": amount_ma5,
            "amount_ma20": amount_ma20,
            "amount_ma5_ma20": amount_ma5_ma20,
            "amount_change_rate": amount_change_rate,
            "amount_return_5d": amount_return_5d,
            "breakout_5d": breakout_5d,
            "breakout_10d": breakout_10d,
            "sma_5": sma_5,
            "sma_20": sma_20,
            "sma_ratio": sma_ratio,
            "ema_12": ema_12,
            "ema_26": ema_26,
            "macd": macd,
            "macd_signal": macd_signal,
            "rsi_14": rsi_14,
            "cci_14": cci_14,
            "atr_14": atr_14,
            "obv": obv,
            "obv_change": obv_change,
            "amplitude": amplitude,
            "price_amplitude": amplitude,
            "turnover_rate": turnover_rate,
            "pe_ttm": pe_ttm,
            "pb": pb,
            "ps_ttm": ps_ttm,
            "dividend_ttm": dividend_ttm,
            "turnover_rate_est": turnover_rate_est,
            "turnover_change": turnover_change,
            "money_flow_proxy": money_flow_proxy,
            "net_mf_amount": net_mf_amount,
            "net_mf_amount_ratio": net_mf_amount_ratio,
            "net_lg_amount_ratio": net_lg_amount_ratio,
            "net_elg_amount_ratio": net_elg_amount_ratio,
            "net_sm_amount_ratio": net_sm_amount_ratio,
            "net_mf_ratio_5d": net_mf_ratio_5d,
            "net_lg_ratio_5d": net_lg_ratio_5d,
            "net_elg_ratio_5d": net_elg_ratio_5d,
            "net_mf_ratio_10d": net_mf_ratio_10d,
            "net_lg_ratio_10d": net_lg_ratio_10d,
            "net_elg_ratio_10d": net_elg_ratio_10d,
            "net_mf_ratio_20d": net_mf_ratio_20d,
            "mf_acceleration": mf_acceleration,
            "mf_change_rate_3d": mf_change_rate_3d,
            "mf_change_rate_5d": mf_change_rate_5d,
            "mf_zscore_20d": mf_zscore_20d,
            "mf_return_divergence_5d": mf_return_divergence_5d,
            "mf_return_confirmation_5d": mf_return_confirmation_5d,
            "lg_minus_mf_ratio": lg_minus_mf_ratio,
            "elg_flow_bias": elg_flow_bias,
            "top_inst_buy_ratio": top_inst_buy_ratio,
            "top_inst_sell_ratio": top_inst_sell_ratio,
            "top_inst_net_buy_ratio": top_inst_net_buy_ratio,
            "top_inst_count": inst_count,
            "top_inst_active": top_inst_active,
            "top_inst_net_buy_5d": top_inst_net_buy_5d,
            "top_inst_active_5d": top_inst_active_5d,
            "margin_rzye_ratio": margin_rzye_ratio,
            "margin_rzmre_ratio": margin_rzmre_ratio,
            "margin_rz_net_buy_ratio": margin_rz_net_buy_ratio,
            "margin_rqye_ratio": margin_rqye_ratio,
            "margin_rq_net_sell_ratio": margin_rq_net_sell_ratio,
            "margin_rqyl_ratio": margin_rqyl_ratio,
            "margin_total_balance_ratio": margin_total_balance_ratio,
            "margin_rz_net_buy_5d": margin_rz_net_buy_5d,
            "margin_rq_net_sell_5d": margin_rq_net_sell_5d,
            "market_cap_ratio": market_cap_ratio,
            "amount_market_cap_ratio": amount_market_cap_ratio,
            "stability_inverse": stability_inverse,
            "volume_concentration": volume_concentration,
            "relative_strength_proxy": relative_strength_proxy,
            "trend_strength": trend_strength,
            "close_to_high": close_to_high,
            "close_to_low": close_to_low,
            "gap_open": gap_open,
            "return_vol_adj_5d": return_vol_adj_5d,
            "mf_volatility_adj": mf_volatility_adj,
            "price_amount_divergence_5d": price_amount_divergence_5d,
            "upper_shadow_ratio": upper_shadow_ratio,
            "lower_shadow_ratio": lower_shadow_ratio,
            "price_acceleration": price_acceleration,
            "volume_acceleration": volume_acceleration,
            "beta_proxy": beta_proxy,
            "swing_peak_count_20": swing_peak_count_20,
            "swing_trough_count_20": swing_trough_count_20,
            "close_to_last_swing_high": close_to_last_swing_high,
            "close_to_last_swing_low": close_to_last_swing_low,
            "chan_trend_type_20": chan_trend_type_20,
            "chan_trend_strength_20": chan_trend_strength_20,
            "chan_zhongshu_exists_15": chan_zhongshu_exists_15,
            "chan_zhongshu_width_15": chan_zhongshu_width_15,
            "chan_price_vs_zhongshu_15": chan_price_vs_zhongshu_15,
            "chan_bearish_divergence_10": chan_bearish_divergence_10,
            "chan_bullish_divergence_10": chan_bullish_divergence_10,
            "chan_divergence_score_10": chan_divergence_score_10,
            "is_tech": is_tech,
            "is_consumer": is_consumer,
            "is_medicine": is_medicine,
            "is_finance": is_finance,
            "is_cycle": is_cycle,
            "is_kcb_cyb": is_kcb_cyb,
            "listed_days": listed_days,
        },
    )
    duplicate_columns = [column_name for column_name in feature_columns.columns if column_name in df.columns]
    if duplicate_columns:
        feature_columns = feature_columns.drop(columns=duplicate_columns)
    return pd.concat([df, feature_columns], axis=1)
