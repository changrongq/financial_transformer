#!/usr/bin/env python3
"""
技术分析工具 - 钳星自主选股研究工具
用于计算技术指标和生成交易信号
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


class TechnicalAnalyzer:
    """技术分析器类"""
    
    def __init__(self, stock_data: pd.DataFrame):
        """
        初始化技术分析器
        
        Args:
            stock_data: 股票数据DataFrame，包含日期、开盘、最高、最低、收盘、成交量
        """
        self.data = stock_data.copy()
        self.calculate_basic_indicators()
    
    def calculate_basic_indicators(self):
        """计算基础技术指标"""
        # 确保数据按日期排序
        self.data = self.data.sort_index()
        
        # 计算日收益率
        self.data['returns'] = self.data['close'].pct_change()
        
        # 计算对数收益率（更符合正态分布）
        self.data['log_returns'] = np.log(self.data['close'] / self.data['close'].shift(1))
    
    def calculate_ma(self, periods: List[int] = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        计算移动平均线
        
        Args:
            periods: 移动平均周期列表
            
        Returns:
            包含各周期移动平均线的DataFrame
        """
        ma_data = self.data.copy()
        
        for period in periods:
            ma_column = f'ma_{period}'
            ma_data[ma_column] = ma_data['close'].rolling(window=period).mean()
            
            # 计算价格与均线的距离（百分比）
            ma_data[f'{ma_column}_distance'] = (
                (ma_data['close'] - ma_data[ma_column]) / ma_data[ma_column] * 100
            )
        
        return ma_data
    
    def calculate_macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        计算MACD指标
        
        Args:
            fast_period: 快速EMA周期
            slow_period: 慢速EMA周期
            signal_period: 信号线周期
            
        Returns:
            包含MACD指标的DataFrame
        """
        macd_data = self.data.copy()
        
        # 计算EMA
        ema_fast = macd_data['close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = macd_data['close'].ewm(span=slow_period, adjust=False).mean()
        
        # 计算DIF（快慢线差）
        macd_data['macd_dif'] = ema_fast - ema_slow
        
        # 计算DEA（信号线）
        macd_data['macd_dea'] = macd_data['macd_dif'].ewm(span=signal_period, adjust=False).mean()
        
        # 计算MACD柱状图
        macd_data['macd_histogram'] = macd_data['macd_dif'] - macd_data['macd_dea']
        
        # 计算MACD信号
        macd_data['macd_signal'] = np.where(
            (macd_data['macd_dif'] > macd_data['macd_dea']) & 
            (macd_data['macd_dif'].shift(1) <= macd_data['macd_dea'].shift(1)),
            'golden_cross',  # 金叉
            np.where(
                (macd_data['macd_dif'] < macd_data['macd_dea']) & 
                (macd_data['macd_dif'].shift(1) >= macd_data['macd_dea'].shift(1)),
                'death_cross',  # 死叉
                'neutral'
            )
        )
        
        return macd_data
    
    def calculate_rsi(self, period: int = 14) -> pd.DataFrame:
        """
        计算RSI指标
        
        Args:
            period: RSI计算周期
            
        Returns:
            包含RSI指标的DataFrame
        """
        rsi_data = self.data.copy()
        
        # 计算价格变化
        delta = rsi_data['close'].diff()
        
        # 分离上涨和下跌
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # 计算RS
        rs = gain / loss
        
        # 计算RSI
        rsi_data['rsi'] = 100 - (100 / (1 + rs))
        
        # 计算RSI信号
        rsi_data['rsi_signal'] = np.where(
            rsi_data['rsi'] < 30,
            'oversold',  # 超卖
            np.where(
                rsi_data['rsi'] > 70,
                'overbought',  # 超买
                'neutral'
            )
        )
        
        return rsi_data
    
    def calculate_bollinger_bands(self, period: int = 20, num_std: float = 2) -> pd.DataFrame:
        """
        计算布林带
        
        Args:
            period: 移动平均周期
            num_std: 标准差倍数
            
        Returns:
            包含布林带的DataFrame
        """
        bb_data = self.data.copy()
        
        # 计算中轨（移动平均）
        bb_data['bb_middle'] = bb_data['close'].rolling(window=period).mean()
        
        # 计算标准差
        bb_std = bb_data['close'].rolling(window=period).std()
        
        # 计算上轨和下轨
        bb_data['bb_upper'] = bb_data['bb_middle'] + (bb_std * num_std)
        bb_data['bb_lower'] = bb_data['bb_middle'] - (bb_std * num_std)
        
        # 计算布林带宽度（波动性指标）
        bb_data['bb_width'] = (bb_data['bb_upper'] - bb_data['bb_lower']) / bb_data['bb_middle']
        
        # 计算价格位置（%b指标）
        bb_data['bb_percent_b'] = (bb_data['close'] - bb_data['bb_lower']) / (bb_data['bb_upper'] - bb_data['bb_lower'])
        
        # 计算布林带信号
        bb_data['bb_signal'] = np.where(
            bb_data['close'] < bb_data['bb_lower'],
            'oversold',  # 跌破下轨，可能超卖
            np.where(
                bb_data['close'] > bb_data['bb_upper'],
                'overbought',  # 突破上轨，可能超买
                'neutral'
            )
        )
        
        return bb_data
    
    def generate_technical_signals(self) -> Dict:
        """
        生成综合技术信号
        
        Returns:
            包含各种技术信号的字典
        """
        # 计算各种指标
        ma_data = self.calculate_ma()
        macd_data = self.calculate_macd()
        rsi_data = self.calculate_rsi()
        bb_data = self.calculate_bollinger_bands()
        
        # 获取最新数据
        latest = {
            'price': self.data['close'].iloc[-1],
            'ma_signals': {
                'ma_5': ma_data['ma_5'].iloc[-1],
                'ma_20': ma_data['ma_20'].iloc[-1],
                'ma_60': ma_data['ma_60'].iloc[-1],
                'trend': 'bullish' if ma_data['ma_5'].iloc[-1] > ma_data['ma_20'].iloc[-1] > ma_data['ma_60'].iloc[-1] else 'bearish'
            },
            'macd_signals': {
                'dif': macd_data['macd_dif'].iloc[-1],
                'dea': macd_data['macd_dea'].iloc[-1],
                'histogram': macd_data['macd_histogram'].iloc[-1],
                'signal': macd_data['macd_signal'].iloc[-1]
            },
            'rsi_signals': {
                'rsi': rsi_data['rsi'].iloc[-1],
                'signal': rsi_data['rsi_signal'].iloc[-1]
            },
            'bollinger_signals': {
                'upper': bb_data['bb_upper'].iloc[-1],
                'middle': bb_data['bb_middle'].iloc[-1],
                'lower': bb_data['bb_lower'].iloc[-1],
                'width': bb_data['bb_width'].iloc[-1],
                'percent_b': bb_data['bb_percent_b'].iloc[-1],
                'signal': bb_data['bb_signal'].iloc[-1]
            }
        }
        
        # 生成综合信号评分（0-100）
        signal_score = 50  # 基础分
        
        # MACD金叉加分，死叉减分
        if latest['macd_signals']['signal'] == 'golden_cross':
            signal_score += 15
        elif latest['macd_signals']['signal'] == 'death_cross':
            signal_score -= 15
        
        # RSI超卖加分，超买减分
        if latest['rsi_signals']['signal'] == 'oversold':
            signal_score += 10
        elif latest['rsi_signals']['signal'] == 'overbought':
            signal_score -= 10
        
        # 价格在布林带下轨附近加分，上轨附近减分
        if latest['bollinger_signals']['signal'] == 'oversold':
            signal_score += 10
        elif latest['bollinger_signals']['signal'] == 'overbought':
            signal_score -= 10
        
        # 趋势向上加分
        if latest['ma_signals']['trend'] == 'bullish':
            signal_score += 15
        
        # 确保分数在0-100之间
        signal_score = max(0, min(100, signal_score))
        
        latest['technical_score'] = signal_score
        latest['technical_rating'] = 'strong_buy' if signal_score >= 80 else \
                                    'buy' if signal_score >= 60 else \
                                    'neutral' if signal_score >= 40 else \
                                    'sell' if signal_score >= 20 else 'strong_sell'
        
        return latest
    
    def analyze_stock(self, stock_code: str, days: int = 60) -> Dict:
        """
        分析单只股票
        
        Args:
            stock_code: 股票代码
            days: 历史数据天数
            
        Returns:
            股票分析结果字典
        """
        try:
            # 尝试从paper_trading_client获取历史数据
            # 这里需要实际的数据获取逻辑
            # 暂时使用示例数据进行分析
            
            # 生成技术信号
            technical_signals = self.generate_technical_signals()
            
            # 基于技术信号生成建议
            recommendation = self._generate_recommendation(technical_signals)
            risk_level = self._assess_risk_level(technical_signals)
            position_suggestion = self._suggest_position(technical_signals)
            
            return {
                'stock_code': stock_code,
                'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
                'data_period': f'{days}天',
                'technical_signals': technical_signals,
                'recommendation': recommendation,
                'risk_level': risk_level,
                'position_suggestion': position_suggestion,
                'analysis_method': '技术分析（MA/MACD/RSI/布林带）'
            }
            
        except Exception as e:
            return {
                'stock_code': stock_code,
                'error': str(e),
                'recommendation': '数据获取失败，请检查数据源',
                'risk_level': 'unknown',
                'position_suggestion': '无法分析'
            }
    
    def _generate_recommendation(self, signals: Dict) -> str:
        """根据技术信号生成建议"""
        rating = signals.get('technical_rating', 'neutral')
        
        recommendation_map = {
            'strong_buy': '强烈买入：多重技术指标显示强烈买入信号',
            'buy': '买入：技术指标总体向好，适合买入',
            'neutral': '中性：技术指标无明显方向，建议观望',
            'sell': '卖出：技术指标总体向差，建议卖出',
            'strong_sell': '强烈卖出：多重技术指标显示强烈卖出信号'
        }
        
        return recommendation_map.get(rating, '无法判断')
    
    def _assess_risk_level(self, signals: Dict) -> str:
        """评估风险等级"""
        score = signals.get('technical_score', 50)
        
        if score >= 70:
            return '低风险：技术面强劲，下跌风险较小'
        elif score >= 50:
            return '中风险：技术面中性，需关注市场变化'
        elif score >= 30:
            return '较高风险：技术面偏弱，需谨慎'
        else:
            return '高风险：技术面很弱，风险较大'
    
    def _suggest_position(self, signals: Dict) -> str:
        """建议仓位"""
        score = signals.get('technical_score', 50)
        rating = signals.get('technical_rating', 'neutral')
        
        if rating == 'strong_buy':
            return '建议仓位：5-8%（强烈信号）'
        elif rating == 'buy':
            return '建议仓位：3-5%（看好信号）'
        elif rating == 'neutral':
            return '建议仓位：0-2%（观望为主）'
        elif rating == 'sell':
            return '建议仓位：0%（建议卖出）'
        elif rating == 'strong_sell':
            return '建议仓位：0%（强烈建议卖出）'
        else:
            return '建议仓位：无法判断'


def main():
    """主函数 - 示例用法"""
    print("钳星技术分析工具 v1.0")
    print("=" * 50)
    
    # 示例数据（实际使用时需要从数据源获取）
    dates = pd.date_range(start='2026-03-01', end='2026-04-20', freq='D')
    np.random.seed(42)
    
    # 生成示例数据
    sample_data = pd.DataFrame({
        'open': np.random.normal(100, 5, len(dates)),
        'high': np.random.normal(105, 5, len(dates)),
        'low': np.random.normal(95, 5, len(dates)),
        'close': np.random.normal(100, 5, len(dates)),
        'volume': np.random.randint(100000, 1000000, len(dates))
    }, index=dates)
    
    # 创建分析器
    analyzer = TechnicalAnalyzer(sample_data)
    
    # 生成技术信号
    signals = analyzer.generate_technical_signals()
    
    print("技术信号分析结果:")
    print(f"价格: {signals['price']:.2f}")
    print(f"技术评分: {signals['technical_score']}/100")
    print(f"技术评级: {signals['technical_rating']}")
    print(f"趋势: {signals['ma_signals']['trend']}")
    print(f"MACD信号: {signals['macd_signals']['signal']}")
    print(f"RSI: {signals['rsi_signals']['rsi']:.1f} ({signals['rsi_signals']['signal']})")
    print(f"布林带信号: {signals['bollinger_signals']['signal']}")


if __name__ == "__main__":
    main()