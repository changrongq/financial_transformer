#!/usr/bin/env python3
"""
候选股票筛选器 - 钳星自主选股研究工具
用于筛选符合标准的候选股票
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from technical_analyzer import TechnicalAnalyzer


class CandidateScreener:
    """候选股票筛选器类"""
    
    def __init__(self):
        """初始化筛选器"""
        self.criteria = self._define_screening_criteria()
        self.candidate_pool = []
        
    def _define_screening_criteria(self) -> Dict:
        """定义筛选标准"""
        return {
            'liquidity': {
                'min_daily_volume': 10000000,  # 1000万元成交量
                'min_daily_amount': 50000000,   # 5000万元成交额
                'min_market_cap': 1000000000,   # 10亿元市值
                'min_turnover_rate': 0.01       # 1%换手率
            },
            'technical': {
                'min_technical_score': 60,      # 技术评分>60
                'trend_requirement': 'upward_or_reversal',  # 上升趋势或底部反转
                'min_rsi': 50,                  # RSI>50
                'macd_signal': 'golden_cross_or_up'  # MACD金叉或向上
            },
            'fundamental': {
                'max_pe_ratio': 50,             # 市盈率<50
                'max_pb_ratio': 5,              # 市净率<5
                'revenue_growth': 'positive',   # 营收正增长
                'net_profit': 'positive'        # 净利润为正
            },
            'risk_control': {
                'no_st_warning': True,          # 无ST/*ST警告
                'max_debt_ratio': 0.70,         # 资产负债率<70%
                'industry_position': 'leader_or_strong'  # 行业龙头或强势
            }
        }
    
    def generate_test_candidates(self) -> List[Dict]:
        """生成测试候选股票（实际使用时需要从数据源获取）"""
        # 这里生成一些测试候选股票
        test_candidates = [
            {
                'stock_code': '000001',
                'name': '平安银行',
                'industry': '银行',
                'market_cap': 350000000000,  # 3500亿元
                'pe_ratio': 6.5,
                'pb_ratio': 0.8,
                'daily_volume': 150000000,   # 1.5亿股
                'daily_amount': 1200000000,  # 12亿元
                'turnover_rate': 0.015,      # 1.5%
                'debt_ratio': 0.92,          # 银行行业特殊
                'revenue_growth': 0.05,      # +5%
                'net_profit': 15000000000    # 150亿元
            },
            {
                'stock_code': '000858',
                'name': '五粮液',
                'industry': '白酒',
                'market_cap': 800000000000,  # 8000亿元
                'pe_ratio': 25,
                'pb_ratio': 8,
                'daily_volume': 20000000,    # 2000万股
                'daily_amount': 3000000000,  # 30亿元
                'turnover_rate': 0.008,      # 0.8%
                'debt_ratio': 0.35,
                'revenue_growth': 0.12,      # +12%
                'net_profit': 30000000000    # 300亿元
            },
            {
                'stock_code': '002415',
                'name': '海康威视',
                'industry': '安防',
                'market_cap': 300000000000,  # 3000亿元
                'pe_ratio': 20,
                'pb_ratio': 3.5,
                'daily_volume': 50000000,    # 5000万股
                'daily_amount': 1500000000,  # 15亿元
                'turnover_rate': 0.012,      # 1.2%
                'debt_ratio': 0.45,
                'revenue_growth': 0.08,      # +8%
                'net_profit': 15000000000    # 150亿元
            },
            {
                'stock_code': '300750',
                'name': '宁德时代',
                'industry': '新能源',
                'market_cap': 1000000000000, # 10000亿元
                'pe_ratio': 40,
                'pb_ratio': 6,
                'daily_volume': 30000000,    # 3000万股
                'daily_amount': 8000000000,  # 80亿元
                'turnover_rate': 0.010,      # 1.0%
                'debt_ratio': 0.60,
                'revenue_growth': 0.25,      # +25%
                'net_profit': 25000000000    # 250亿元
            },
            {
                'stock_code': '600519',
                'name': '贵州茅台',
                'industry': '白酒',
                'market_cap': 2000000000000, # 20000亿元
                'pe_ratio': 35,
                'pb_ratio': 10,
                'daily_volume': 10000000,    # 1000万股
                'daily_amount': 20000000000, # 200亿元
                'turnover_rate': 0.005,      # 0.5%
                'debt_ratio': 0.25,
                'revenue_growth': 0.15,      # +15%
                'net_profit': 60000000000    # 600亿元
            }
        ]
        
        # 为每个候选股票生成技术信号（模拟）
        for candidate in test_candidates:
            candidate['technical_score'] = np.random.randint(40, 85)
            candidate['trend'] = np.random.choice(['upward', 'sideways', 'reversal'])
            candidate['rsi'] = np.random.randint(30, 80)
            candidate['macd_signal'] = np.random.choice(['golden_cross', 'up', 'neutral', 'down'])
            
        return test_candidates
    
    def screen_candidates(self, candidates: List[Dict]) -> List[Dict]:
        """筛选候选股票"""
        screened_candidates = []
        
        for candidate in candidates:
            # 检查流动性标准
            liquidity_passed = self._check_liquidity(candidate)
            
            # 检查技术标准
            technical_passed = self._check_technical(candidate)
            
            # 检查基本面标准
            fundamental_passed = self._check_fundamental(candidate)
            
            # 检查风险控制标准
            risk_passed = self._check_risk_control(candidate)
            
            # 计算综合得分
            overall_score = self._calculate_overall_score(candidate)
            
            if liquidity_passed and technical_passed and fundamental_passed and risk_passed:
                candidate['screening_result'] = 'passed'
                candidate['overall_score'] = overall_score
                screened_candidates.append(candidate)
            else:
                candidate['screening_result'] = 'failed'
                candidate['overall_score'] = overall_score
        
        # 按综合得分排序
        screened_candidates.sort(key=lambda x: x['overall_score'], reverse=True)
        
        return screened_candidates
    
    def _check_liquidity(self, candidate: Dict) -> bool:
        """检查流动性标准"""
        criteria = self.criteria['liquidity']
        
        # 检查成交量
        if candidate.get('daily_volume', 0) < criteria['min_daily_volume']:
            return False
        
        # 检查成交额
        if candidate.get('daily_amount', 0) < criteria['min_daily_amount']:
            return False
        
        # 检查市值
        if candidate.get('market_cap', 0) < criteria['min_market_cap']:
            return False
        
        # 检查换手率
        if candidate.get('turnover_rate', 0) < criteria['min_turnover_rate']:
            return False
        
        return True
    
    def _check_technical(self, candidate: Dict) -> bool:
        """检查技术标准"""
        criteria = self.criteria['technical']
        
        # 检查技术评分
        if candidate.get('technical_score', 0) < criteria['min_technical_score']:
            return False
        
        # 检查RSI
        if candidate.get('rsi', 0) < criteria['min_rsi']:
            return False
        
        # 检查趋势
        trend = candidate.get('trend', '')
        if criteria['trend_requirement'] == 'upward_or_reversal':
            if trend not in ['upward', 'reversal']:
                return False
        
        # 检查MACD信号
        macd_signal = candidate.get('macd_signal', '')
        if criteria['macd_signal'] == 'golden_cross_or_up':
            if macd_signal not in ['golden_cross', 'up']:
                return False
        
        return True
    
    def _check_fundamental(self, candidate: Dict) -> bool:
        """检查基本面标准"""
        criteria = self.criteria['fundamental']
        
        # 检查市盈率
        if candidate.get('pe_ratio', 0) > criteria['max_pe_ratio']:
            return False
        
        # 检查市净率
        if candidate.get('pb_ratio', 0) > criteria['max_pb_ratio']:
            return False
        
        # 检查营收增长
        if criteria['revenue_growth'] == 'positive':
            if candidate.get('revenue_growth', 0) <= 0:
                return False
        
        # 检查净利润
        if criteria['net_profit'] == 'positive':
            if candidate.get('net_profit', 0) <= 0:
                return False
        
        return True
    
    def _check_risk_control(self, candidate: Dict) -> bool:
        """检查风险控制标准"""
        criteria = self.criteria['risk_control']
        
        # 检查资产负债率（某些行业如银行除外）
        industry = candidate.get('industry', '')
        if industry != '银行':  # 银行行业资产负债率通常很高
            if candidate.get('debt_ratio', 0) > criteria['max_debt_ratio']:
                return False
        
        return True
    
    def _calculate_overall_score(self, candidate: Dict) -> float:
        """计算综合得分（0-100）"""
        score = 50  # 基础分
        
        # 技术评分贡献（30%）
        tech_score = candidate.get('technical_score', 50)
        score += (tech_score - 50) * 0.3
        
        # 基本面贡献（30%）
        # 市盈率越低越好
        pe_ratio = candidate.get('pe_ratio', 50)
        if pe_ratio < 20:
            score += 10
        elif pe_ratio < 30:
            score += 5
        elif pe_ratio > 40:
            score -= 5
        
        # 营收增长贡献（20%）
        revenue_growth = candidate.get('revenue_growth', 0)
        if revenue_growth > 0.20:
            score += 10
        elif revenue_growth > 0.10:
            score += 5
        elif revenue_growth < 0:
            score -= 5
        
        # 流动性贡献（20%）
        turnover_rate = candidate.get('turnover_rate', 0)
        if turnover_rate > 0.02:
            score += 5
        elif turnover_rate < 0.005:
            score -= 5
        
        # 确保分数在0-100之间
        score = max(0, min(100, score))
        
        return round(score, 2)
    
    def generate_screening_report(self, candidates: List[Dict]) -> str:
        """生成筛选报告"""
        report = []
        report.append("# 候选股票筛选报告")
        report.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"筛选标准: {len(self.criteria)}大类，{sum(len(v) for v in self.criteria.values())}项具体标准")
        report.append("")
        
        # 通过筛选的候选
        passed_candidates = [c for c in candidates if c.get('screening_result') == 'passed']
        failed_candidates = [c for c in candidates if c.get('screening_result') == 'failed']
        
        report.append(f"## 筛选结果摘要")
        report.append(f"- 总候选数: {len(candidates)}")
        report.append(f"- 通过筛选: {len(passed_candidates)}")
        report.append(f"- 未通过: {len(failed_candidates)}")
        
        # 避免除零错误
        if len(candidates) > 0:
            pass_rate = len(passed_candidates) / len(candidates) * 100
            report.append(f"- 通过率: {pass_rate:.1f}%")
        else:
            report.append(f"- 通过率: N/A (无候选股票)")
        
        report.append("")
        
        if passed_candidates:
            report.append("## 通过筛选的候选股票（按综合得分排序）")
            report.append("")
            report.append("| 排名 | 股票代码 | 名称 | 行业 | 综合得分 | 技术评分 | 市盈率 | 备注 |")
            report.append("|------|----------|------|------|----------|----------|--------|------|")
            
            for i, candidate in enumerate(passed_candidates[:10], 1):  # 显示前10名
                remark = ""
                if candidate['overall_score'] >= 80:
                    remark = "⭐ 强烈推荐"
                elif candidate['overall_score'] >= 70:
                    remark = "✅ 推荐"
                elif candidate['overall_score'] >= 60:
                    remark = "🟡 可考虑"
                
                report.append(f"| {i} | {candidate['stock_code']} | {candidate['name']} | {candidate['industry']} | {candidate['overall_score']} | {candidate.get('technical_score', 'N/A')} | {candidate.get('pe_ratio', 'N/A')} | {remark} |")
        
        return "\n".join(report)


def main():
    """主函数 - 测试筛选器"""
    print("钳星候选股票筛选器 v1.0")
    print("=" * 60)
    
    # 创建筛选器
    screener = CandidateScreener()
    
    # 生成测试候选
    print("生成测试候选股票...")
    test_candidates = screener.generate_test_candidates()
    print(f"生成 {len(test_candidates)} 只测试候选股票")
    
    # 筛选候选
    print("执行筛选...")
    screened_candidates = screener.screen_candidates(test_candidates)
    
    # 生成报告
    report = screener.generate_screening_report(screened_candidates)
    print("\n" + report)
    
    # 显示详细结果
    passed = [c for c in screened_candidates if c['screening_result'] == 'passed']
    if passed:
        print("\n" + "=" * 60)
        print("通过筛选的候选股票详情:")
        for i, candidate in enumerate(passed[:5], 1):
            print(f"{i}. {candidate['stock_code']} {candidate['name']} - 综合得分: {candidate['overall_score']:.1f}")
    else:
        print("⚠️ 没有候选股票通过筛选，请调整筛选标准")


if __name__ == "__main__":
    main()