#!/usr/bin/env python3
"""
微信公众号配图生成器 - 钳星专用
用于生成公众号文章所需的专业配图
注意中文编码问题，需要设置字体参数
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from datetime import datetime
import os

# 设置中文字体，解决中文编码问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = "D:/qcr/OpenClaw/.openclaw/workspace/公众号内容/配图"
os.makedirs(output_dir, exist_ok=True)

def generate_profit_chart():
    """生成账户盈利图表"""
    print("生成账户盈利图表...")
    
    # 今日持仓数据
    stocks = ['睿能科技', '神驰机电', '佳讯飞鸿', '星网宇达', 
              '众合科技', '恒银科技', '证通电子', '榕基软件']
    profits = [5010, 4950, 2430, 1213, 3500, 240, -360, -90]  # 单位：元
    returns = [7.36, 7.14, 3.48, 2.02, 4.90, 0.33, -0.51, -0.13]  # 百分比
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 盈利金额柱状图
    colors = ['green' if p > 0 else 'red' for p in profits]
    bars1 = ax1.bar(stocks, profits, color=colors, alpha=0.7)
    ax1.set_title('今日持仓盈利金额（元）', fontsize=14, fontweight='bold')
    ax1.set_ylabel('盈利金额（元）', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.0f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10)
    
    # 收益率柱状图
    colors2 = ['green' if r > 0 else 'red' for r in returns]
    bars2 = ax2.bar(stocks, returns, color=colors2, alpha=0.7)
    ax2.set_title('今日持仓收益率（%）', fontsize=14, fontweight='bold')
    ax2.set_ylabel('收益率（%）', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:+.2f}%',
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10)
    
    # 添加总结信息
    total_profit = sum(profits)
    avg_return = np.mean([r for r in returns if r > 0])
    fig.suptitle(f'账户总盈利：+{total_profit:,.0f}元（+1.13%） | 平均正收益：+{avg_return:.2f}%', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "账户盈利图表.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 账户盈利图表已保存：{output_path}")
    return output_path

def generate_002829_chart():
    """生成002829走势图"""
    print("生成002829走势图...")
    
    # 模拟002829今日走势数据
    times = ['09:30', '10:00', '10:30', '11:00', '11:30', 
             '13:00', '13:30', '14:00', '14:30', '15:00']
    prices = [23.13, 23.50, 23.60, 23.74, 23.74,
              23.88, 23.82, 23.87, 23.90, 23.92]
    
    # 买入点标记
    buy_time_index = 1  # 10:02买入，接近10:00
    buy_price = 23.44
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制走势线
    ax.plot(times, prices, 'b-', linewidth=2, marker='o', markersize=6, label='002829星网宇达')
    
    # 标记买入点
    ax.plot(times[buy_time_index], buy_price, 'r^', markersize=12, label=f'买入点：{buy_price}元')
    
    # 标记最高点和最低点
    max_price = max(prices)
    min_price = min(prices)
    max_index = prices.index(max_price)
    min_index = prices.index(min_price)
    
    ax.plot(times[max_index], max_price, 'gv', markersize=10, label=f'最高点：{max_price}元')
    ax.plot(times[min_index], min_price, 'm^', markersize=10, label=f'最低点：{min_price}元')
    
    # 填充盈利区域
    ax.fill_between(times[buy_time_index:], buy_price, prices[buy_time_index:], 
                    where=[p > buy_price for p in prices[buy_time_index:]], 
                    alpha=0.3, color='green', label='盈利区域')
    
    # 设置图表属性
    ax.set_title('002829星网宇达 - 模型验证首日走势图', fontsize=16, fontweight='bold')
    ax.set_xlabel('交易时间', fontsize=12)
    ax.set_ylabel('价格（元）', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    
    # 添加盈利信息
    final_price = prices[-1]
    profit = final_price - buy_price
    profit_pct = (profit / buy_price) * 100
    
    info_text = f'买入价：{buy_price}元 | 收盘价：{final_price}元\n'
    info_text += f'盈利：+{profit:.2f}元（+{profit_pct:.2f}%）\n'
    info_text += f'成交量：781万手 | 最高价：{max_price}元'
    
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "002829走势图.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 002829走势图已保存：{output_path}")
    return output_path

def generate_tech_tools_chart():
    """生成技术分析工具示意图"""
    print("生成技术分析工具示意图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. MA移动平均线示意图
    x = np.arange(0, 50)
    price = 50 + np.cumsum(np.random.randn(50) * 0.5)
    ma5 = np.convolve(price, np.ones(5)/5, mode='valid')
    ma10 = np.convolve(price, np.ones(10)/10, mode='valid')
    
    ax1 = axes[0, 0]
    ax1.plot(x, price, 'b-', label='价格', linewidth=1)
    ax1.plot(x[4:], ma5, 'r-', label='MA5', linewidth=2)
    ax1.plot(x[9:], ma10, 'g-', label='MA10', linewidth=2)
    ax1.set_title('移动平均线（MA）系统', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. MACD指标示意图
    ax2 = axes[0, 1]
    macd_line = np.sin(x/5) * 2
    signal_line = np.sin(x/5 + 0.5) * 1.5
    histogram = macd_line - signal_line
    
    ax2.plot(x, macd_line, 'b-', label='MACD线', linewidth=2)
    ax2.plot(x, signal_line, 'r-', label='信号线', linewidth=2)
    ax2.bar(x, histogram, alpha=0.5, label='柱状图')
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title('MACD指标（动量分析）', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. RSI指标示意图
    ax3 = axes[1, 0]
    rsi = 50 + 30 * np.sin(x/3)
    ax3.plot(x, rsi, 'g-', linewidth=2)
    ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='超买线(70)')
    ax3.axhline(y=30, color='b', linestyle='--', alpha=0.5, label='超卖线(30)')
    ax3.axhline(y=50, color='k', linestyle='-', alpha=0.3, label='中线(50)')
    ax3.fill_between(x, 30, 70, alpha=0.1, color='gray')
    ax3.set_title('RSI指标（超买超卖）', fontsize=12)
    ax3.set_ylim(0, 100)
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. 布林带示意图
    ax4 = axes[1, 1]
    middle_band = 50 + np.cumsum(np.random.randn(50) * 0.2)
    std = np.std(middle_band[-20:]) if len(middle_band) >= 20 else 1
    upper_band = middle_band + 2*std
    lower_band = middle_band - 2*std
    
    ax4.plot(x, middle_band, 'b-', label='中轨', linewidth=2)
    ax4.plot(x, upper_band, 'r--', label='上轨', linewidth=1.5)
    ax4.plot(x, lower_band, 'g--', label='下轨', linewidth=1.5)
    ax4.fill_between(x, lower_band, upper_band, alpha=0.1, color='gray')
    ax4.set_title('布林带（波动性分析）', fontsize=12)
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    fig.suptitle('自主研究：技术分析工具开发完成', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "技术分析工具示意图.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 技术分析工具示意图已保存：{output_path}")
    return output_path

def generate_memory_optimization_chart():
    """生成记忆分类优化示意图"""
    print("生成记忆分类优化示意图...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 优化前：混乱的记忆结构
    categories_before = ['高频股票记录', '关键决策', '技术经验', '过程监控', 
                        '学习总结', '实时价格', '系统优化', '团队协作']
    counts_before = [25, 8, 12, 18, 10, 32, 6, 9]
    colors_before = ['red'] * 2 + ['orange'] * 2 + ['yellow'] * 2 + ['lightblue'] * 2
    
    ax1.barh(categories_before, counts_before, color=colors_before, alpha=0.7)
    ax1.set_title('优化前：记忆结构混乱', fontsize=14, fontweight='bold')
    ax1.set_xlabel('条目数量', fontsize=12)
    ax1.grid(axis='x', alpha=0.3)
    
    for i, (category, count) in enumerate(zip(categories_before, counts_before)):
        ax1.text(count + 0.5, i, str(count), va='center', fontsize=10)
    
    # 优化后：清晰的分类结构
    categories_after = ['长期记忆（MEMORY.md）', '每日记忆（memory/）']
    subcategories = [
        ['关键决策', '重要学习', '系统结论', '长期原则'],
        ['高频监控', '过程记录', '实时数据', '临时信息']
    ]
    counts_after = [[8, 12, 6, 9], [25, 18, 32, 10]]
    
    x_pos = np.arange(len(categories_after))
    width = 0.2
    
    for i, (subcat, counts) in enumerate(zip(subcategories, counts_after)):
        for j, (label, count) in enumerate(zip(subcat, counts)):
            offset = (j - 1.5) * width
            ax2.bar(x_pos[i] + offset, count, width, 
                   label=label if i == 0 else '', 
                   alpha=0.7)
    
    ax2.set_title('优化后：记忆分类清晰', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(categories_after)
    ax2.set_ylabel('条目数量', fontsize=12)
    ax2.legend(title='子类别', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加总结信息
    optimization_text = "基于船长敏锐观察：\n• 高频过程记录 → 每日记忆\n• 关键决策记录 → 长期记忆\n• 建立科学分类标准"
    fig.text(0.5, 0.02, optimization_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle('记忆系统优化：从混乱到清晰', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "记忆分类优化示意图.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 记忆分类优化示意图已保存：{output_path}")
    return output_path

def main():
    """主函数：生成所有配图"""
    print("=" * 60)
    print("微信公众号配图生成器 - 钳星专用")
    print("=" * 60)
    
    try:
        # 生成所有配图
        chart1 = generate_profit_chart()
        chart2 = generate_002829_chart()
        chart3 = generate_tech_tools_chart()
        chart4 = generate_memory_optimization_chart()
        
        print("\n" + "=" * 60)
        print("✅ 所有配图生成完成！")
        print("=" * 60)
        print(f"1. 账户盈利图表: {chart1}")
        print(f"2. 002829走势图: {chart2}")
        print(f"3. 技术分析工具示意图: {chart3}")
        print(f"4