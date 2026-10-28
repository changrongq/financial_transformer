#!/usr/bin/env python3
"""
股票分析系统命令行入口 - 基于船长架构建议设计
设计原则:
1. 单一入口: 永远只需要记住一个命令
2. 解析命令行: 入口脚本主要解析命令行参数
3. 调用子脚本: 解析后调用其他脚本完成实际工作
4. 可扩展架构: 以后需要相关能力，往这个命令行加
5. 工具复用: 自己写的脚本和工具要复用

船长指导时间: 2026年4月21日14:02
船长指导内容: "入口的脚本主要解析命令行，解析完命令行下一步就可以调用其他脚本完成实际工作了"
"""

import argparse
import sys
import os
import subprocess
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIG_DIR = PROJECT_ROOT / "config"

def setup_argparse():
    """设置命令行参数解析"""
    parser = argparse.ArgumentParser(
        description="股票分析系统命令行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 获取股票数据
  python stock_cli.py data --code 002829 --days 30
  
  # 技术分析
  python stock_cli.py analyze --code 002829 --indicators ma,macd,rsi
  
  # 候选筛选
  python stock_cli.py screen --market A --min-volume 10000000
  
  # 模型推荐
  python stock_cli.py model --date 20260421
  
  # 账户监控
  python stock_cli.py monitor --account paper
  
  # 帮助
  python stock_cli.py --help
  python stock_cli.py data --help
        """
    )
    
    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # data 命令: 数据获取
    data_parser = subparsers.add_parser("data", help="获取股票数据")
    data_parser.add_argument("--code", type=str, help="股票代码，如002829")
    data_parser.add_argument("--name", type=str, help="股票名称")
    data_parser.add_argument("--days", type=int, default=30, help="获取天数")
    data_parser.add_argument("--start-date", type=str, help="开始日期YYYYMMDD")
    data_parser.add_argument("--end-date", type=str, help="结束日期YYYYMMDD")
    data_parser.add_argument("--output", type=str, help="输出文件路径")
    data_parser.add_argument("--format", choices=["csv", "json", "excel"], default="csv", help="输出格式")
    
    # analyze 命令: 技术分析
    analyze_parser = subparsers.add_parser("analyze", help="技术分析")
    analyze_parser.add_argument("--code", type=str, required=True, help="股票代码")
    analyze_parser.add_argument("--indicators", type=str, default="ma,macd,rsi,boll", 
                               help="技术指标，逗号分隔: ma,macd,rsi,boll")
    analyze_parser.add_argument("--period", type=int, default=60, help="分析周期(天)")
    analyze_parser.add_argument("--output", type=str, help="输出文件路径")
    
    # screen 命令: 候选筛选
    screen_parser = subparsers.add_parser("screen", help="候选股票筛选")
    screen_parser.add_argument("--market", type=str, default="A", help="市场: A(主板),创业板,科创板")
    screen_parser.add_argument("--min-volume", type=int, default=10000000, help="最小成交量(股)")
    screen_parser.add_argument("--min-price", type=float, default=5.0, help="最低价格")
    screen_parser.add_argument("--max-pe", type=float, default=100.0, help="最高市盈率")
    screen_parser.add_argument("--output", type=str, help="输出文件路径")
    screen_parser.add_argument("--limit", type=int, default=50, help="输出数量限制")
    
    # model 命令: 模型推荐
    model_parser = subparsers.add_parser("model", help="模型推荐")
    model_parser.add_argument("--date", type=str, help="推荐日期YYYYMMDD，默认今天")
    model_parser.add_argument("--skip-refresh", action="store_true", help="跳过数据刷新")
    model_parser.add_argument("--output", type=str, help="输出文件路径")
    
    # monitor 命令: 账户监控
    monitor_parser = subparsers.add_parser("monitor", help="账户监控")
    monitor_parser.add_argument("--account", type=str, default="paper", help="账户类型: paper(模拟), real(实盘)")
    monitor_parser.add_argument("--interval", type=int, default=15, help="监控间隔(分钟)")
    monitor_parser.add_argument("--alerts", action="store_true", help="启用警报")
    monitor_parser.add_argument("--output", type=str, help="输出文件路径")
    
    # portfolio 命令: 投资组合
    portfolio_parser = subparsers.add_parser("portfolio", help="投资组合管理")
    portfolio_parser.add_argument("--action", choices=["view", "add", "remove", "rebalance"], 
                                 default="view", help="操作类型")
    portfolio_parser.add_argument("--code", type=str, help="股票代码")
    portfolio_parser.add_argument("--shares", type=int, help="股数")
    portfolio_parser.add_argument("--price", type=float, help="价格")
    portfolio_parser.add_argument("--output", type=str, help="输出文件路径")
    
    return parser

def run_data_command(args):
    """运行数据获取命令"""
    print(f"数据获取: 代码={args.code}, 天数={args.days}")
    
    # 检查数据获取脚本
    data_script = SCRIPTS_DIR / "fetch_stock_data.py"
    if not data_script.exists():
        print(f"错误: 数据获取脚本不存在 {data_script}")
        return False
    
    # 构建命令
    cmd = [
        sys.executable, str(data_script),
        "--code", args.code,
        "--days", str(args.days)
    ]
    
    if args.start_date:
        cmd.extend(["--start-date", args.start_date])
    if args.end_date:
        cmd.extend(["--end-date", args.end_date])
    if args.output:
        cmd.extend(["--output", args.output])
    
    cmd.extend(["--format", args.format])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 执行
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"警告: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        print(f"标准输出: {e.stdout}")
        print(f"标准错误: {e.stderr}")
        return False

def run_analyze_command(args):
    """运行技术分析命令"""
    print(f"技术分析: 代码={args.code}, 指标={args.indicators}")
    
    # 检查技术分析脚本
    analyze_script = SCRIPTS_DIR / "technical_analyzer.py"
    if not analyze_script.exists():
        print(f"错误: 技术分析脚本不存在 {analyze_script}")
        return False
    
    # 构建命令
    cmd = [
        sys.executable, str(analyze_script),
        "--code", args.code,
        "--indicators", args.indicators,
        "--period", str(args.period)
    ]
    
    if args.output:
        cmd.extend(["--output", args.output])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 执行
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"警告: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        print(f"标准输出: {e.stdout}")
        print(f"标准错误: {e.stderr}")
        return False

def run_screen_command(args):
    """运行候选筛选命令"""
    print(f"候选筛选: 市场={args.market}, 最小成交量={args.min_volume}")
    
    # 检查候选筛选脚本
    screen_script = SCRIPTS_DIR / "candidate_screener.py"
    if not screen_script.exists():
        print(f"错误: 候选筛选脚本不存在 {screen_script}")
        return False
    
    # 构建命令
    cmd = [
        sys.executable, str(screen_script),
        "--market", args.market,
        "--min-volume", str(args.min_volume),
        "--min-price", str(args.min_price),
        "--max-pe", str(args.max_pe),
        "--limit", str(args.limit)
    ]
    
    if args.output:
        cmd.extend(["--output", args.output])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 执行
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"警告: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        print(f"标准输出: {e.stdout}")
        print(f"标准错误: {e.stderr}")
        return False

def run_model_command(args):
    """运行模型推荐命令"""
    print(f"模型推荐: 日期={args.date if args.date else '今天'}")
    
    # 检查模型推荐脚本
    model_script = SCRIPTS_DIR / "signal_report.py"
    if not model_script.exists():
        print(f"错误: 模型推荐脚本不存在 {model_script}")
        return False
    
    # 构建命令
    cmd = [sys.executable, str(model_script)]
    
    if args.date:
        cmd.extend(["--date", args.date])
    if args.skip_refresh:
        cmd.append("--skip-refresh-data")
    if args.output:
        cmd.extend(["--output", args.output])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 执行
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"警告: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        print(f"标准输出: {e.stdout}")
        print(f"标准错误: {e.stderr}")
        return False

def run_monitor_command(args):
    """运行账户监控命令"""
    print(f"账户监控: 账户={args.account}, 间隔={args.interval}分钟")
    
    # 检查账户监控脚本
    monitor_script = SCRIPTS_DIR / "account_monitor.py"
    if not monitor_script.exists():
        print(f"错误: 账户监控脚本不存在 {monitor_script}")
        return False
    
    # 构建命令
    cmd = [
        sys.executable, str(monitor_script),
        "--account", args.account,
        "--interval", str(args.interval)
    ]
    
    if args.alerts:
        cmd.append("--alerts")
    if args.output:
        cmd.extend(["--output", args.output])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 执行
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"警告: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        print(f"标准输出: {e.stdout}")
        print(f"标准错误: {e.stderr}")
        return False

def run_portfolio_command(args):
    """运行投资组合命令"""
    print(f"投资组合: 操作={args.action}")
    
    # 检查投资组合脚本
    portfolio_script = SCRIPTS_DIR / "portfolio_manager.py"
    if not portfolio_script.exists():
        print(f"错误: 投资组合脚本不存在 {portfolio_script}")
        return False
    
    # 构建命令
    cmd = [sys.executable, str(portfolio_script), "--action", args.action]
    
    if args.code:
        cmd.extend(["--code", args.code])
    if args.shares:
        cmd.extend(["--shares", str(args.shares)])
    if args.price:
        cmd.extend(["--price", str(args.price)])
    if args.output:
        cmd.extend(["--output", args.output])
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 执行
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(f"警告: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"错误: {e}")
        print(f"标准输出: {e.stdout}")
        print(f"标准错误: {e.stderr}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("股票分析系统命令行入口")
    print("基于船长架构建议设计: '入口的脚本主要解析命令行'")
    print(f"项目目录: {PROJECT_ROOT}")
    print("=" * 60)
    
    # 解析命令行参数
    parser = setup_argparse()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # 根据命令调用相应函数
    success = False
    if args.command == "data":
        success = run_data_command(args)
    elif args.command == "analyze":
        success = run_analyze_command(args)
    elif args.command == "screen":
        success = run_screen_command(args)
    elif args.command == "model":
        success = run_model_command(args)
    elif args.command == "monitor":
        success = run_monitor_command(args)
    elif args.command == "portfolio":
        success = run_portfolio_command(args)
    else:
        print(f"未知命令: {args.command}")
        parser.print_help()
        return 1
    
    # 输出结果
    print("=" * 60)
    if success:
        print("✅ 命令执行成功")
        return 0
    else:
        print("❌ 命令执行失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())