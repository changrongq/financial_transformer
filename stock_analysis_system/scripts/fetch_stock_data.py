#!/usr/bin/env python3
"""
股票数据获取脚本

Token 通过环境变量 TUSHARE_TOKEN 或 config/tushare_token.local 提供。
"""

import argparse
import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import tushare as ts
    from config.tushare_config import TUSHARE_TOKEN
    print("✅ Tushare配置加载成功")
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装tushare: pip install tushare")
    sys.exit(1)

def setup_argparse():
    """设置命令行参数解析"""
    parser = argparse.ArgumentParser(description="股票数据获取脚本")
    
    parser.add_argument("--code", type=str, help="股票代码，如002829")
    parser.add_argument("--name", type=str, help="股票名称")
    parser.add_argument("--days", type=int, default=30, help="获取天数，默认30天")
    parser.add_argument("--start-date", type=str, help="开始日期YYYYMMDD")
    parser.add_argument("--end-date", type=str, help="结束日期YYYYMMDD")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--format", choices=["csv", "json", "excel"], default="csv", help="输出格式")
    
    return parser

def init_tushare():
    """初始化Tushare"""
    try:
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()
        print(f"✅ Tushare初始化成功 (Token: {TUSHARE_TOKEN[:10]}...{TUSHARE_TOKEN[-10:]})")
        return pro
    except Exception as e:
        print(f"❌ Tushare初始化失败: {e}")
        return None

def get_stock_basic(pro, code=None, name=None):
    """获取股票基础信息"""
    try:
        # 获取所有股票基础信息
        df_basic = pro.stock_basic()
        
        if code:
            # 按代码查找
            if code.endswith(('.SZ', '.SH')):
                ts_code = code
            else:
                # 自动补全市场代码
                if code.startswith('6'):
                    ts_code = f"{code}.SH"
                else:
                    ts_code = f"{code}.SZ"
            
            result = df_basic[df_basic['ts_code'] == ts_code]
            if len(result) == 0:
                # 尝试按symbol查找
                result = df_basic[df_basic['symbol'] == code]
        elif name:
            # 按名称查找
            result = df_basic[df_basic['name'].str.contains(name)]
        else:
            result = df_basic
        
        if len(result) == 0:
            print(f"❌ 未找到股票: code={code}, name={name}")
            return None
        
        print(f"✅ 找到 {len(result)} 只股票")
        return result
    
    except Exception as e:
        print(f"❌ 获取股票基础信息失败: {e}")
        return None

def get_daily_data(pro, ts_code, start_date, end_date):
    """获取日线行情数据"""
    try:
        df_daily = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        
        if len(df_daily) == 0:
            print(f"❌ 未获取到日线数据: {ts_code}")
            return None
        
        # 按日期排序
        df_daily = df_daily.sort_values('trade_date')
        
        print(f"✅ 获取到 {len(df_daily)} 条日线数据 ({start_date} 到 {end_date})")
        return df_daily
    
    except Exception as e:
        print(f"❌ 获取日线数据失败: {e}")
        return None

def save_data(df, output_path, format_type):
    """保存数据到文件"""
    try:
        if format_type == "csv":
            df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"✅ 数据已保存到 CSV: {output_path}")
        elif format_type == "json":
            df.to_json(output_path, orient='records', force_ascii=False)
            print(f"✅ 数据已保存到 JSON: {output_path}")
        elif format_type == "excel":
            df.to_excel(output_path, index=False)
            print(f"✅ 数据已保存到 Excel: {output_path}")
        
        return True
    
    except Exception as e:
        print(f"❌ 保存数据失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("股票数据获取脚本")
    print("使用船长提供的Tushare Token")
    print("=" * 60)
    
    # 解析命令行参数
    parser = setup_argparse()
    args = parser.parse_args()
    
    # 检查必要参数
    if not args.code and not args.name:
        print("❌ 必须提供 --code 或 --name 参数")
        parser.print_help()
        return 1
    
    # 初始化Tushare
    pro = init_tushare()
    if not pro:
        return 1
    
    # 获取股票基础信息
    df_basic = get_stock_basic(pro, args.code, args.name)
    if df_basic is None or len(df_basic) == 0:
        return 1
    
    # 显示找到的股票
    print("\n📊 找到的股票:")
    for idx, row in df_basic.iterrows():
        print(f"  {row['ts_code']} - {row['name']} ({row['industry']})")
    
    # 默认处理第一只股票
    ts_code = df_basic.iloc[0]['ts_code']
    stock_name = df_basic.iloc[0]['name']
    
    print(f"\n🎯 选择股票: {ts_code} - {stock_name}")
    
    # 计算日期范围
    end_date = datetime.now().strftime("%Y%m%d")
    
    if args.start_date:
        start_date = args.start_date
    else:
        # 计算开始日期
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y%m%d")
    
    if args.end_date:
        end_date = args.end_date
    
    print(f"📅 日期范围: {start_date} 到 {end_date}")
    
    # 获取日线数据
    df_daily = get_daily_data(pro, ts_code, start_date, end_date)
    if df_daily is None:
        return 1
    
    # 显示数据摘要
    print("\n📈 数据摘要:")
    print(f"  数据条数: {len(df_daily)}")
    print(f"  日期范围: {df_daily['trade_date'].min()} 到 {df_daily['trade_date'].max()}")
    print(f"  价格范围: {df_daily['close'].min():.2f} 到 {df_daily['close'].max():.2f}")
    print(f"  平均成交量: {df_daily['vol'].mean():,.0f} 股")
    
    # 显示最新数据
    latest = df_daily.iloc[-1]
    print(f"\n📊 最新数据 ({latest['trade_date']}):")
    print(f"  收盘价: {latest['close']:.2f}")
    print(f"  涨跌幅: {latest['pct_chg']:.2f}%")
    print(f"  成交量: {latest['vol']:,} 股")
    print(f"  成交额: {latest['amount']:,.0f} 元")
    
    # 保存数据
    if args.output:
        success = save_data(df_daily, args.output, args.format)
        if not success:
            return 1
    else:
        # 默认保存到data目录
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)
        
        output_file = data_dir / f"{ts_code}_{start_date}_{end_date}.csv"
        success = save_data(df_daily, output_file, "csv")
        if not success:
            return 1
    
    print("\n" + "=" * 60)
    print("✅ 数据获取完成")
    return 0

if __name__ == "__main__":
    sys.exit(main())