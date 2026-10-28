# -*- coding: utf-8 -*-
"""回测：首板涨停买入，次日开盘卖出（诚实口径）。

- 标的：最近一个月每个交易日的"首板涨停"股
- 首板定义：当日涨幅达到涨停板（主板10%/创业板科创板20%/ST5%），且【前一交易日未涨停】
- 买入：当日收盘（=涨停价，假设可成交，含一字板）
- 卖出：次日开盘价 open
- 费率：双向佣金万2.5(各最低5元) 已并入简化 + 卖出印花税千1
"""
import tushare as ts
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config.tushare_config import TUSHARE_TOKEN

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

def limit_pct(code, name):
    if 'ST' in name.upper():
        return 0.05
    if code.startswith(('300', '301', '688')):
        return 0.20
    return 0.10

def main():
    cal = pro.trade_cal(start_date='20260701', end_date='20260905')
    trade_days = cal[cal['is_open'] == 1]['cal_date'].sort_values().tolist()
    buy_days = [d for d in trade_days if '20260804' <= d <= '20260903']
    print(f"买入窗口: {len(buy_days)} 天  {buy_days[0]} ~ {buy_days[-1]}")

    stocks = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
    stocks = stocks[stocks['ts_code'].str.match(r'^(60|00|30|68)')]
    code2name = dict(zip(stocks['ts_code'], stocks['name']))
    print(f"股票池: {len(stocks)} 只")

    # 预取窗口内所有交易日的全市场日线（含前一日用于判首板）
    all_days = [d for d in trade_days if '20260803' <= d <= '20260905']
    daily = {}
    for d in all_days:
        try:
            df = pro.daily(trade_date=d)
            daily[d] = df[['ts_code','open','close','pct_chg']]
        except Exception as e:
            daily[d] = None
    print(f"已拉取日线: {sum(1 for v in daily.values() if v is not None)} 天")

    day_idx = {d: i for i, d in enumerate(all_days)}

    trades = []
    for d in buy_days:
        df = daily.get(d)
        if df is None:
            continue
        prev_d = all_days[day_idx[d] - 1]
        prev_df = daily.get(prev_d)
        prev_limit = set()
        if prev_df is not None:
            for _, r in prev_df.iterrows():
                nm = code2name.get(r['ts_code'], '')
                lp = limit_pct(r['ts_code'], nm)
                if r['pct_chg'] >= (lp * 100 - 1.2):
                    prev_limit.add(r['ts_code'])

        for _, r in df.iterrows():
            code = r['ts_code']
            nm = code2name.get(code, '')
            lp = limit_pct(code, nm)
            if r['pct_chg'] >= (lp * 100 - 1.2):
                if code in prev_limit:
                    continue  # 已涨停过，非首板
                trades.append((d, code, nm, r['close']))

    print(f"首板涨停样本: {len(trades)} 笔")

    results = []
    miss = 0
    for d, code, name, buy in trades:
        if pd.isna(buy):
            miss += 1; continue
        nxt = all_days[day_idx[d] + 1]
        ndf = daily.get(nxt)
        if ndf is None:
            miss += 1; continue
        row = ndf[ndf['ts_code'] == code]
        if row.empty:
            results.append((d, code, name, buy, None, 0.0, '次日停牌'))
            miss += 1; continue
        sell = float(row.iloc[0]['open'])
        gross = (sell - buy) / buy
        fee = 0.00025 * 2 + 0.001
        net = gross - fee
        results.append((d, code, name, buy, sell, net, ''))

    cols = ['buy_date','ts_code','name','buy','sell','ret','note']
    dfr = pd.DataFrame(results, columns=cols)
    valid = dfr[dfr['sell'].notna()].copy()

    print(f"\n=== 结果（首板涨停→次日开盘卖出）===")
    print(f"有效样本 {len(valid)} 笔；停牌/无数据 {miss} 笔")
    if valid.empty:
        print("无有效样本"); return
    print(f"平均收益: {valid['ret'].mean()*100:+.3f}%")
    print(f"中位数:   {valid['ret'].median()*100:+.3f}%")
    print(f"胜率(正收益): {(valid['ret']>0).mean()*100:.1f}%")
    print(f"最大盈利: {valid['ret'].max()*100:+.2f}%  最大亏损: {valid['ret'].min()*100:+.2f}%")
    print(f"累计(等权): {valid['ret'].sum()*100:+.2f}%")

    bins = [-10, -0.10, -0.05, 0, 0.05, 0.10, 10]
    labels = ['<-10%','-10~-5%','-5~0%','0~5%','5~10%','>10%']
    valid['bucket'] = pd.cut(valid['ret'], bins=bins, labels=labels)
    print("\n收益分布：")
    print(valid['bucket'].value_counts().reindex(labels).to_string())

    valid.to_csv('reports/limit_up_backtest_1mo.csv', index=False, encoding='utf-8-sig')
    print("\n明细已存 reports/limit_up_backtest_1mo.csv")

if __name__ == '__main__':
    main()
