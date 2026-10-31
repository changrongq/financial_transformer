"""验证：能不能用"高换手股票 + 只做日内"赚到钱？

⚠️⚠️ 重要：本脚本的结论目前【不可信】，原因是底层成交额数据有问题。

知微审查第4条指出了前视信息问题（已修），但进一步诊断发现更严重的问题：

stock_data 的「成交额」字段：
  · 2024-10-11 ~ 2026-03-31 完全为空（18个月）
  · 兜底方案 close x volume 在多个时期严重偏离真实值：
      2025-05 满足"5000万"门槛的占 23.6%
      2025-07 满足的占 67.4%
      2025-09 满足的占 72.4%
      2025-12 又掉回 0.6%
      2026-02 只有 0.7%
  → 同一门槛下通过率从 0% 跳到 72% 再掉回 0.6%，
    说明 close x volume 不能代表真实成交额，流动性筛选完全失真。

因此：
  · 本文的 22.73bp / 41.58bp 等日内收益数字不可作为依据
  · 盈亏平衡成本（0.114% / 0.208%）同样不可靠
  · 需要先修好成交额数据（用 tushare daily 的 amount 字段重建）
    才能重做这个分析

已修正的口径问题（保留作为记录）：
  · 前视信息：流动性筛选改用【前一日】成交额（amount_lag）
  · 涨停判断：从 abs(gap)>=9.5% 改为 gap>=+9.5%（排除向下跳空）
  · 单位：明确 close x volume 得到的是【元】（成交量单位=股）

原始设计：
  换手率最高20%的股票隔夜更负、日内更正。诱人想法是只做日内。
  但个股日内成本（印花税+佣金+滑点）远高于 ETF，需验证成本可行性。
  同时需验证：高换手小票能否买到、优势是否稳定。

用法：
  python scripts/test_high_turnover_intraday.py
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def _resolve_db() -> str:
    config_path = ROOT / "config.yaml"
    if config_path.exists():
        import yaml
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("databases", {}).get("trade_db")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = ROOT / p
            return str(p.resolve())
    return str((ROOT.parent / "stock_history.db").resolve())


def load_data(db: str) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    df = pd.read_sql_query(
        "SELECT stock_code, [日期] AS date, [开盘] AS open, [收盘] AS close, "
        "[成交量] AS volume, [换手率] AS turnover, [成交额] AS amount "
        "FROM stock_data", conn)
    bi = pd.read_sql_query("SELECT stock_code, 名称 FROM stock_basic_info", conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stock_code"] = df["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    for c in ["open", "close", "volume", "turnover", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "close", "open"])
    df = df[(df["open"] > 0) & (df["close"] > 0)]
    df = df[df["stock_code"].str.len() == 6]
    df = df.sort_values(["stock_code", "date"]).reset_index(drop=True)

    # 成交额兜底
    g = df.groupby("stock_code")
    # 成交额：stock_data 的「成交额」字段有 18 个月空洞，用 close×volume 兜底
    # ⚠️ 单位：stock_data 的「成交量」单位是【股】（已验证 ratio=1.0），
    #    所以 close×volume 得到的是【元】。
    miss = int(df["amount"].isna().sum())
    if miss:
        df["amount"] = df["amount"].fillna(df["close"] * df["volume"])
    # ⚠️ 前视信息修正（知微审查第4条）：流动性筛选必须用【昨日】成交额
    g = df.groupby("stock_code")
    df["amount_lag"] = g["amount"].shift(1)

    df["prev_close"] = g["close"].shift(1)
    df["gap"] = df["open"] / df["prev_close"] - 1.0
    df["intraday"] = df["close"] / df["open"] - 1.0
    # 20日平均换手率（用前19天+今天，避免未来函数则用shift）
    df["turnover_ma20"] = g["turnover"].transform(lambda s: s.rolling(20, min_periods=10).mean())
    df["turnover_ma20_lag"] = g["turnover_ma20"].shift(1)  # 用昨天的，避免未来函数

    bi["stock_code"] = bi["stock_code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
    df = df.merge(bi[["stock_code", "名称"]], on="stock_code", how="left")
    return df.dropna(subset=["gap"]).reset_index(drop=True)


def main() -> None:
    df = load_data(_resolve_db())
    st = set(df.loc[df["名称"].astype(str).str.contains("ST|退", na=False), "stock_code"])
    df = df[~df["stock_code"].isin(st)]
    print(f"数据: {df['date'].min().date()} ~ {df['date'].max().date()}  "
          f"{df['stock_code'].nunique()} 只（已剔ST）")

    df = df.dropna(subset=["turnover_ma20_lag"]).reset_index(drop=True)

    # ===== T1: 高换手组日内收益 vs 成本 =====
    print("\n" + "=" * 96)
    print("【T1】高换手组「只做日内」的收益 vs 成本")
    print("=" * 96)

    # 每天按换手率排序，取最高的一组
    print(f"{'分组':<16}{'日均日内bp':>13}{'日均隔夜bp':>13}{'样本(股票日)':>14}")
    print("-" * 96)

    groups = {}
    for label, lo, hi in [("换手率最高10%", 0.90, 1.00), ("换手率最高20%", 0.80, 1.00),
                          ("换手率最高30%", 0.70, 1.00), ("全市场", 0.0, 1.00)]:
        rows = []
        for d, sub in df.groupby("date"):
            sub = sub.dropna(subset=["turnover_ma20_lag"])
            if len(sub) < 100:
                continue
            qlo = sub["turnover_ma20_lag"].quantile(lo)
            qhi = sub["turnover_ma20_lag"].quantile(hi)
            sel = sub[(sub["turnover_ma20_lag"] >= qlo) & (sub["turnover_ma20_lag"] <= qhi)]
            # ✅ 流动性过滤：用【前一日】成交额，避免前视信息（知微审查第4条）
            #    当天完整成交额要收盘后才知道，不能用来做开盘时的选股。
            #    单位：amount_lag 已经是 元（收盘价×成交量，成交量单位=股）
            sel = sel[sel["amount_lag"] >= 5e7]   # 5000万元
            if len(sel) < 10:
                continue
            rows.append({"date": d, "intra": sel["intraday"].mean(),
                         "gap": sel["gap"].mean(), "n": len(sel)})
        r = pd.DataFrame(rows)
        if r.empty:
            continue
        groups[label] = r
        print(f"{label:<16}{r['intra'].mean()*10000:>+12.2f}{r['gap'].mean()*10000:>+12.2f}"
              f"{int(r['n'].mean()):>14}")

    print("-" * 96)

    # 成本测试
    print("\n  成本测算（高换手最高20%组，只做日内）：")
    r = groups.get("换手率最高20%")
    if r is not None:
        daily_bp = r["intra"].mean() * 10000
        print(f"    日内毛收益: {daily_bp:+.2f} bp/天")
        print(f"\n    {'单边成本':<14}{'每日净(bp)':>14}{'年化(244天)':>16}{'结论':>12}")
        print("    " + "-" * 58)
        cost_rows = []
        for cost_pct in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
            net_bp = daily_bp - cost_pct * 2 * 100  # 单边成本转bp，买卖2次
            ann = net_bp / 10000 * 244 * 100
            note = "赚" if net_bp > 0 else "亏"
            print(f"    {cost_pct:>6.2f}%       {net_bp:>+13.2f}{ann:>+15.1f}%{note:>12}")
            cost_rows.append({"cost_one_way_pct": cost_pct, "net_bp": round(net_bp, 2),
                              "ann_pct": round(ann, 1)})

        # 盈亏平衡成本
        breakeven = daily_bp / (2 * 100)
        print(f"\n    盈亏平衡单边成本: {breakeven:.3f}%")
        print(f"    A股个股实际成本估算: 印花税0.05%(卖) + 佣金0.025%×2 + 滑点0.1~0.3%")
        print(f"    → 单边综合约 0.10%~0.20%（高换手小票滑点更大）")
        if breakeven > 0.15:
            print(f"    → 临界点 {breakeven:.3f}% 高于成本，**理论可行**")
        else:
            print(f"    → 临界点 {breakeven:.3f}% 低于/接近成本，**不可行**")

    # ===== T2: 涨停买不进的影响 =====
    print("\n" + "=" * 96)
    print("【T2】高换手股票的可交易性")
    print("=" * 96)
    r20 = groups.get("换手率最高20%")
    if r20 is not None:
        # 统计每日选中的股票后续是否涨停
        limit_up_days = 0
        total_days = 0
        for d, sub in df.groupby("date"):
            sub = sub.dropna(subset=["turnover_ma20_lag"])
            if len(sub) < 100:
                continue
            qlo = sub["turnover_ma20_lag"].quantile(0.80)
            sel = sub[(sub["turnover_ma20_lag"] >= qlo) & (sub["amount_lag"] >= 5e7)]
            if len(sel) < 10:
                continue
            total_days += 1
            # ✅ 修正（知微审查第4条）：
            #    旧实现用 abs(gap)>=9.5%，把【向下跳空】也算成"买不进"，这是错的。
            #    买不进只对应【向上】跳空到涨停。
            #    这里改为：向上跳空 >= +9.5%（近似涨停开盘，未区分板块/日期/真实涨停价）。
            #    ⚠️ 仍是粗糙代理：
            #      · 创业板/科创板 涨停是 20%，主板 ST 是 5%
            #      · 未考虑卖单排队的实际成交可能
            if (sel["gap"] >= 0.095).mean() > 0.02:
                limit_up_days += 1
        if total_days:
            print(f"  在 {total_days} 个交易日中，有 {limit_up_days} 天该组超过2%的股票")
            print(f"  开盘向上跳空 >= +9.5%（近似涨停开盘，买不进）")
            print(f"  占比 {limit_up_days/total_days*100:.1f}%")
            print(f"  ⚠️ 这是粗糙代理：未区分板块涨停幅度（创业板/科创板20%、ST股5%），")
            print(f"     也未考虑卖单排队的实际成交可能。真实买不进比例需分钟级数据确认。")

    # ===== T3: 稳定性（前后半段）=====
    print("\n" + "=" * 96)
    print("【T3】稳定性：高换手组日内收益，前后半段是否一致？")
    print("=" * 96)
    if r20 is not None:
        dates = sorted(r20["date"].unique())
        mid = dates[len(dates) // 2]
        h1 = r20[r20["date"] < mid]["intra"].mean() * 10000
        h2 = r20[r20["date"] >= mid]["intra"].mean() * 10000
        print(f"  前半段: {h1:+.2f} bp/天")
        print(f"  后半段: {h2:+.2f} bp/天")
        if h1 > 0 and h2 > 0:
            print(f"  → 两段都为正，{'较稳定' if min(h1,h2) > h1*0.5 else '但衰减明显'}")
        else:
            print(f"  → 不稳定（存在负值段）")

    # ===== 结论 =====
    print("\n" + "=" * 96)
    print("结论")
    print("=" * 96)
    if r20 is not None:
        daily_bp = r20["intra"].mean() * 10000
        breakeven = daily_bp / 200
        print(f"  1. 高换手20%组日内毛收益: {daily_bp:+.2f} bp/天（约年化 {daily_bp/10000*244*100:+.0f}%）")
        print(f"  2. 盈亏平衡单边成本: {breakeven:.3f}%")
        print(f"  3. A股个股实际单边成本: 约 0.10%~0.20%")
        print(f"  4. 是否可行: {'勉强可行' if breakeven > 0.15 else '基本不可行' if breakeven < 0.10 else '边缘'}")
        print()
        print(f"  对比 ETF 版本：")
        print(f"    - ETF 日内策略临界点 0.032%，实际成本 0.03-0.05% → 边缘")
        print(f"    - 个股日内策略临界点 {breakeven:.3f}%，实际成本 0.10-0.20% → {'边缘' if breakeven > 0.10 else '更差'}")
        print(f"    - 且个股还需承担：涨停买不进、退市风险、滑点更大")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "high_turnover_intraday.json"
    out.write_text(json.dumps({
        "groups_stats": {k: {"intra_bp": round(float(v["intra"].mean()*10000), 2),
                             "gap_bp": round(float(v["gap"].mean()*10000), 2),
                             "avg_n": int(v["n"].mean())}
                         for k, v in groups.items()},
        "cost_analysis": cost_rows if r20 is not None else [],
        "breakeven_cost_pct": round(breakeven, 4) if r20 is not None else None,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
