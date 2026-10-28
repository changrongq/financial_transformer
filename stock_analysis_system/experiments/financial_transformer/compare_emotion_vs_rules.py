"""「无情绪 vs 情绪化」三方对照实验。

回答豆包提出的核心问题：
  「情绪这个演化出来的内置奖励系统，在交易这个场景里，到底是优势，还是包袱？」

三个智能体（同一套行情、同一套成本）：
  1. RULE 规则型（无情绪·有智慧）：大盘 10 日均收益 < 0 → 空仓，否则满仓
  2. CONST 恒定型（无情绪·不思考）：恒定 50% 仓位
  3. EMOTION 情绪化（恐惧/贪婪驱动）：
     - 贪婪：连续盈利后加仓（越赚越自信）
     - 恐惧/损失厌恶：连续亏损后减仓/离场（怕继续亏）
     - 追涨杀跌：单日大涨加仓、单日大跌割肉

评价指标（豆包建议）：
  收益率、最大回撤、胜率、盈亏比、连续亏损时的行为偏移

用法：
  python experiments/financial_transformer/compare_emotion_vs_rules.py
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]

POSITION_STEP = 0.25  # 情绪化策略每次加减仓幅度
COST = 0.0005         # 单边交易成本（ETF 近似）


def _resolve_db() -> str:
    config_path = PROJECT_ROOT / "config.yaml"
    if config_path.exists():
        import yaml
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        raw = cfg.get("databases", {}).get("trade_db")
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            return str(p.resolve())
    return str((PROJECT_ROOT.parent / "stock_history.db").resolve())


def load_index(db: str) -> pd.DataFrame:
    conn = sqlite3.connect(db)
    df = pd.read_sql_query("SELECT date, close, pct_chg FROM index_daily_sh ORDER BY date", conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["ret"] = pd.to_numeric(df["pct_chg"], errors="coerce") / 100.0
    df = df.dropna(subset=["ret"]).reset_index(drop=True)
    df["regime10"] = df["ret"].rolling(10, min_periods=3).mean()
    return df.dropna(subset=["regime10"]).reset_index(drop=True)


def run_agent(df: pd.DataFrame, decide_fn, name: str) -> dict:
    """通用回测：每步调 decide_fn(history_state) 得到目标仓位。"""
    position = 0.0
    equity = 1.0
    peak = 1.0
    equity_curve = [1.0]
    positions = []
    daily_rets = []
    win_count = 0
    loss_count = 0
    # 行为追踪
    consec_loss = 0
    consec_win = 0
    pos_after_losses = []  # 连续亏损后的仓位（测损失厌恶）
    pos_after_wins = []    # 连续盈利后的仓位（测贪婪）

    state = {
        "consec_loss": 0,
        "consec_win": 0,
        "position": 0.5,
        "flat_days": 0,
    }

    for i in range(len(df)):
        row = df.iloc[i]
        target = decide_fn(row, i, df, state)
        target = float(np.clip(target, 0.0, 1.0))

        turnover = abs(target - position)
        cost = turnover * COST
        day_ret = target * row["ret"] - cost

        equity *= (1.0 + day_ret)
        peak = max(peak, equity)
        equity_curve.append(equity)
        positions.append(target)
        daily_rets.append(day_ret)

        if day_ret > 0:
            win_count += 1
            state["consec_win"] += 1
            state["consec_loss"] = 0
            pos_after_wins.append(target)
        else:
            loss_count += 1
            state["consec_loss"] += 1
            state["consec_win"] = 0
            pos_after_losses.append(target)

        position = target

    rets = pd.Series(daily_rets)
    cum = np.cumprod(1.0 + rets.values)
    mdd = float((cum / np.maximum.accumulate(cum) - 1).min())

    # 盈亏比
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    profit_loss_ratio = float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else None

    return {
        "name": name,
        "total_return": float(equity - 1.0),
        "max_drawdown": mdd,
        "win_rate": float(win_count / max(win_count + loss_count, 1)),
        "profit_loss_ratio": profit_loss_ratio,
        "avg_position": float(np.mean(positions)),
        "position_std": float(np.std(positions)),
        "days": len(df),
        # 行为特征
        "avg_pos_after_2losses": float(np.mean(pos_after_losses[-100:])) if len(pos_after_losses) >= 10 else None,
        "avg_pos_after_2wins": float(np.mean(pos_after_wins[-100:])) if len(pos_after_wins) >= 10 else None,
        "equiv_curve": [round(v, 6) for v in equity_curve[::20]],  # 采样用于展示
    }


# ============ 三个智能体的决策函数 ============

def decide_rule(row, i, df, state):
    """规则型（无情绪·有智慧）：大盘10日均收益<0空仓。"""
    return 1.0 if row["regime10"] >= 0 else 0.0


def decide_const(row, i, df, state):
    """恒定型（无情绪·不思考）：恒定50%。"""
    return 0.5


def decide_emotion(row, i, df, state):
    """情绪化（恐惧/贪婪驱动）—— 更真实的散户"情绪循环"。

    关键修正：割肉后不会永久离场，会在市场大涨时 FOMO 追回来。
    这模拟了真实散户"割肉→踏空→追高→再套牢"的循环。

    行为规则：
      - 损失厌恶：连续亏损 → 减仓（但恐惧会随空仓时间衰减）
      - 贪婪：连续盈利 → 加仓（赚了就自信膨胀）
      - FOMO 追涨：市场大涨 >1.5% 且自己仓位低 → 追进去
      - 恐慌割肉：单日大跌 < -2% → 减仓
      - 踏空焦虑：空仓期间市场连续上涨 → 忍不住追
    """
    base = state.get("position", 0.5)
    flat_days = state.get("flat_days", 0)

    # 1) 损失厌恶：连续亏损减仓
    if state["consec_loss"] >= 3:
        base -= POSITION_STEP * 2
    elif state["consec_loss"] == 2:
        base -= POSITION_STEP

    # 2) 贪婪：连续盈利加仓
    if state["consec_win"] >= 2:
        base += POSITION_STEP
    if state["consec_win"] >= 4:
        base += POSITION_STEP  # 极度自信

    # 3) FOMO 追涨（散户最典型）：大涨时情绪被点燃
    if row["ret"] > 0.015:
        base += POSITION_STEP * 1.5
    elif row["ret"] > 0.03:
        base += POSITION_STEP * 2

    # 4) 恐慌割肉：大跌时恐惧主导
    if row["ret"] < -0.02:
        base -= POSITION_STEP * 1.5
    elif row["ret"] < -0.04:
        base -= POSITION_STEP * 2

    # 5) 踏空焦虑：空仓久了、市场还在涨 → 忍不住追
    if flat_days >= 5 and row["regime10"] > 0:
        base += POSITION_STEP * 2  # 踏空比亏钱还难受

    target = float(np.clip(base, 0.0, 1.0))
    state["position"] = target
    state["flat_days"] = flat_days + 1 if target < 0.1 else 0
    return target


def main() -> None:
    db = _resolve_db()
    df = load_index(db)
    print(f"数据: {df['date'].min().date()} ~ {df['date'].max().date()}, {len(df)} 个交易日")

    results = []
    for fn, name in [
        (decide_rule, "RULE 规则型(无情绪·有智慧)"),
        (decide_const, "CONST 恒定型(无情绪·不思考)"),
        (decide_emotion, "EMOTION 情绪化(恐惧贪婪)"),
    ]:
        r = run_agent(df, fn, name)
        results.append(r)

    # 满仓基准
    bh = float(np.prod(1.0 + df["ret"].values) - 1.0)
    cum_bh = np.cumprod(1.0 + df["ret"].values)
    bh_mdd = float((cum_bh / np.maximum.accumulate(cum_bh) - 1).min())

    print("\n" + "=" * 78)
    print("三方对照结果（2019-2026 全期）")
    print("=" * 78)
    print(f"{'智能体':<28}{'总收益':>10}{'最大回撤':>10}{'胜率':>8}{'盈亏比':>8}{'平均仓位':>10}")
    print(f"{'满仓持有(基准)':<28}{bh*100:>9.1f}%{bh_mdd*100:>9.1f}%{'':>8}{'':>8}{'100%':>10}")
    for r in results:
        plr = f"{r['profit_loss_ratio']:.2f}" if r["profit_loss_ratio"] else "N/A"
        print(f"{r['name']:<28}{r['total_return']*100:>9.1f}%{r['max_drawdown']*100:>9.1f}%{r['win_rate']*100:>7.1f}%{plr:>8}{r['avg_position']*100:>9.0f}%")

    # 逐年对比
    print("\n" + "=" * 78)
    print("逐年一致性检验")
    print("=" * 78)
    print(f"{'年份':<8}{'RULE':>12}{'CONST':>12}{'EMOTION':>12}{'满仓':>12}")
    yearly = {}
    for year in sorted(df["date"].dt.year.unique()):
        sub = df[df["date"].dt.year == year].reset_index(drop=True)
        if len(sub) < 60:
            continue
        yr = {}
        for fn, name in [(decide_rule, "RULE"), (decide_const, "CONST"), (decide_emotion, "EMOTION")]:
            r = run_agent(sub, fn, name)
            yr[name] = r["total_return"]
        yr["buy_hold"] = float(np.prod(1.0 + sub["ret"].values) - 1.0)
        yearly[int(year)] = yr
        print(f"{year:<8}{yr['RULE']*100:>11.1f}%{yr['CONST']*100:>11.1f}%{yr['EMOTION']*100:>11.1f}%{yr['buy_hold']*100:>11.1f}%")

    # 统计
    if yearly:
        rule_win = sum(1 for y in yearly.values() if y["RULE"] > y["EMOTION"])
        const_win = sum(1 for y in yearly.values() if y["CONST"] > y["EMOTION"])
        print(f"\n  RULE 跑赢 EMOTION: {rule_win}/{len(yearly)} 年")
        print(f"  CONST 跑赢 EMOTION: {const_win}/{len(yearly)} 年")

    # 行为特征
    print("\n" + "=" * 78)
    print("行为特征（回答豆包的问题：情绪是不是包袱）")
    print("=" * 78)
    for r in results:
        if r["name"].startswith("EMOTION"):
            print(f"  情绪化：亏损后平均仓位 {r['avg_pos_after_2losses']*100:.0f}%" if r["avg_pos_after_2losses"] else "  情绪化：亏损后数据不足")
            print(f"           盈利后平均仓位 {r['avg_pos_after_2wins']*100:.0f}%" if r["avg_pos_after_2wins"] else "           盈利后数据不足")
        if r["name"].startswith("CONST"):
            print(f"  恒定型：仓位标准差 {r['position_std']*100:.0f}%（不思考=不动）")
        if r["name"].startswith("RULE"):
            print(f"  规则型：仓位标准差 {r['position_std']*100:.0f}%（按规则切换）")

    out = ROOT / "results" / "emotion_vs_rules_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "buy_hold": {"total_return": bh, "max_drawdown": bh_mdd},
        "agents": [{k: v for k, v in r.items() if k != "equiv_curve"} for r in results],
        "yearly": yearly,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入: {out}")


if __name__ == "__main__":
    main()
