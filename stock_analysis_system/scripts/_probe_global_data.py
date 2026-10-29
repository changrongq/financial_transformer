"""探测美股/港股数据可用性（用于验证隔夜/日内规律是否全球普适）。"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for k in list(os.environ):
    if "PROXY" in k.upper():
        os.environ.pop(k, None)


def get_token() -> str:
    t = os.environ.get("TUSHARE_TOKEN")
    if t:
        return t.strip()
    p = ROOT / "config" / "tushare_token.local"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    raise SystemExit("无 token")


def main() -> None:
    import tushare as ts
    pro = ts.pro_api(get_token())

    print("=" * 76)
    print("1. 尝试 tushare 的美股接口")
    print("=" * 76)
    for fn, kw in [
        ("us_daily", {"ts_code": "SPY"}),
        ("us_daily", {"ts_code": "QQQ"}),
        ("us_basic", {"ts_code": "SPY"}),
    ]:
        try:
            f = getattr(pro, fn, None)
            if f is None:
                print(f"  {fn}: 接口不存在")
                continue
            df = f(**kw)
            if df is None or df.empty:
                print(f"  {fn}{kw}: 返回空")
            else:
                print(f"  {fn}{kw}: [OK] {len(df)} 行")
                print(f"    列: {list(df.columns)}")
                print(df.head(2).to_string(index=False))
        except Exception as e:
            print(f"  {fn}{kw}: [X] {type(e).__name__}: {str(e)[:100]}")

    print()
    print("=" * 76)
    print("2. 尝试港股接口")
    print("=" * 76)
    for fn, kw in [
        ("hk_daily", {"ts_code": "00700.HK"}),
        ("hk_daily", {"ts_code": "02800.HK"}),
    ]:
        try:
            f = getattr(pro, fn, None)
            if f is None:
                print(f"  {fn}: 接口不存在")
                continue
            df = f(**kw)
            if df is None or df.empty:
                print(f"  {fn}{kw}: 返回空")
            else:
                print(f"  {fn}{kw}: [OK] {len(df)} 行, 列: {list(df.columns)}")
                print(df.head(2).to_string(index=False))
        except Exception as e:
            print(f"  {fn}{kw}: [X] {type(e).__name__}: {str(e)[:100]}")

    print()
    print("=" * 76)
    print("3. 检查本地是否已装 yfinance / akshare")
    print("=" * 76)
    for mod in ["yfinance", "akshare", "pandas_datareader"]:
        try:
            m = __import__(mod)
            print(f"  {mod}: [OK] {getattr(m, '__version__', '?')}")
        except ImportError:
            print(f"  {mod}: 未安装")


if __name__ == "__main__":
    main()
