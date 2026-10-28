"""
Tushare API 配置文件

Token 来源（按优先级）：
  1. 环境变量 TUSHARE_TOKEN
  2. 本地文件 config/tushare_token.local（不入版本控制）

说明：为避免 Token 泄露到公开仓库，不再在代码中硬编码 Token。
请通过环境变量或本地配置文件提供：
    # PowerShell
    $env:TUSHARE_TOKEN = "你的token"
    # 或写入 config/tushare_token.local（仅一行 token）
"""

import os
from pathlib import Path


def _load_token() -> str:
    """按优先级读取 Tushare Token。"""
    env_token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if env_token:
        return env_token

    local_file = Path(__file__).resolve().parent / "tushare_token.local"
    if local_file.exists():
        try:
            token = local_file.read_text(encoding="utf-8").strip()
            if token:
                return token
        except Exception:
            pass

    # 兜底：从项目根 config.yaml 的 tushare.token 读取（若已配置真实值）
    try:
        import yaml
        cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
        if cfg_path.exists():
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            token = str((cfg.get("tushare") or {}).get("token") or "").strip()
            if token and not token.startswith("${"):
                return token
    except Exception:
        pass

    return ""


TUSHARE_TOKEN = _load_token()

# API配置
TUSHARE_API_URL = "http://api.tushare.pro"
TUSHARE_TIMEOUT = 30

# 数据获取配置
# 股票基础信息
STOCK_BASIC_FIELDS = [
    "ts_code", "symbol", "name", "area", "industry", 
    "market", "list_date", "fullname", "enname", "cnspell"
]

# 日线行情
DAILY_FIELDS = [
    "ts_code", "trade_date", "open", "high", "low", "close",
    "pre_close", "change", "pct_chg", "vol", "amount"
]

# 财务指标
FINANCE_FIELDS = [
    "ts_code", "ann_date", "end_date", "eps", "bps", "roe",
    "pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_revenue",
    "revenue", "total_assets", "current_assets"
]

# 使用说明
"""
使用方法:
1. 导入配置: from config.tushare_config import TUSHARE_TOKEN
2. 初始化tushare: ts.set_token(TUSHARE_TOKEN)
3. 获取数据: pro = ts.pro_api()

示例:
import tushare as ts
from config.tushare_config import TUSHARE_TOKEN

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# 获取股票基础信息
df = pro.stock_basic(fields=STOCK_BASIC_FIELDS)

# 获取日线行情
df_daily = pro.daily(ts_code='000001.SZ', start_date='20260101', end_date='20260421')
"""

# 安全说明
"""
安全注意事项:
1. 此Token仅用于个人研究和学习
2. 不要将Token提交到公开仓库
3. 定期检查Token使用情况
4. 如有异常使用，及时报告船长
"""

print("Tushare配置加载成功")
print(f"Token: {TUSHARE_TOKEN[:10]}...{TUSHARE_TOKEN[-10:]}")
print("配置完成时间: 2026-04-21 14:05")