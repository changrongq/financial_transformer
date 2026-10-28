from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sqlite3

import pandas as pd

from .config import SystemConfig
from .logging_utils import setup_logger
from .market_cap import normalize_ts_code
from .trade_data import fetch_realtime_quotes

logger = setup_logger("hierarchical.paper_trading")

DEFAULT_COMMISSION_RATE = 0.0003
DEFAULT_STAMP_DUTY_RATE = 0.001
DEFAULT_MIN_COMMISSION = 5.0
DEFAULT_BOARD_LOT = 100


def get_paper_trading_db_path(config: SystemConfig) -> Path:
    return config.results_dir / "paper_trading.db"


def ensure_paper_trading_schema(db_path: str | Path) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_account (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                initial_cash REAL NOT NULL,
                initial_equity REAL NOT NULL,
                cash REAL NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(paper_account)").fetchall()}
        if "initial_equity" not in columns:
            conn.execute("ALTER TABLE paper_account ADD COLUMN initial_equity REAL")
            conn.execute("UPDATE paper_account SET initial_equity = initial_cash WHERE initial_equity IS NULL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_positions (
                stock_code TEXT PRIMARY KEY,
                ts_code TEXT NOT NULL,
                name TEXT,
                quantity INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                last_trade_price REAL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_time TEXT NOT NULL,
                side TEXT NOT NULL,
                stock_code TEXT NOT NULL,
                ts_code TEXT NOT NULL,
                name TEXT,
                price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                gross_amount REAL NOT NULL,
                commission REAL NOT NULL,
                stamp_duty REAL NOT NULL,
                net_amount REAL NOT NULL,
                realized_pnl REAL NOT NULL,
                note TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_paper_trades_time
            ON paper_trades(trade_time DESC)
            """
        )
        conn.commit()
    finally:
        conn.close()


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_stock_code(stock_code: str) -> str:
    return str(stock_code).split(".")[0].zfill(6)


def _normalize_stock_code_list(stock_codes: list[str] | tuple[str, ...] | str) -> list[str]:
    raw_codes = stock_codes.split(",") if isinstance(stock_codes, str) else list(stock_codes)
    normalized_codes: list[str] = []
    seen: set[str] = set()
    for raw_code in raw_codes:
        code = _normalize_stock_code(str(raw_code).strip())
        if not code or code in seen:
            continue
        seen.add(code)
        normalized_codes.append(code)
    if not normalized_codes:
        raise ValueError("至少提供一个股票代码。")
    return normalized_codes


def _normalize_position_inputs(positions: list[dict] | None) -> list[dict]:
    normalized: list[dict] = []
    for item in positions or []:
        stock_code = _normalize_stock_code(item["stock_code"])
        quantity = int(item["quantity"])
        if quantity <= 0:
            raise ValueError(f"初始持仓数量必须大于 0: {stock_code}")
        avg_cost = item.get("avg_cost")
        if avg_cost is not None:
            avg_cost = float(avg_cost)
            if avg_cost <= 0:
                raise ValueError(f"初始持仓成本价必须大于 0: {stock_code}")
        normalized.append(
            {
                "stock_code": stock_code,
                "quantity": quantity,
                "avg_cost": avg_cost,
            }
        )
    return normalized


def _load_account_row(conn: sqlite3.Connection) -> tuple[float, float, float, str, str]:
    row = conn.execute(
        "SELECT initial_cash, COALESCE(initial_equity, initial_cash), cash, created_at, updated_at FROM paper_account WHERE id = 1"
    ).fetchone()
    if row is None:
        raise RuntimeError("模拟盘账户尚未初始化，请先调用 /paper-trading/account/init。")
    return float(row[0]), float(row[1]), float(row[2]), str(row[3]), str(row[4])


def _load_positions_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT stock_code, ts_code, name, quantity, avg_cost, last_trade_price, updated_at FROM paper_positions ORDER BY stock_code",
        conn,
    )


def _load_trades_frame(conn: sqlite3.Connection, limit: int | None = None) -> pd.DataFrame:
    query = (
        "SELECT id, trade_time, side, stock_code, ts_code, name, price, quantity, gross_amount, commission, stamp_duty, "
        "net_amount, realized_pnl, note FROM paper_trades ORDER BY id DESC"
    )
    params: list[int] = []
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))
    return pd.read_sql_query(query, conn, params=params)


def _get_quote_frame(config: SystemConfig, stock_codes: list[str], src: str) -> pd.DataFrame:
    if not stock_codes:
        return pd.DataFrame()
    frame = fetch_realtime_quotes(config, stock_codes, src=src)
    if "stock_code" in frame.columns:
        frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
    return frame


def _validate_limit_price(side: str, trade_price: float, limit_price: float | None, stock_code: str) -> None:
    if limit_price is None:
        return
    expected_price = float(limit_price)
    if expected_price <= 0:
        raise ValueError("limit_price 必须大于 0。")
    if side == "BUY" and trade_price > expected_price:
        raise ValueError(f"买入价格保护触发: {stock_code} 实时价 {trade_price:.3f} 高于限价 {expected_price:.3f}。")
    if side == "SELL" and trade_price < expected_price:
        raise ValueError(f"卖出价格保护触发: {stock_code} 实时价 {trade_price:.3f} 低于限价 {expected_price:.3f}。")


def _commission(amount: float, commission_rate: float, min_commission: float) -> float:
    if amount <= 0:
        return 0.0
    return max(float(amount) * float(commission_rate), float(min_commission))


def _buy_quantity_from_budget(
    budget: float,
    price: float,
    commission_rate: float,
    min_commission: float,
    board_lot: int,
) -> int:
    lot = max(int(board_lot), 1)
    quantity = int(budget // price)
    quantity = (quantity // lot) * lot
    while quantity > 0:
        gross_amount = price * quantity
        total_cost = gross_amount + _commission(gross_amount, commission_rate, min_commission)
        if total_cost <= budget:
            return quantity
        quantity -= lot
    return 0


def initialize_account(
    config: SystemConfig,
    initial_cash: float,
    positions: list[dict] | None = None,
    reset: bool = True,
    quote_source: str = "auto",
) -> dict:
    if initial_cash < 0:
        raise ValueError("initial_cash 不能小于 0。")

    normalized_positions = _normalize_position_inputs(positions)
    db_path = get_paper_trading_db_path(config)
    ensure_paper_trading_schema(db_path)
    quote_frame = _get_quote_frame(config, [item["stock_code"] for item in normalized_positions], src=quote_source)
    quote_map = {
        str(row["stock_code"]): row
        for _, row in quote_frame.iterrows()
    } if not quote_frame.empty else {}
    initial_equity = float(initial_cash)

    now_text = _now_text()
    conn = sqlite3.connect(db_path)
    try:
        if reset:
            conn.execute("DELETE FROM paper_trades")
            conn.execute("DELETE FROM paper_positions")
            conn.execute("DELETE FROM paper_account")

        conn.execute(
            "INSERT OR REPLACE INTO paper_account(id, initial_cash, initial_equity, cash, created_at, updated_at) VALUES (1, ?, ?, ?, ?, ?)",
            (float(initial_cash), 0.0, float(initial_cash), now_text, now_text),
        )

        for item in normalized_positions:
            stock_code = item["stock_code"]
            quote_row = quote_map.get(stock_code)
            price = float(item["avg_cost"] if item["avg_cost"] is not None else (quote_row.get("price") if quote_row is not None else 0.0))
            if price <= 0:
                raise ValueError(f"无法确定初始持仓成本价: {stock_code}")
            ts_code = str(quote_row.get("ts_code") if quote_row is not None else normalize_ts_code(stock_code))
            name = str(quote_row.get("name") if quote_row is not None and pd.notna(quote_row.get("name")) else "")
            conn.execute(
                "INSERT OR REPLACE INTO paper_positions(stock_code, ts_code, name, quantity, avg_cost, last_trade_price, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (stock_code, ts_code, name, int(item["quantity"]), float(price), float(price), now_text),
            )
            gross_amount = float(price) * int(item["quantity"])
            conn.execute(
                "INSERT INTO paper_trades(trade_time, side, stock_code, ts_code, name, price, quantity, gross_amount, commission, stamp_duty, net_amount, realized_pnl, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (now_text, "INIT", stock_code, ts_code, name, float(price), int(item["quantity"]), gross_amount, 0.0, 0.0, gross_amount, 0.0, "initial position"),
            )
            initial_equity += gross_amount
        conn.execute(
            "UPDATE paper_account SET initial_equity = ? WHERE id = 1",
            (initial_equity,),
        )
        conn.commit()
    finally:
        conn.close()

    return get_account_snapshot(config, src=quote_source)


def _position_row(conn: sqlite3.Connection, stock_code: str) -> tuple | None:
    return conn.execute(
        "SELECT stock_code, ts_code, name, quantity, avg_cost, last_trade_price FROM paper_positions WHERE stock_code = ?",
        (_normalize_stock_code(stock_code),),
    ).fetchone()


def place_buy_order(
    config: SystemConfig,
    stock_code: str,
    quantity: int | None = None,
    cash_amount: float | None = None,
    position_fraction: float | None = None,
    limit_price: float | None = None,
    src: str = "auto",
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    min_commission: float = DEFAULT_MIN_COMMISSION,
    board_lot: int = DEFAULT_BOARD_LOT,
    note: str | None = None,
) -> dict:
    normalized_code = _normalize_stock_code(stock_code)
    quote_frame = _get_quote_frame(config, [normalized_code], src=src)
    if quote_frame.empty:
        raise RuntimeError(f"未获取到实时行情: {normalized_code}")
    quote_row = quote_frame.iloc[0]
    trade_price = float(pd.to_numeric(quote_row.get("price"), errors="coerce") or 0.0)
    if trade_price <= 0:
        raise RuntimeError(f"实时价格无效，无法买入: {normalized_code}")
    _validate_limit_price("BUY", trade_price, limit_price, normalized_code)

    db_path = get_paper_trading_db_path(config)
    ensure_paper_trading_schema(db_path)
    now_text = _now_text()

    conn = sqlite3.connect(db_path)
    try:
        initial_cash, _, cash, _, _ = _load_account_row(conn)
        requested_quantity = int(quantity) if quantity is not None else None
        if requested_quantity is not None and requested_quantity <= 0:
            raise ValueError("quantity 必须大于 0。")
        if cash_amount is not None and cash_amount <= 0:
            raise ValueError("cash_amount 必须大于 0。")
        if position_fraction is not None and not (0 < position_fraction <= 1):
            raise ValueError("position_fraction 必须在 (0, 1] 范围内。")

        if requested_quantity is None:
            budget = float(cash_amount) if cash_amount is not None else float(cash) * float(position_fraction or 1.0)
            budget = min(float(cash), budget)
            requested_quantity = _buy_quantity_from_budget(
                budget=budget,
                price=trade_price,
                commission_rate=commission_rate,
                min_commission=min_commission,
                board_lot=board_lot,
            )
        else:
            requested_quantity = int(requested_quantity)

        if requested_quantity <= 0:
            raise ValueError("可买数量为 0，请提高预算或直接指定 quantity。")

        gross_amount = trade_price * requested_quantity
        commission = _commission(gross_amount, commission_rate, min_commission)
        total_cost = gross_amount + commission
        if total_cost > cash:
            raise ValueError(f"现金不足，所需 {total_cost:.2f}，可用 {cash:.2f}。")

        existing = _position_row(conn, normalized_code)
        if existing is None:
            new_quantity = requested_quantity
            new_avg_cost = total_cost / new_quantity
        else:
            old_quantity = int(existing[3])
            old_avg_cost = float(existing[4])
            new_quantity = old_quantity + requested_quantity
            new_avg_cost = ((old_avg_cost * old_quantity) + total_cost) / new_quantity

        new_cash = float(cash) - total_cost
        conn.execute(
            "UPDATE paper_account SET cash = ?, updated_at = ? WHERE id = 1",
            (new_cash, now_text),
        )
        conn.execute(
            "INSERT OR REPLACE INTO paper_positions(stock_code, ts_code, name, quantity, avg_cost, last_trade_price, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                normalized_code,
                str(quote_row.get("ts_code") or normalize_ts_code(normalized_code)),
                str(quote_row.get("name") or ""),
                new_quantity,
                float(new_avg_cost),
                trade_price,
                now_text,
            ),
        )
        conn.execute(
            "INSERT INTO paper_trades(trade_time, side, stock_code, ts_code, name, price, quantity, gross_amount, commission, stamp_duty, net_amount, realized_pnl, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                now_text,
                "BUY",
                normalized_code,
                str(quote_row.get("ts_code") or normalize_ts_code(normalized_code)),
                str(quote_row.get("name") or ""),
                trade_price,
                requested_quantity,
                gross_amount,
                commission,
                0.0,
                total_cost,
                0.0,
                note or "",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    snapshot = get_account_snapshot(config, src=src)
    snapshot["order"] = {
        "side": "BUY",
        "stock_code": normalized_code,
        "price": trade_price,
        "limit_price": float(limit_price) if limit_price is not None else None,
        "quote_timestamp": str(quote_row.get("timestamp") or ""),
        "quantity": requested_quantity,
        "gross_amount": gross_amount,
        "commission": commission,
        "net_amount": total_cost,
        "cash_after": snapshot["summary"]["cash"],
        "initial_cash": initial_cash,
    }
    return snapshot


def place_sell_order(
    config: SystemConfig,
    stock_code: str,
    quantity: int | None = None,
    position_fraction: float | None = None,
    limit_price: float | None = None,
    src: str = "auto",
    commission_rate: float = DEFAULT_COMMISSION_RATE,
    min_commission: float = DEFAULT_MIN_COMMISSION,
    stamp_duty_rate: float = DEFAULT_STAMP_DUTY_RATE,
    note: str | None = None,
) -> dict:
    normalized_code = _normalize_stock_code(stock_code)
    db_path = get_paper_trading_db_path(config)
    ensure_paper_trading_schema(db_path)

    quote_frame = _get_quote_frame(config, [normalized_code], src=src)
    if quote_frame.empty:
        raise RuntimeError(f"未获取到实时行情: {normalized_code}")
    quote_row = quote_frame.iloc[0]
    trade_price = float(pd.to_numeric(quote_row.get("price"), errors="coerce") or 0.0)
    if trade_price <= 0:
        raise RuntimeError(f"实时价格无效，无法卖出: {normalized_code}")
    _validate_limit_price("SELL", trade_price, limit_price, normalized_code)

    now_text = _now_text()
    conn = sqlite3.connect(db_path)
    try:
        initial_cash, _, cash, _, _ = _load_account_row(conn)
        existing = _position_row(conn, normalized_code)
        if existing is None:
            raise ValueError(f"当前无持仓，无法卖出: {normalized_code}")

        holding_quantity = int(existing[3])
        avg_cost = float(existing[4])
        if quantity is not None:
            sell_quantity = int(quantity)
        else:
            fraction = float(position_fraction if position_fraction is not None else 1.0)
            if not (0 < fraction <= 1):
                raise ValueError("position_fraction 必须在 (0, 1] 范围内。")
            sell_quantity = max(int(holding_quantity * fraction), 1)

        if sell_quantity <= 0:
            raise ValueError("quantity 必须大于 0。")
        if sell_quantity > holding_quantity:
            raise ValueError(f"卖出数量超过持仓，可卖 {holding_quantity}，请求 {sell_quantity}。")

        gross_amount = trade_price * sell_quantity
        commission = _commission(gross_amount, commission_rate, min_commission)
        stamp_duty = gross_amount * float(stamp_duty_rate)
        net_amount = gross_amount - commission - stamp_duty
        realized_pnl = net_amount - (avg_cost * sell_quantity)
        new_cash = float(cash) + net_amount
        remaining_quantity = holding_quantity - sell_quantity

        conn.execute(
            "UPDATE paper_account SET cash = ?, updated_at = ? WHERE id = 1",
            (new_cash, now_text),
        )
        if remaining_quantity > 0:
            conn.execute(
                "UPDATE paper_positions SET quantity = ?, last_trade_price = ?, updated_at = ? WHERE stock_code = ?",
                (remaining_quantity, trade_price, now_text, normalized_code),
            )
        else:
            conn.execute("DELETE FROM paper_positions WHERE stock_code = ?", (normalized_code,))
        conn.execute(
            "INSERT INTO paper_trades(trade_time, side, stock_code, ts_code, name, price, quantity, gross_amount, commission, stamp_duty, net_amount, realized_pnl, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                now_text,
                "SELL",
                normalized_code,
                str(quote_row.get("ts_code") or normalize_ts_code(normalized_code)),
                str(quote_row.get("name") or existing[2] or ""),
                trade_price,
                sell_quantity,
                gross_amount,
                commission,
                stamp_duty,
                net_amount,
                realized_pnl,
                note or "",
            ),
        )
        conn.commit()
    finally:
        conn.close()

    snapshot = get_account_snapshot(config, src=src)
    snapshot["order"] = {
        "side": "SELL",
        "stock_code": normalized_code,
        "price": trade_price,
        "limit_price": float(limit_price) if limit_price is not None else None,
        "quote_timestamp": str(quote_row.get("timestamp") or ""),
        "quantity": sell_quantity,
        "gross_amount": gross_amount,
        "commission": commission,
        "stamp_duty": stamp_duty,
        "net_amount": net_amount,
        "realized_pnl": realized_pnl,
        "cash_after": snapshot["summary"]["cash"],
        "initial_cash": initial_cash,
    }
    return snapshot


def place_batch_buy_orders(
    config: SystemConfig,
    orders: list[dict],
) -> dict:
    if not orders:
        raise ValueError("orders 不能为空。")

    results: list[dict] = []
    for item in orders:
        snapshot = place_buy_order(
            config,
            stock_code=item["stock_code"],
            quantity=item.get("quantity"),
            cash_amount=item.get("cash_amount"),
            position_fraction=item.get("position_fraction"),
            limit_price=item.get("limit_price"),
            src=item.get("src", "auto"),
            commission_rate=item.get("commission_rate", DEFAULT_COMMISSION_RATE),
            min_commission=item.get("min_commission", DEFAULT_MIN_COMMISSION),
            board_lot=item.get("board_lot", DEFAULT_BOARD_LOT),
            note=item.get("note"),
        )
        if snapshot.get("order"):
            results.append(snapshot["order"])

    final_snapshot = get_account_snapshot(config, src=str(orders[-1].get("src", "auto")))
    final_snapshot["batch_orders"] = results
    return final_snapshot


def place_batch_sell_orders(
    config: SystemConfig,
    orders: list[dict],
) -> dict:
    if not orders:
        raise ValueError("orders 不能为空。")

    results: list[dict] = []
    for item in orders:
        snapshot = place_sell_order(
            config,
            stock_code=item["stock_code"],
            quantity=item.get("quantity"),
            position_fraction=item.get("position_fraction"),
            limit_price=item.get("limit_price"),
            src=item.get("src", "auto"),
            commission_rate=item.get("commission_rate", DEFAULT_COMMISSION_RATE),
            min_commission=item.get("min_commission", DEFAULT_MIN_COMMISSION),
            stamp_duty_rate=item.get("stamp_duty_rate", DEFAULT_STAMP_DUTY_RATE),
            note=item.get("note"),
        )
        if snapshot.get("order"):
            results.append(snapshot["order"])

    final_snapshot = get_account_snapshot(config, src=str(orders[-1].get("src", "auto")))
    final_snapshot["batch_orders"] = results
    return final_snapshot


def get_account_snapshot(config: SystemConfig, src: str = "auto", trades_limit: int = 50) -> dict:
    db_path = get_paper_trading_db_path(config)
    ensure_paper_trading_schema(db_path)
    conn = sqlite3.connect(db_path)
    try:
        initial_cash, initial_equity, cash, created_at, updated_at = _load_account_row(conn)
        positions_df = _load_positions_frame(conn)
        trades_df = _load_trades_frame(conn, limit=trades_limit)
        trades_count = int(conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0])
    finally:
        conn.close()

    quote_source = str(src).lower()
    if not positions_df.empty:
        positions_df["stock_code"] = positions_df["stock_code"].astype(str).str.zfill(6)
        quote_frame = _get_quote_frame(config, positions_df["stock_code"].astype(str).tolist(), src=src)
        quote_source = quote_frame.attrs.get("realtime_source", quote_source) if not quote_frame.empty else quote_source
        quote_columns = [
            "stock_code",
            "timestamp",
            "date",
            "time",
            "price",
            "pct_change",
            "change",
            "open",
            "high",
            "low",
            "bid",
            "ask",
            "volume",
            "amount",
        ]
        available_quote_columns = [column for column in quote_columns if column in quote_frame.columns]
        positions_df = positions_df.merge(
            quote_frame[available_quote_columns].rename(columns={"amount": "realtime_amount", "date": "quote_date", "time": "quote_time"}),
            on="stock_code",
            how="left",
            suffixes=("", "_quote"),
        )
        positions_df["market_price"] = pd.to_numeric(positions_df.get("price"), errors="coerce")
        positions_df["market_value"] = positions_df["market_price"] * pd.to_numeric(positions_df["quantity"], errors="coerce")
        positions_df["cost_basis"] = pd.to_numeric(positions_df["avg_cost"], errors="coerce") * pd.to_numeric(positions_df["quantity"], errors="coerce")
        positions_df["unrealized_pnl"] = positions_df["market_value"] - positions_df["cost_basis"]
        previous_cost = positions_df["avg_cost"].replace(0, pd.NA)
        positions_df["unrealized_pnl_pct"] = (positions_df["market_price"] - previous_cost) / previous_cost * 100.0
        total_assets_for_weight = float(cash) + float(positions_df["market_value"].fillna(0).sum())
        positions_df["position_weight"] = positions_df["market_value"] / total_assets_for_weight if total_assets_for_weight > 0 else 0.0
    else:
        quote_source = str(src).lower()
        positions_df = pd.DataFrame(columns=["stock_code", "ts_code", "name", "quantity", "avg_cost", "market_price", "market_value", "unrealized_pnl"])

    realized_pnl = float(pd.to_numeric(trades_df.get("realized_pnl"), errors="coerce").fillna(0).sum()) if not trades_df.empty else 0.0
    market_value = float(pd.to_numeric(positions_df.get("market_value"), errors="coerce").fillna(0).sum()) if not positions_df.empty else 0.0
    unrealized_pnl = float(pd.to_numeric(positions_df.get("unrealized_pnl"), errors="coerce").fillna(0).sum()) if not positions_df.empty else 0.0
    total_assets = float(cash) + market_value
    total_pnl = total_assets - float(initial_equity)

    preferred_position_columns = [
        "stock_code",
        "ts_code",
        "name",
        "quantity",
        "avg_cost",
        "market_price",
        "market_value",
        "cost_basis",
        "unrealized_pnl",
        "unrealized_pnl_pct",
        "position_weight",
        "quote_date",
        "quote_time",
        "timestamp",
        "pct_change",
        "change",
        "open",
        "high",
        "low",
        "bid",
        "ask",
        "volume",
        "realtime_amount",
        "updated_at",
    ]
    ordered_position_columns = [column for column in preferred_position_columns if column in positions_df.columns]
    remaining_position_columns = [column for column in positions_df.columns if column not in ordered_position_columns]
    positions_df = positions_df[ordered_position_columns + remaining_position_columns]

    return {
        "source": quote_source,
        "summary": {
            "initial_cash": float(initial_cash),
            "initial_equity": float(initial_equity),
            "cash": float(cash),
            "market_value": market_value,
            "total_assets": total_assets,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "total_return_pct": (total_pnl / float(initial_equity) * 100.0) if float(initial_equity) > 0 else None,
            "positions_count": int(len(positions_df)),
            "trades_count": trades_count,
            "created_at": created_at,
            "updated_at": updated_at,
        },
        "positions": positions_df,
        "trades": trades_df,
    }


def get_trade_history(config: SystemConfig, limit: int = 100) -> pd.DataFrame:
    db_path = get_paper_trading_db_path(config)
    ensure_paper_trading_schema(db_path)
    conn = sqlite3.connect(db_path)
    try:
        return _load_trades_frame(conn, limit=limit)
    finally:
        conn.close()


def get_position_watch(config: SystemConfig, stock_codes: list[str] | tuple[str, ...] | str, src: str = "auto") -> dict:
    normalized_codes = _normalize_stock_code_list(stock_codes)
    db_path = get_paper_trading_db_path(config)
    ensure_paper_trading_schema(db_path)
    conn = sqlite3.connect(db_path)
    try:
        placeholders = ", ".join(["?"] * len(normalized_codes))
        holding_df = pd.read_sql_query(
            f"SELECT stock_code, ts_code, name, quantity, avg_cost, last_trade_price, updated_at FROM paper_positions WHERE stock_code IN ({placeholders}) ORDER BY stock_code",
            conn,
            params=normalized_codes,
        )
    finally:
        conn.close()

    quote_frame = _get_quote_frame(config, normalized_codes, src=src)
    if quote_frame.empty:
        raise RuntimeError(f"未获取到实时行情: {','.join(normalized_codes)}")
    quote_source = quote_frame.attrs.get("realtime_source", str(src).lower())

    holding_view = pd.DataFrame()
    if not holding_df.empty:
        holding_df["stock_code"] = holding_df["stock_code"].astype(str).str.zfill(6)
        holding_view = holding_df.merge(
            quote_frame[[column for column in ["stock_code", "timestamp", "date", "time", "price", "pct_change", "change", "open", "high", "low", "bid", "ask", "volume", "amount"] if column in quote_frame.columns]].rename(columns={"amount": "realtime_amount", "date": "quote_date", "time": "quote_time"}),
            on="stock_code",
            how="left",
        )
        holding_view["market_price"] = pd.to_numeric(holding_view.get("price"), errors="coerce")
        holding_view["market_value"] = holding_view["market_price"] * pd.to_numeric(holding_view["quantity"], errors="coerce")
        holding_view["cost_basis"] = pd.to_numeric(holding_view["avg_cost"], errors="coerce") * pd.to_numeric(holding_view["quantity"], errors="coerce")
        holding_view["unrealized_pnl"] = holding_view["market_value"] - holding_view["cost_basis"]
        avg_cost = holding_view["avg_cost"].replace(0, pd.NA)
        holding_view["unrealized_pnl_pct"] = (holding_view["market_price"] - avg_cost) / avg_cost * 100.0

    quote_view = quote_frame.copy()
    quote_view = quote_view.sort_values([column for column in ["stock_code", "time"] if column in quote_view.columns]).reset_index(drop=True)

    return {
        "source": quote_source,
        "quote": quote_view,
        "holding": holding_view,
    }