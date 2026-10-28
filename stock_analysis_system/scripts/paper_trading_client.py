from __future__ import annotations

import argparse
import json
from typing import Any
from urllib import error, parse, request


DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _should_bypass_proxy(url: str) -> bool:
    parsed = parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    return host in {"127.0.0.1", "localhost", "::1"}


def _format_http_error(method: str, url: str, status_code: int, detail: Any) -> str:
    detail_text = json.dumps(detail, ensure_ascii=False)
    lines = [f"HTTP {status_code} {method.upper()} {url}", f"响应: {detail_text}"]

    if status_code == 503:
        lines.append(
            "诊断: 交易服务当前不可用。若响应里只有空 detail，这通常更像服务端临时故障、重启或依赖异常，不是下单参数本身的问题。"
        )
        lines.append(
            "建议: 先确认 http://127.0.0.1:8000 服务是否恢复，再重试交易命令；恢复前可先用 account/positions 验证读接口状态。"
        )
    elif status_code >= 500:
        lines.append("诊断: 服务端发生异常，请优先检查服务日志、依赖状态和近期重启情况。")
    elif status_code == 400:
        lines.append("诊断: 请求参数或 JSON 结构可能有误，请检查 stock_code、数量字段和 JSON 转义。")

    return "\n".join(lines)


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url=url, data=data, headers=headers, method=method.upper())
    try:
        opener = request.build_opener(request.ProxyHandler({})) if _should_bypass_proxy(url) else request.build_opener()
        with opener.open(req) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else None
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body)
        except json.JSONDecodeError:
            detail = {"detail": body}
        raise SystemExit(_format_http_error(method, url, exc.code, detail)) from exc
    except error.URLError as exc:
        raise SystemExit(
            "\n".join(
                [
                    f"请求失败: {method.upper()} {url}",
                    f"原因: {exc}",
                    "建议: 检查服务地址、端口、网络连通性，以及 127.0.0.1:8000 是否正在监听。",
                ]
            )
        ) from exc


def _print(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _base_url(args: argparse.Namespace) -> str:
    return str(args.base_url).rstrip("/")


def _load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def command_init(args: argparse.Namespace) -> None:
    positions = []
    for item in args.position:
        parts = item.split(":")
        if len(parts) not in {2, 3}:
            raise SystemExit("--position 格式必须是 stock_code:quantity[:avg_cost]")
        stock_code = parts[0]
        quantity = int(parts[1])
        avg_cost = float(parts[2]) if len(parts) == 3 else None
        payload = {"stock_code": stock_code, "quantity": quantity}
        if avg_cost is not None:
            payload["avg_cost"] = avg_cost
        positions.append(payload)

    payload = {
        "initial_cash": args.initial_cash,
        "reset": not args.no_reset,
        "quote_source": args.src,
        "positions": positions,
    }
    _print(_request_json("POST", f"{_base_url(args)}/paper-trading/account/init", payload))


def command_account(args: argparse.Namespace) -> None:
    query = parse.urlencode({"src": args.src, "trades_limit": args.trades_limit})
    _print(_request_json("GET", f"{_base_url(args)}/paper-trading/account?{query}"))


def command_positions(args: argparse.Namespace) -> None:
    query = parse.urlencode({"src": args.src})
    _print(_request_json("GET", f"{_base_url(args)}/paper-trading/positions?{query}"))


def command_trades(args: argparse.Namespace) -> None:
    query = parse.urlencode({"limit": args.limit})
    _print(_request_json("GET", f"{_base_url(args)}/paper-trading/trades?{query}"))


def command_quote(args: argparse.Namespace) -> None:
    params = {"src": args.src}
    if len(args.stock_code) == 1:
        params["stock_code"] = args.stock_code[0]
    else:
        params["stock_codes"] = ",".join(args.stock_code)
    _print(_request_json("GET", f"{_base_url(args)}/paper-trading/quote?{parse.urlencode(params)}"))


def command_buy(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {
        "stock_code": args.stock_code,
        "src": args.src,
    }
    if args.quantity is not None:
        payload["quantity"] = args.quantity
    if args.cash_amount is not None:
        payload["cash_amount"] = args.cash_amount
    if args.position_fraction is not None:
        payload["position_fraction"] = args.position_fraction
    if args.limit_price is not None:
        payload["limit_price"] = args.limit_price
    if args.note:
        payload["note"] = args.note
    _print(_request_json("POST", f"{_base_url(args)}/paper-trading/orders/buy", payload))


def command_sell(args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {
        "stock_code": args.stock_code,
        "src": args.src,
    }
    if args.quantity is not None:
        payload["quantity"] = args.quantity
    if args.position_fraction is not None:
        payload["position_fraction"] = args.position_fraction
    if args.limit_price is not None:
        payload["limit_price"] = args.limit_price
    if args.note:
        payload["note"] = args.note
    _print(_request_json("POST", f"{_base_url(args)}/paper-trading/orders/sell", payload))


def command_batch_buy(args: argparse.Namespace) -> None:
    payload = _load_json_file(args.file)
    _print(_request_json("POST", f"{_base_url(args)}/paper-trading/orders/batch-buy", payload))


def command_batch_sell(args: argparse.Namespace) -> None:
    payload = _load_json_file(args.file)
    _print(_request_json("POST", f"{_base_url(args)}/paper-trading/orders/batch-sell", payload))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper trading API client for qianxing.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base url, default: http://127.0.0.1:8000")

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize paper trading account")
    init_parser.add_argument("--initial-cash", type=float, required=True)
    init_parser.add_argument("--position", action="append", default=[], help="stock_code:quantity[:avg_cost], can be repeated")
    init_parser.add_argument("--src", default="auto")
    init_parser.add_argument("--no-reset", action="store_true")
    init_parser.set_defaults(func=command_init)

    account_parser = subparsers.add_parser("account", help="Get account summary")
    account_parser.add_argument("--src", default="auto")
    account_parser.add_argument("--trades-limit", type=int, default=20)
    account_parser.set_defaults(func=command_account)

    positions_parser = subparsers.add_parser("positions", help="Get current positions")
    positions_parser.add_argument("--src", default="auto")
    positions_parser.set_defaults(func=command_positions)

    trades_parser = subparsers.add_parser("trades", help="Get recent trades")
    trades_parser.add_argument("--limit", type=int, default=20)
    trades_parser.set_defaults(func=command_trades)

    quote_parser = subparsers.add_parser("quote", help="Get realtime quotes and holdings for one or more stocks")
    quote_parser.add_argument("stock_code", nargs="+", help="One or more stock codes")
    quote_parser.add_argument("--src", default="auto")
    quote_parser.set_defaults(func=command_quote)

    buy_parser = subparsers.add_parser("buy", help="Submit a paper buy order")
    buy_parser.add_argument("stock_code")
    buy_parser.add_argument("--quantity", type=int)
    buy_parser.add_argument("--cash-amount", type=float)
    buy_parser.add_argument("--position-fraction", type=float)
    buy_parser.add_argument("--limit-price", type=float)
    buy_parser.add_argument("--src", default="auto")
    buy_parser.add_argument("--note")
    buy_parser.set_defaults(func=command_buy)

    batch_buy_parser = subparsers.add_parser("batch-buy", help="Submit paper buy orders from a JSON file")
    batch_buy_parser.add_argument("file", help="Path to a JSON file containing {\"orders\": [...]} payload")
    batch_buy_parser.set_defaults(func=command_batch_buy)

    sell_parser = subparsers.add_parser("sell", help="Submit a paper sell order")
    sell_parser.add_argument("stock_code")
    sell_parser.add_argument("--quantity", type=int)
    sell_parser.add_argument("--position-fraction", type=float)
    sell_parser.add_argument("--limit-price", type=float)
    sell_parser.add_argument("--src", default="auto")
    sell_parser.add_argument("--note")
    sell_parser.set_defaults(func=command_sell)

    batch_sell_parser = subparsers.add_parser("batch-sell", help="Submit paper sell orders from a JSON file")
    batch_sell_parser.add_argument("file", help="Path to a JSON file containing {\"orders\": [...]} payload")
    batch_sell_parser.set_defaults(func=command_batch_sell)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()