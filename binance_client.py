"""
Binance Demo Mode API Client
==============================
Handles all exchange communication for the live trading bot.

Uses Binance Demo Mode endpoints:
- REST: https://demo-api.binance.com/api
- WebSocket: wss://demo-stream.binance.com/ws

Requirements:
  pip install requests websocket-client

No external trading libraries needed — we use raw REST/WebSocket
for maximum control and transparency.
"""

import hashlib
import hmac
import json
import logging
import time
from typing import Dict, List, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger("binance_client")


class BinanceDemoClient:
    """
    REST API client for Binance Demo Mode.
    Implements only the endpoints our bot needs.
    """

    def __init__(self, api_key: str, api_secret: str,
                 base_url: str = "https://demo-api.binance.com"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.api_key,
        })

    def _sign(self, params: dict) -> dict:
        """Add timestamp and HMAC-SHA256 signature to request params."""
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _get(self, path: str, params: dict = None, signed: bool = False) -> dict:
        params = params or {}
        if signed:
            params = self._sign(params)
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, params: dict = None, signed: bool = True) -> dict:
        params = params or {}
        if signed:
            params = self._sign(params)
        url = f"{self.base_url}{path}"
        resp = self.session.post(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, path: str, params: dict = None, signed: bool = True) -> dict:
        params = params or {}
        if signed:
            params = self._sign(params)
        url = f"{self.base_url}{path}"
        resp = self.session.delete(url, params=params)
        resp.raise_for_status()
        return resp.json()

    # === Market Data (public, no signature needed) ===

    def get_klines(self, symbol: str, interval: str, limit: int = 1000,
                   start_time: Optional[int] = None) -> List[list]:
        """
        Fetch historical klines/candlestick data.
        Returns list of [open_time, open, high, low, close, volume, ...].
        Max 1000 per request.
        """
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        return self._get("/api/v3/klines", params)

    def get_ticker_price(self, symbol: str) -> dict:
        """Get current price for a symbol."""
        return self._get("/api/v3/ticker/price", {"symbol": symbol})

    def get_exchange_info(self, symbol: str = None) -> dict:
        """Get exchange trading rules and symbol info."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._get("/api/v3/exchangeInfo", params)

    # === Account (signed) ===

    def get_account(self) -> dict:
        """Get current account information including balances."""
        return self._get("/api/v3/account", signed=True)

    def get_balance(self, asset: str = "USDT") -> float:
        """Get free balance for a specific asset."""
        account = self.get_account()
        for bal in account.get("balances", []):
            if bal["asset"] == asset:
                return float(bal["free"])
        return 0.0

    # === Orders ===

    def place_market_order(self, symbol: str, side: str,
                           quantity: float) -> dict:
        """
        Place a market order.
        side: "BUY" or "SELL"
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": f"{quantity:.8f}".rstrip("0").rstrip("."),
        }
        logger.info(f"MARKET {side} {quantity} {symbol}")
        return self._post("/api/v3/order", params)

    def place_limit_order(self, symbol: str, side: str,
                          quantity: float, price: float,
                          time_in_force: str = "GTC") -> dict:
        """Place a limit order."""
        params = {
            "symbol": symbol,
            "side": side,
            "type": "LIMIT",
            "timeInForce": time_in_force,
            "quantity": f"{quantity:.8f}".rstrip("0").rstrip("."),
            "price": f"{price:.8f}".rstrip("0").rstrip("."),
        }
        logger.info(f"LIMIT {side} {quantity} {symbol} @ {price}")
        return self._post("/api/v3/order", params)

    def place_oco_order(self, symbol: str, side: str, quantity: float,
                        price: float, stop_price: float,
                        stop_limit_price: float) -> dict:
        """
        Place an OCO (One-Cancels-Other) order for exit management.
        This places BOTH the take-profit limit and stop-loss simultaneously.
        When one fills, the other is automatically cancelled.

        For a LONG position exit:
          side = "SELL"
          price = take-profit price (above current)
          stop_price = stop-loss trigger price
          stop_limit_price = stop-loss limit price (slightly below stop_price)

        For a SHORT position exit (on spot, this means selling what you bought):
          Spot doesn't support true shorting — see note in bot.
        """
        params = {
            "symbol": symbol,
            "side": side,
            "quantity": f"{quantity:.8f}".rstrip("0").rstrip("."),
            "price": f"{price:.8f}".rstrip("0").rstrip("."),
            "stopPrice": f"{stop_price:.8f}".rstrip("0").rstrip("."),
            "stopLimitPrice": f"{stop_limit_price:.8f}".rstrip("0").rstrip("."),
            "stopLimitTimeInForce": "GTC",
        }
        logger.info(f"OCO {side} {quantity} {symbol}: TP={price}, SL={stop_price}")
        return self._post("/api/v3/order/oco", params)

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        """Cancel an open order."""
        params = {"symbol": symbol, "orderId": order_id}
        return self._delete("/api/v3/order", params)

    def get_open_orders(self, symbol: str = None) -> list:
        """Get all open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._get("/api/v3/openOrders", params, signed=True)

    def cancel_all_orders(self, symbol: str) -> dict:
        """Cancel all open orders for a symbol."""
        params = {"symbol": symbol}
        return self._delete("/api/v3/openOrders", params)

    # === Symbol Info Helpers ===

    def get_symbol_filters(self, symbol: str) -> dict:
        """Get trading filters (lot size, price filter, etc.) for a symbol."""
        info = self.get_exchange_info(symbol)
        for s in info.get("symbols", []):
            if s["symbol"] == symbol:
                filters = {}
                for f in s.get("filters", []):
                    filters[f["filterType"]] = f
                return {
                    "filters": filters,
                    "baseAsset": s["baseAsset"],
                    "quoteAsset": s["quoteAsset"],
                    "status": s["status"],
                    "baseAssetPrecision": s["baseAssetPrecision"],
                    "quotePrecision": s["quotePrecision"],
                }
        return {}

    def round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to the symbol's LOT_SIZE step."""
        info = self.get_symbol_filters(symbol)
        lot_size = info.get("filters", {}).get("LOT_SIZE", {})
        step = float(lot_size.get("stepSize", "0.00001"))
        precision = len(str(step).rstrip("0").split(".")[-1])
        return round(quantity - (quantity % step), precision)

    def round_price(self, symbol: str, price: float) -> float:
        """Round price to the symbol's PRICE_FILTER tick size."""
        info = self.get_symbol_filters(symbol)
        pf = info.get("filters", {}).get("PRICE_FILTER", {})
        tick = float(pf.get("tickSize", "0.01"))
        precision = len(str(tick).rstrip("0").split(".")[-1])
        return round(price - (price % tick), precision)


def fetch_full_klines(client: BinanceDemoClient, symbol: str,
                      interval: str, total_bars: int = 2000) -> List[dict]:
    """
    Fetch up to total_bars of historical klines, paginating as needed.
    Returns list of dicts with keys: timestamp, open, high, low, close, volume.
    """
    all_klines = []
    end_time = None

    while len(all_klines) < total_bars:
        remaining = total_bars - len(all_klines)
        limit = min(remaining, 1000)

        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if end_time:
            params["endTime"] = end_time - 1

        raw = client._get("/api/v3/klines", params)
        if not raw:
            break

        for k in raw:
            all_klines.append({
                "timestamp": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": k[6],
            })

        end_time = raw[0][0]
        if len(raw) < limit:
            break

    all_klines.sort(key=lambda x: x["timestamp"])

    # Trim to last total_bars
    if len(all_klines) > total_bars:
        all_klines = all_klines[-total_bars:]

    return all_klines
