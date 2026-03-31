"""
Binance USDⓈ-M Futures Demo API Client
======================================
Handles exchange communication for a futures demo trading bot.

Official USDⓈ-M Futures demo/testnet REST base:
  https://demo-fapi.binance.com

Notes:
- Futures REST endpoints use /fapi/... paths, not /api/v3/...
- Signed endpoints require:
    * X-MBX-APIKEY header
    * timestamp
    * HMAC SHA256 signature
- Spot OCO logic is not used here. Futures exits are handled with
  separate STOP_MARKET and TAKE_PROFIT_MARKET reduce-only orders.
"""

import hashlib
import hmac
import logging
import time
from typing import Dict, List, Optional
from urllib.parse import urlencode

import requests

logger = logging.getLogger("binance_client")


class BinanceFuturesDemoClient:
    """
    REST API client for Binance USDⓈ-M Futures Demo Mode.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://demo-fapi.binance.com",
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "X-MBX-APIKEY": self.api_key,
        })

    # -------------------------------------------------------------------------
    # Internal request helpers
    # -------------------------------------------------------------------------

    def _sign(self, params: dict) -> dict:
        """
        Add timestamp / recvWindow / signature.
        Binance requires signature to be computed over the query string.
        """
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000)
        params.setdefault("recvWindow", 5000)

        query = urlencode(params, doseq=True)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        params["signature"] = signature
        return params

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        signed: bool = False,
    ):
        params = params or {}
        if signed:
            params = self._sign(params)

        url = f"{self.base_url}{path}"

        try:
            resp = self.session.request(
                method=method.upper(),
                url=url,
                params=params,
                timeout=20,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            body = ""
            try:
                body = resp.text
            except Exception:
                pass
            logger.error("HTTP %s %s failed: %s | body=%s", method, path, e, body)
            raise

    def _get(self, path: str, params: Optional[dict] = None, signed: bool = False):
        return self._request("GET", path, params=params, signed=signed)

    def _post(self, path: str, params: Optional[dict] = None, signed: bool = True):
        return self._request("POST", path, params=params, signed=signed)

    def _delete(self, path: str, params: Optional[dict] = None, signed: bool = True):
        return self._request("DELETE", path, params=params, signed=signed)

    # -------------------------------------------------------------------------
    # Public market data
    # -------------------------------------------------------------------------

    def ping(self) -> dict:
        return self._get("/fapi/v1/ping")

    def get_server_time(self) -> dict:
        return self._get("/fapi/v1/time")

    def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[list]:
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        return self._get("/fapi/v1/klines", params)

    def get_ticker_price(self, symbol: str) -> dict:
        return self._get("/fapi/v1/ticker/price", {"symbol": symbol})

    def get_exchange_info(self, symbol: Optional[str] = None) -> dict:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._get("/fapi/v1/exchangeInfo", params)

    # -------------------------------------------------------------------------
    # Account / balances / positions
    # -------------------------------------------------------------------------

    def get_account(self) -> dict:
        """
        Futures account snapshot.
        """
        return self._get("/fapi/v3/account", signed=True)

    def get_balances(self) -> List[dict]:
        """
        Futures balances list.
        """
        return self._get("/fapi/v3/balance", signed=True)

    def get_balance(self, asset: str = "USDT") -> float:
        """
        Returns availableBalance for the requested futures asset.
        """
        balances = self.get_balances()
        for bal in balances:
            if bal.get("asset") == asset:
                return float(bal.get("availableBalance", 0.0))
        return 0.0

    def get_position_risk(self, symbol: Optional[str] = None):
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._get("/fapi/v3/positionRisk", params=params, signed=True)

    # -------------------------------------------------------------------------
    # Orders
    # -------------------------------------------------------------------------

    def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
        new_client_order_id: Optional[str] = None,
    ) -> dict:
        """
        One-way mode futures market order.
        side:
          BUY  -> opens/increases long OR closes short if reduce_only=True
          SELL -> opens/increases short OR closes long if reduce_only=True
        """
        params = {
            "symbol": symbol,
            "side": side,
            "type": "MARKET",
            "quantity": self.format_quantity(symbol, quantity),
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        if new_client_order_id:
            params["newClientOrderId"] = new_client_order_id

        logger.info("FUTURES MARKET %s %s %s", side, quantity, symbol)
        return self._post("/fapi/v1/order", params)

    def place_stop_market_order(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        quantity: float,
        reduce_only: bool = True,
        working_type: str = "MARK_PRICE",
        new_client_order_id: Optional[str] = None,
    ) -> dict:
        params = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "triggerPrice": self.format_price(symbol, stop_price),
            "quantity": self.format_quantity(symbol, quantity),
            "workingType": working_type,
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        if new_client_order_id:
            params["clientAlgoId"] = new_client_order_id

        logger.info(
            "FUTURES ALGO STOP_MARKET %s %s %s trigger=%s",
            side, quantity, symbol, stop_price
        )
        return self._post("/fapi/v1/algoOrder", params)

    def place_take_profit_market_order(
        self,
        symbol: str,
        side: str,
        stop_price: float,
        quantity: float,
        reduce_only: bool = True,
        working_type: str = "MARK_PRICE",
        new_client_order_id: Optional[str] = None,
    ) -> dict:
        params = {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "triggerPrice": self.format_price(symbol, stop_price),
            "quantity": self.format_quantity(symbol, quantity),
            "workingType": working_type,
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        if new_client_order_id:
            params["clientAlgoId"] = new_client_order_id

        logger.info(
            "FUTURES ALGO TAKE_PROFIT_MARKET %s %s %s trigger=%s",
            side, quantity, symbol, stop_price
        )
        return self._post("/fapi/v1/algoOrder", params)

    def get_open_orders(self, symbol: Optional[str] = None) -> list:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._get("/fapi/v1/openOrders", params=params, signed=True)

    def cancel_order(self, symbol: str, order_id: int) -> dict:
        return self._delete("/fapi/v1/order", {"symbol": symbol, "orderId": order_id})

    def cancel_all_orders(self, symbol: str) -> dict:
        return self._delete("/fapi/v1/allOpenOrders", {"symbol": symbol})

    # -------------------------------------------------------------------------
    # Symbol metadata / formatting helpers
    # -------------------------------------------------------------------------

    def get_symbol_filters(self, symbol: str) -> dict:
        info = self.get_exchange_info(symbol)
        for s in info.get("symbols", []):
            if s.get("symbol") == symbol:
                filters = {}
                for f in s.get("filters", []):
                    filters[f["filterType"]] = f
                return {
                    "filters": filters,
                    "status": s.get("status"),
                    "baseAsset": s.get("baseAsset"),
                    "quoteAsset": s.get("quoteAsset"),
                    "pricePrecision": s.get("pricePrecision"),
                    "quantityPrecision": s.get("quantityPrecision"),
                }
        return {}

    def round_quantity(self, symbol: str, quantity: float) -> float:
        info = self.get_symbol_filters(symbol)
        lot_size = info.get("filters", {}).get("LOT_SIZE", {})
        step = float(lot_size.get("stepSize", "0.001"))
        if step <= 0:
            return quantity

        rounded = quantity - (quantity % step)
        precision = max(0, len(str(step).rstrip("0").split(".")[-1]))
        return round(rounded, precision)

    def round_price(self, symbol: str, price: float) -> float:
        info = self.get_symbol_filters(symbol)
        pf = info.get("filters", {}).get("PRICE_FILTER", {})
        tick = float(pf.get("tickSize", "0.01"))
        if tick <= 0:
            return price

        rounded = price - (price % tick)
        precision = max(0, len(str(tick).rstrip("0").split(".")[-1]))
        return round(rounded, precision)

    def format_quantity(self, symbol: str, quantity: float) -> str:
        q = self.round_quantity(symbol, quantity)
        return f"{q:.8f}".rstrip("0").rstrip(".")

    def format_price(self, symbol: str, price: float) -> str:
        p = self.round_price(symbol, price)
        return f"{p:.8f}".rstrip("0").rstrip(".")


def fetch_full_klines(
    client: BinanceFuturesDemoClient,
    symbol: str,
    interval: str,
    total_bars: int = 2000,
) -> List[dict]:
    """
    Fetch up to total_bars of futures historical klines, paginating backward.
    """
    all_klines = []
    end_time = None

    while len(all_klines) < total_bars:
        remaining = total_bars - len(all_klines)
        limit = min(remaining, 1000)

        raw = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
            end_time=(end_time - 1) if end_time else None,
        )
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

    if len(all_klines) > total_bars:
        all_klines = all_klines[-total_bars:]

    return all_klines
