import time
from typing import Dict, List, Optional

import ccxt
import pandas as pd


def normalize_symbol_label(symbol: str) -> str:
    return symbol.replace('/', '').replace(':USDT', '').replace('USDT', '')


def _to_record(symbol: str, df: pd.DataFrame) -> Optional[Dict]:
    if df is None or df.empty:
        return None

    frame = df.copy()
    frame = frame.dropna(subset=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    frame = frame.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    if frame.empty:
        return None

    return {
        'symbol': normalize_symbol_label(symbol),
        'opens': frame['open'].to_numpy(dtype=float),
        'highs': frame['high'].to_numpy(dtype=float),
        'lows': frame['low'].to_numpy(dtype=float),
        'closes': frame['close'].to_numpy(dtype=float),
        'volumes': frame['volume'].to_numpy(dtype=float),
        'timestamps': frame['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
    }


def fetch_ohlcv_full(
    exchange,
    symbol: str,
    timeframe: str = '1h',
    since_ms: Optional[int] = None,
    limit_per_call: int = 1000,
    max_bars: Optional[int] = None,
    sleep_factor: float = 1.1,
) -> pd.DataFrame:
    all_rows = []
    next_since = since_ms

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=next_since, limit=limit_per_call)
        if not batch:
            break

        all_rows.extend(batch)

        if max_bars is not None and len(all_rows) >= max_bars:
            all_rows = all_rows[:max_bars]
            break

        last_ts = batch[-1][0]
        new_since = last_ts + 1
        if next_since is not None and new_since <= next_since:
            break
        next_since = new_since

        if len(batch) < limit_per_call:
            break

        time.sleep((exchange.rateLimit / 1000.0) * sleep_factor)

    if not all_rows:
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(all_rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['ts'], unit='ms', utc=True).dt.tz_convert(None)
    return df.drop(columns=['ts'])


def build_exchange_dataset(
    symbols: List[str],
    timeframe: str = '1h',
    exchange_id: str = 'binance',
    start_date: str = '2021-01-01',
    min_bars: int = 5000,
    max_bars: Optional[int] = None,
    market_type: str = 'spot',
) -> List[Dict]:
    exchange_class = getattr(ccxt, exchange_id)
    options = {'enableRateLimit': True}
    if market_type and market_type != 'spot':
        options['options'] = {'defaultType': market_type}
    exchange = exchange_class(options)
    exchange.load_markets()

    since_ms = int(pd.Timestamp(start_date, tz='UTC').timestamp() * 1000)
    datasets: List[Dict] = []

    for symbol in symbols:
        if symbol not in exchange.markets:
            print(f'Skipping {symbol}: not found on {exchange_id}')
            continue
        try:
            df = fetch_ohlcv_full(
                exchange=exchange,
                symbol=symbol,
                timeframe=timeframe,
                since_ms=since_ms,
                limit_per_call=min(1000, exchange.features.get('spot', {}).get('fetchOHLCV', {}).get('limit', 1000) if hasattr(exchange, 'features') else 1000),
                max_bars=max_bars,
            )
            rec = _to_record(symbol, df)
            if rec is not None and len(rec['closes']) >= min_bars:
                datasets.append(rec)
                print(f"Loaded {symbol}: {len(rec['closes'])} candles")
            else:
                got = 0 if rec is None else len(rec['closes'])
                print(f'Skipping {symbol}: only {got} candles')
        except Exception as exc:
            print(f'Failed {symbol}: {exc}')

    return datasets
