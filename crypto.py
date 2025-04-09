# crypto.py

import ccxt

def get_crypto_tickers(n=10, quote_currency="USDT", exchange_name="binance"):
    exchange = getattr(ccxt, exchange_name)()
    markets = exchange.load_markets()

    spot_markets = [
        symbol for symbol, data in markets.items()
        if quote_currency in symbol and data['active'] and data['spot']
    ]

    top_pairs = []
    for symbol in spot_markets:
        try:
            ticker_data = exchange.fetch_ticker(symbol)
            volume = ticker_data['quoteVolume']
            top_pairs.append((symbol, volume))
        except Exception:
            continue

    sorted_pairs = sorted(top_pairs, key=lambda x: x[1], reverse=True)
    return [symbol for symbol, _ in sorted_pairs[:n]]
