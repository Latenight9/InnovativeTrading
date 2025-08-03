from Analysis import (
    load_data, calculate_top_correlations, run_johansen_tests,
    check_spread_stationarity, plot_top_pairs, plot_spread
)
from crypto import get_crypto_tickers
from stock import get_stock_tickers
from config import DATA_MODE, LOOKBACK_PERIOD

# Nur für Stocks nötig:
START_DATE = "2006-05-01"
END_DATE = "2010-04-01"
INTERVAL = "1d"

# Für Crypto:
SINCE_DAYS = 365
CRYPTO_INTERVAL = "1h"

def main():
    if DATA_MODE == "crypto":
        tickers = get_crypto_tickers(n=25)
        print(f"✅ {len(tickers)} Krypto-Ticker geladen.")

        prices = load_data(tickers, interval=CRYPTO_INTERVAL, since_days=SINCE_DAYS)

    else:
        tickers = get_stock_tickers()
        print(f"✅ {len(tickers)} Aktien-Ticker geladen.")

        prices = load_data(tickers, start_date=START_DATE, end_date=END_DATE, interval=INTERVAL)

    top_pairs = calculate_top_correlations(prices, top_n=15)
    plot_top_pairs(top_pairs)

    johansen_results = run_johansen_tests(top_pairs, prices)

    stationary_pairs = check_spread_stationarity(prices, johansen_results)
    
    for pair_key, pair_data in stationary_pairs.items():
        print(f"\n📉 Plot für {pair_key[0]} - {pair_key[1]}")
        plot_spread(pair_key, pair_data)
        break  

    print("\n📌 Stationäre Paare:")
    for pair in stationary_pairs:
        print(pair)

if __name__ == "__main__":
    main()
