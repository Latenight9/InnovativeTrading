def get_stock_tickers():
    import pandas as pd
    sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_df = pd.read_html(sp500_url)[0]
    tickers = sp500_df["Symbol"].tolist()
    invalid = ['BF.B', 'BRK.B']
    return [t for t in tickers if t not in invalid]
