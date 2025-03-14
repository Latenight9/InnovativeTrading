import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen

#Symbole aus Tabelle in Wikipedia auslesen
sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
sp500_df = pd.read_html(sp500_url)[0]
sp500_tickers = sp500_df["Symbol"].tolist()
invalid_tickers = ['BF.B', 'BRK.B']
sp500_tickers = [t for t in sp500_tickers if t not in invalid_tickers]
#print(sp500_tickers[:10])


start_date = "2024-01-01"
end_date = "2025-01-01"

#Preise aus Yfinance erhalten
closing_prices_df = yf.download(sp500_tickers, start=start_date, end=end_date)["Close"]

#Zwischenspeichern
closing_prices_dict = {ticker: closing_prices_df[ticker] for ticker in closing_prices_df.columns}

#print(closing_prices_df.head())

#Daten Aufbereitung / NaNs entfernen
# Fehlende Werte behandeln
closing_prices_df.ffill(inplace=True)  # Erst mit vorherigem Wert füllen
closing_prices_df.bfill(inplace=True)  # Dann mit nächstem Wert füllen
closing_prices_df.interpolate(method='linear', inplace=True)  # Falls noch Lücken da sind, interpolieren

#print("Anzahl verbleibender NaN-Werte:", closing_prices_df.isna().sum().sum())

#Datenpunkte Anzahl überprüfen
# 🔍 Prüfen, ob alle Spalten (Aktien) gleich viele Datenpunkte haben
row_counts = closing_prices_df.count()

# Prüfen, ob alle Spalten die gleiche Anzahl an Datenpunkten haben
if row_counts.nunique() == 1:
    print("✅ Alle Preisserien haben die gleiche Anzahl an Datenpunkten:", row_counts.iloc[0])
else:
    print("⚠ Unterschiedliche Anzahl an Datenpunkten gefunden!")
    print(row_counts.value_counts())  
    
#Korrelation berchnen
correlation_matrix = closing_prices_df.corr(method="pearson")
correlation_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))

#Korrelationen ordnen, ausgeben, ploten 
correlation_pairs = correlation_matrix.unstack().dropna().sort_values(ascending=False)
correlation_pairs = correlation_pairs[correlation_pairs < 0.9999]

top_15_correlations = correlation_pairs.head(15)



#Plot der 15 höhst korrelierenden
plt.figure(figsize=(12, 6))
plt.barh(top_15_correlations.index.map(lambda x: f"{x[0]} - {x[1]}"), top_15_correlations.values, color='green') #Extra umwandlung in Lesbare Strings für Plot
plt.xlabel("Pearson-Korrelationskoeffizient")
plt.ylabel("Aktienpaare")
plt.title("Top 15 höchsten Korrelationen zwischen Aktien")  
plt.xlim(0.85, 1)  
plt.xticks(np.arange(0.85, 1.1, 0.02))  
plt.gca().invert_yaxis()  
plt.show()

#Kointegrationstest mittels Johansen Test
cointegration_results = {}
for stock1, stock2 in top_15_correlations.index:
    print(f"🔄 Berechne Johansen-Test für: {stock1} & {stock2}")  
    prices_matrix = closing_prices_df[[stock1, stock2]]
    
result = coint_johansen(prices_matrix, det_order=0, k_ar_diff=1)

trace_stat = result.lr1[0]
critical_value = result.cvt[0, 1]
is_cointegrated = trace_stat > critical_value  

cointegration_results[(stock1, stock2)] = {
        "Trace-Statistik": trace_stat,
        "Kritischer Wert (95%)": critical_value,
        "Cointegrated": is_cointegrated
    }

print("\n📈 Ergebnisse des Johansen-Tests für Top-Paare:\n")
for pair, result in cointegration_results.items():
    print(f"{pair}: Trace-Statistik = {result['Trace-Statistik']:.3f}, "
          f"Kritischer Wert (95%) = {result['Kritischer Wert (95%)']:.3f}, "
          f"Cointegrated: {result['Cointegrated']}")
    
