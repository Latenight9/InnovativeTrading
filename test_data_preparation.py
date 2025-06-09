from Analysis import load_data
from data_preparation import prepare_data

def main():
    df = load_data(["BTC/USDT", "ETH/USDT"], interval="1h", since_days=10)

    if df.empty:
        print("❌ Keine Daten gefunden!")
        return

    patches = prepare_data(df, window_size=24, step_size=1, patch_size=6)
    print("\n✅ Patches-Shape:", patches.shape)  # z. B. (217, 4, 6, 2)

if __name__ == "__main__":
    main()
