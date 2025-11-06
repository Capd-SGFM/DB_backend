import os
import requests
import pandas as pd

REST_URL = "https://fapi.binance.com/fapi/v1/ticker/24hr"


def save_all_symbols_by_volume():
    """Binance 선물 거래소의 모든 USDT 기반 심볼을 거래대금 순으로 CSV로 저장 (symbol만 저장)"""
    resp = requests.get(REST_URL)
    tickers = resp.json()
    df = pd.DataFrame(tickers)

    df["quoteVolume"] = df["quoteVolume"].astype(float)

    df = df[df["symbol"].str.endswith("USDT")]
    exclude = ["BUSD", "USDC", "TUSD", "FDUSD", "USDP"]
    df = df[~df["symbol"].str.contains("|".join(exclude))]

    df = df.sort_values("quoteVolume", ascending=False)

    df["symbol"] = df["symbol"].str.replace("USDT", "", regex=False)

    symbols_df = df[["symbol"]].reset_index(drop=True)

    output_path = os.path.join(os.getcwd(), "all_usdt_symbols.csv")
    symbols_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"{len(symbols_df)}개 심볼 저장 완료")

    return symbols_df


if __name__ == "__main__":
    save_all_symbols_by_volume()
