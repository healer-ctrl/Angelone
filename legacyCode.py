from datetime import datetime
import http.client
import json
import schedule
import threading
import time
import pyotp
import pandas as pd
import numpy as np
from SmartApi import SmartConnect
from flask import Flask

# Flask setup (not used here but included in case you want to use it later)
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Angel One API credentials
username = 'U20441'
pwd = '2657'
api_key = "Ca1mkcLk"
token = "G5NV6C5WFPRFMUMDQLOUYUKUDA"
totp = pyotp.TOTP(token).now()
smartApi = SmartConnect(api_key)
n = 0
job_1s = None
global ltpboughtprice

# Authenticate
data = smartApi.generateSession(username, pwd, totp)
authToken = data['data']['jwtToken']
print("Auth Token:", authToken)

# HTTP connection
conn = http.client.HTTPSConnection("apiconnect.angelone.in")
bought = 'false'

# Global variables
past_data = []
current_data = {}

# --- Supertrend Indicator ---
def compute_supertrend(df, atr_period=10, multiplier=3.0):
    df = df.copy()
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(atr_period).mean()
    df['hl2'] = (df['high'] + df['low']) / 2
    df['UpperBand'] = df['hl2'] + multiplier * df['ATR']
    df['LowerBand'] = df['hl2'] - multiplier * df['ATR']

    supertrend = [np.nan] * len(df)
    direction = [1] * len(df)

    for i in range(atr_period, len(df)):
        if direction[i - 1] == 1:
            if df['close'].iloc[i] < df['LowerBand'].iloc[i]:
                direction[i] = -1
                supertrend[i] = df['UpperBand'].iloc[i]
            else:
                direction[i] = 1
                supertrend[i] = max(df['LowerBand'].iloc[i], supertrend[i - 1])
        else:
            if df['close'].iloc[i] > df['UpperBand'].iloc[i]:
                direction[i] = 1
                supertrend[i] = df['LowerBand'].iloc[i]
            else:
                direction[i] = -1
                supertrend[i] = min(df['UpperBand'].iloc[i], supertrend[i - 1])

    df['Supertrend'] = supertrend
    df['Direction'] = direction
    return df

# --- EMA + Bollinger Bands ---
def compute_ema_and_bollinger(df, ema_length=9, smoothing_type="SMA + Bollinger Bands", smoothing_length=14, bb_std=2.0):
    close = df['ltp']
    df['EMA'] = close.ewm(span=ema_length, adjust=False).mean()

    def apply_smoothing(series, kind):
        if kind == "SMA":
            return series.rolling(smoothing_length).mean()
        elif kind == "SMA + Bollinger Bands":
            return series.rolling(smoothing_length).mean()
        elif kind == "EMA":
            return series.ewm(span=smoothing_length, adjust=False).mean()
        elif kind == "SMMA":
            return series.ewm(alpha=1/smoothing_length, adjust=False).mean()
        elif kind == "WMA":
            weights = np.arange(1, smoothing_length + 1)
            return series.rolling(smoothing_length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
        elif kind == "VWMA":
            if 'volume' not in df:
                raise ValueError("VWMA requires 'volume' data.")
            return (df['close'] * df['volume']).rolling(smoothing_length).sum() / df['volume'].rolling(smoothing_length).sum()
        else:
            return pd.Series(index=series.index, dtype='float64')

    if smoothing_type != "None":
        df['SmoothingMA'] = apply_smoothing(df['EMA'], smoothing_type)
        if smoothing_type == "SMA + Bollinger Bands":
            std_dev = df['EMA'].rolling(smoothing_length).std()
            df['BB_Upper'] = df['SmoothingMA'] + bb_std * std_dev
            df['BB_Lower'] = df['SmoothingMA'] - bb_std * std_dev
        else:
            df['BB_Upper'] = np.nan
            df['BB_Lower'] = np.nan
    else:
        df['SmoothingMA'] = np.nan
        df['BB_Upper'] = np.nan
        df['BB_Lower'] = np.nan

    return df

# --- Place Buy Order ---
def place_buy_order(market_price, quantity):
    disclosed_qty = min(quantity - 1, 1)
    rule_id = None

    try:
        price = current_data['ltp']
        tradingsymbol = 'JIOFIN-EQ'

        gttCreateParams = {
            "tradingsymbol": tradingsymbol,
            "symboltoken": "18143",
            "exchange": "NSE",
            "producttype": "DELIVERY",
            "transactiontype": "BUY",
            "price": f'{price}',
            "qty": f'{quantity}',
            "disclosedqty": f'{disclosed_qty}',
            "triggerprice": f'{market_price}',
            "timeperiod": 365
        }

        rule_id = smartApi.gttCreateRule(gttCreateParams)
        print(f"[BUY] GTT rule created. Rule ID: {rule_id}")

    except Exception as e:
        print(f"[BUY] GTT Rule creation failed: {e}")

    return rule_id

# --- Place Sell Order ---
def place_sell_order(market_price, quantity):
    disclosed_qty = min(quantity - 1, 1)
    rule_id = None

    try:
        price = current_data['ltp']
        tradingsymbol = 'JIOFIN-EQ'

        gttCreateParams = {
            "tradingsymbol": tradingsymbol,
            "symboltoken": "18143",
            "exchange": "NSE",
            "producttype": "DELIVERY",
            "transactiontype": "SELL",
            "price": f'{price}',
            "qty": f'{quantity}',
            "disclosedqty": f'{disclosed_qty}',
            "triggerprice": f'{market_price}',
            "timeperiod": 365
        }

        rule_id = smartApi.gttCreateRule(gttCreateParams)
        print(f"[SELL] GTT rule created. Rule ID: {rule_id}")

    except Exception as e:
        print(f"[SELL] GTT Rule creation failed: {e}")

    return rule_id

# --- Get Live OHLC + LTP from New API ---
def get_live_data():
    global past_data, current_data, bought, n, job_1s, ltpboughtprice

    n += 1
    print(f"Running get_live_data... Call #{n}")

    if n == 13:
        ltpboughtprice = 267
        bought = 'true'
        schedule.cancel_job(job_1s)
        print("Finished 13 runs. Switching to 5-minute interval.")
        schedule.every(1).minutes.do(get_live_data)

    # Updated OHLC + LTP fetch logic
    payload = json.dumps({
        "mode": "OHLC",
        "exchangeTokens": {
            "NSE": ["18143"]
        }
    })

    headers = {
        'X-PrivateKey': 'Ca1mkcLk',
        'Accept': 'application/json',
        'X-SourceID': 'WEB',
        'X-ClientLocalIP': 'CLIENT_LOCAL_IP',
        'X-ClientPublicIP': 'CLIENT_PUBLIC_IP',
        'X-MACAddress': 'MAC_ADDRESS',
        'X-UserType': 'USER',
        'Authorization': f'{authToken}',
        'Content-Type': 'application/json'
    }

    try:
        conn.request("POST", "/rest/secure/angelbroking/market/v1/quote/", payload, headers)

        res = conn.getresponse()
        print(res.raw);

        data = res.read()
        json_data = json.loads(data)
        print(json);

        fetched_list = json_data["data"]["fetched"]
        symbol_token = "18143"

        # Find matching symbol data
        stock_data = next((item for item in fetched_list if item["symbolToken"] == symbol_token), None)

        if not stock_data:
            print("Symbol data not found.")
            return

        ltp = float(stock_data["ltp"])
        current_time = datetime.now()

        current_data = {
            'open': float(stock_data['open']),
            'high': float(stock_data['high']),
            'low': float(stock_data['low']),
            'close': float(stock_data['close']),
            'ltp': ltp,
            'timestamp': current_time.isoformat()
        }
        print(current_data)

        past_data.append(current_data)
        if len(past_data) > 200:
            past_data = past_data[-200:]

        df = pd.DataFrame(past_data)

        if len(df) >= 5:
            df = compute_supertrend(df, atr_period=10, multiplier=3.0)
            df = compute_ema_and_bollinger(df, ema_length=9, smoothing_type="SMA + Bollinger Bands", smoothing_length=14, bb_std=2.0)

            print("Latest Indicators:")
            print(df[['close','ltp', 'Supertrend', 'Direction', 'EMA']].tail())

            # Buy condition
            if df['Direction'].iloc[-1] == 1 and df['close'].iloc[-1] > df['EMA'].iloc[-1] and bought == 'false':
                place_buy_order(market_price=current_data['ltp'], quantity=1)
                bought = 'true'
                ltpboughtprice = current_data['ltp']
                print("I Bought a share of 1 quantity though")

            # Sell condition
            if df['Direction'].iloc[-1] == -1 and bought == 'true':
                if current_data['ltp'] > (ltpboughtprice + ltpboughtprice * 0.01):
                    place_sell_order(market_price=current_data['ltp'], quantity=1)
                    bought = 'false'
                    print("I sold a share of 1 quantity though")

        else:
            print("Waiting for more data to compute indicators...")

    except Exception as e:
        print(f"Error fetching live data: {e}")

# --- Scheduler ---
def run_schedule():
    while True:
        schedule.run_pending()
        time.sleep(1)

# --- Main ---
if __name__ == '__main__':
    job_1s = schedule.every(1).seconds.do(get_live_data)
    threading.Thread(target=run_schedule, daemon=True).start()

    print("Scheduler started. Fetching data...")
    while True:
        time.sleep(60)
