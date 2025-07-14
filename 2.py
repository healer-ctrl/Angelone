import pandas as pd
import numpy as np
import requests
import time
import json
from datetime import datetime, timedelta
import websocket
import threading
from SmartApi import SmartConnect
import logging
import smtplib
from concurrent.futures import ThreadPoolExecutor
import queue
import pyotp
from http.server import BaseHTTPRequestHandler, HTTPServer
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

stock_status_html = """<html><head><title>Bot Status</title></head><body><h1>Live Stock Indicator Report</h1>{table}</body></html>"""

class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        table_html = bot.render_stock_status_html()
        self.wfile.write(table_html.encode())

class MultiStockTradingBot:
    def __init__(self, api_key, client_code, pin, totp_token):
        self.api_key = api_key
        self.client_code = client_code
        self.pin = pin
        self.totp_token = totp_token
        self.smartApi = SmartConnect(api_key=api_key)
        self.auth_token = None
        self.feed_token = None
        self.refresh_token = None

        self.stocks = {
            "SBIN": {"symbol": "SBIN-EQ", "token": "3045", "quantity": 1},
            "NIFTY31JUL25FUT": {"symbol": "NIFTY31JUL25FUT", "token": "53216", "quantity": 1}
        }

        self.exchange = "NSE"
        self.timeframe = "FIVE_MINUTE"
        self.positions = {stock: False for stock in self.stocks}
        self.last_signal_times = {stock: None for stock in self.stocks}
        self.stock_dataframes = {stock: pd.DataFrame() for stock in self.stocks}
        self.thread_pool = ThreadPoolExecutor(max_workers=len(self.stocks))
        self.signal_queue = queue.Queue()
        self.stock_html_log = {}  # For live UI reporting

    def login(self):
        try:
            data = self.smartApi.generateSession(self.client_code, self.pin, self.totp_token)
            if data['status']:
                self.auth_token = data['data']['jwtToken']
                self.feed_token = data['data']['feedToken']
                self.refresh_token = data['data']['refreshToken']
                logger.info("Login successful!")
                return True
            else:
                logger.error(f"Login failed: {data['message']}")
                return False
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return False

    def get_historical_data(self, symbol, token, days=5):
        try:
            to_date = datetime.now().strftime("%Y-%m-%d %H:%M")
            from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
            historicParam = {
                "exchange": self.exchange,
                "symboltoken": token,
                "interval": self.timeframe,
                "fromdate": from_date,
                "todate": to_date
            }
            if "NIFTY" in symbol:
                historicParam["exchange"] = "NFO"
            response = self.smartApi.getCandleData(historicParam)
            if response['status']:
                df = pd.DataFrame(response['data'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df = df.astype(float)
                return df
            else:
                logger.error(f"Historical data error: {response['message']}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Data fetch error: {str(e)}")
            return pd.DataFrame()

    def calculate_vwap(self, df):
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap_numerator'] = df['typical_price'] * df['volume']
        df['vwap_denominator'] = df['volume']
        df['date'] = df.index.date
        df['cumulative_vwap_num'] = df.groupby('date')['vwap_numerator'].cumsum()
        df['cumulative_vwap_den'] = df.groupby('date')['vwap_denominator'].cumsum()
        df['vwap'] = df['cumulative_vwap_num'] / df['cumulative_vwap_den']
        return df

    def calculate_ema(self, df, period):
        return df['close'].ewm(span=period, adjust=False).mean()

    def calculate_indicators(self, df):
        df = self.calculate_vwap(df)
        df['ema_5'] = self.calculate_ema(df, 5)
        df['ema_50'] = self.calculate_ema(df, 50)
        return df

    def log_stock_status(self, stock_name, df):
        current = df.iloc[-1]
        html = f"""
        <div>
            <h2>{stock_name}</h2>
            <ul>
                <li>Close: {current['close']:.2f}</li>
                <li>VWAP: {current['vwap']:.2f}</li>
                <li>EMA 5: {current['ema_5']:.2f}</li>
                <li>EMA 50: {current['ema_50']:.2f}</li>
                <li>Time: {current.name.strftime('%Y-%m-%d %H:%M')}</li>
            </ul>
        </div>
        """
        self.stock_html_log[stock_name] = html

    def render_stock_status_html(self):
        all_logs = "\n".join(self.stock_html_log.values())
        return stock_status_html.format(table=all_logs)

    def process_stock(self, stock_name):
        try:
            cfg = self.stocks[stock_name]
            df = self.get_historical_data(cfg['symbol'], cfg['token'])
            if not df.empty:
                df = self.calculate_indicators(df)
                self.stock_dataframes[stock_name] = df
                self.log_stock_status(stock_name, df)
        except Exception as e:
            logger.error(f"Error processing {stock_name}: {str(e)}")

    def test_loop(self):
        if not self.login():
            return
        while True:
            for stock in self.stocks:
                self.process_stock(stock)
            time.sleep(60)

if __name__ == "__main__":
    API_KEY = "Ca1mkcLk"
    CLIENT_CODE = "U20441"
    PIN = "2657"
    token = "G5NV6C5WFPRFMUMDQLOUYUKUDA"
    TOTP_TOKEN = pyotp.TOTP(token).now()

    bot = MultiStockTradingBot(API_KEY, CLIENT_CODE, PIN, TOTP_TOKEN)

    print("Bot running with live HTML indicator status")

    def keep_alive():
        port = int(os.environ.get("PORT", 10000))
        server = HTTPServer(("0.0.0.0", port), DummyHandler)
        logger.info(f"Web server started on port {port}")
        server.serve_forever()

    threading.Thread(target=keep_alive, daemon=True).start()
    threading.Thread(target=bot.test_loop).start()

    while True:
        time.sleep(60)
