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
import os
import pyotp
from http.server import BaseHTTPRequestHandler, HTTPServer
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
from zoneinfo import ZoneInfo
IST = ZoneInfo("Asia/Kolkata")
current_time = datetime.now(IST)


class DummyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running!")
class MultiStockTradingBot:
    def __init__(self, api_key, client_code, pin, totp_token):
        """
        Initialize the trading bot with Angel One credentials
        """
        self.api_key = api_key
        self.client_code = client_code
        self.pin = pin
        self.totp_token = totp_token
        self.smartApi = SmartConnect(api_key=api_key)
        self.auth_token = None
        self.feed_token = None
        self.refresh_token = None
        self.last_processed_minute = None
        
        # Multiple stocks configuration
        self.stocks = {
            "SBIN": {"symbol": "SBIN-EQ", "token": "3045", "quantity": 1, "exchange": "NSE"}
        }
        
        self.timeframe = "FIFTEEN_MINUTE"
        
        # Track positions and signals for each stock
        self.positions = {stock: False for stock in self.stocks}
        self.last_signal_times = {stock: None for stock in self.stocks}
        self.stock_dataframes = {stock: pd.DataFrame() for stock in self.stocks}
        
        # Thread pool for concurrent processing
        self.thread_pool = ThreadPoolExecutor(max_workers=len(self.stocks))
        self.signal_queue = queue.Queue()
        
        # Email configuration
        self.smtp_server = None

    def setup_email(self):
        """Setup email configuration"""
        try:
            if self.smtp_server:
                self.smtp_server.quit()
            
            self.smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
            self.smtp_server.starttls()
            self.smtp_server.login("madhu17702@gmail.com", "ndqx cnsi yqoc pgwm")
            self.email_from = "madhu17702@gmail.com"
            self.email_to = "msr459@gmail.com"
            return True
        except Exception as e:
            logger.error(f"Email setup failed: {str(e)}")
            self.smtp_server = None
            return False

    def login(self):
        """
        Login to Angel One API
        """
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

    def get_historical_data(self, symbol, token, exchange, days=30):
        """
        Fetch historical data for calculating initial indicators
        """
        try:
            to_date = datetime.now(IST).strftime("%Y-%m-%d %H:%M")
            from_date = (datetime.now(IST) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
            print(to_date)
            print(from_date)
            historicParam = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": self.timeframe,
                "fromdate": from_date,
                "todate": to_date
            }
            
            response = self.smartApi.getCandleData(historicParam)
            if response['status']:
                data = response['data']
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                
                # Convert to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                
                return df
            else:
                logger.error(f"Failed to fetch historical data for {symbol}: {response['message']}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def calculate_vwap(self, df):
        """
        Calculate Volume Weighted Average Price (VWAP)
        """
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap_numerator'] = df['typical_price'] * df['volume']
        df['vwap_denominator'] = df['volume']
        
        # Calculate cumulative VWAP for each trading day
        df['date'] = df.index.date
        df['cumulative_vwap_num'] = df.groupby('date')['vwap_numerator'].cumsum()
        df['cumulative_vwap_den'] = df.groupby('date')['vwap_denominator'].cumsum()
        df['vwap'] = df['cumulative_vwap_num'] / df['cumulative_vwap_den']
        
        return df

    def calculate_ema(self, df, period):
        """
        Calculate Exponential Moving Average
        """
        return df['close'].ewm(span=period, adjust=False).mean()

    def calculate_indicators(self, df):
        """
        Calculate all technical indicators
        """
        # Calculate VWAP
        df = self.calculate_vwap(df)
        
        # Calculate EMAs
        df['ema_5'] = self.calculate_ema(df, 5)
        df['ema_50'] = self.calculate_ema(df, 50)
        
        return df

    def check_sell_signal(self, df, stock_name):
        """
        Check for sell signal based on strategy
        Fixed logic: Sell when EMA5 crosses below EMA50 and price is below VWAP
        """
        if len(df) < 2:
            return False, {}
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Check for EMA bearish crossover (5 EMA crossing below 50 EMA)
        current_5_below_50 = current['ema_5'] < current['ema_50']
        previous_5_above_50 = previous['ema_5'] >= previous['ema_50']
        ema_bearish_crossover = current_5_below_50 and previous_5_above_50
        
        # Alternative: Just check if EMA5 is below EMA50 (more signals)
        ema_bearish_condition = current_5_below_50
        
        # Check if price conditions are met
        close_below_vwap = current['close'] < current['vwap']
        close_below_emas = current['close'] < current['ema_5'] and current['close'] < current['ema_50']
        
        # Sell signal conditions
        sell_signal = ema_bearish_condition and close_below_vwap and close_below_emas
        
        signal_data = {
            'timestamp': current.name,
            'close': current['close'],
            'vwap': current['vwap'],
            'ema_5': current['ema_5'],
            'ema_50': current['ema_50'],
            'ema_bearish_crossover': ema_bearish_crossover,
            'ema_bearish_condition': ema_bearish_condition,
            'close_below_vwap': close_below_vwap,
            'close_below_emas': close_below_emas,
            'sell_signal': sell_signal
        }
        
        if sell_signal:
            logger.info(f"SELL SIGNAL DETECTED for {stock_name}!")
            logger.info(f"Time: {current.name}")
            logger.info(f"Current EMA5: {current['ema_5']:.2f}, Current EMA50: {current['ema_50']:.2f}")
            logger.info(f"Current close: {current['close']:.2f}, Current VWAP: {current['vwap']:.2f}")
        
        return sell_signal, signal_data

    def check_buy_signal(self, df, stock_name):
        """
        Check for buy signal based on strategy
        Fixed logic: Buy when EMA5 crosses above EMA50 and price is above VWAP
        """
        if len(df) < 2:
            return False, {}
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Check for EMA bullish crossover (5 EMA crossing above 50 EMA)
        current_5_above_50 = current['ema_5'] > current['ema_50']
        previous_5_below_50 = previous['ema_5'] <= previous['ema_50']
        ema_bullish_crossover = current_5_above_50 and previous_5_below_50
        
        # Alternative: Just check if EMA5 is above EMA50 (more signals)
        ema_bullish_condition = current_5_above_50
        
        # Check if price conditions are met
        close_above_vwap = current['close'] > current['vwap']
        close_above_emas = current['close'] > current['ema_5'] and current['close'] > current['ema_50']
        
        # Buy signal conditions
        buy_signal = ema_bullish_condition and close_above_vwap and close_above_emas
        
        signal_data = {
            'timestamp': current.name,
            'close': current['close'],
            'vwap': current['vwap'],
            'ema_5': current['ema_5'],
            'ema_50': current['ema_50'],
            'ema_bullish_crossover': ema_bullish_crossover,
            'ema_bullish_condition': ema_bullish_condition,
            'close_above_vwap': close_above_vwap,
            'close_above_emas': close_above_emas,
            'buy_signal': buy_signal
        }
        
        if buy_signal:
            logger.info(f"BUY SIGNAL DETECTED for {stock_name}!")
            logger.info(f"Time: {current.name}")
            logger.info(f"Current EMA5: {current['ema_5']:.2f}, Current EMA50: {current['ema_50']:.2f}")
            logger.info(f"Current close: {current['close']:.2f}, Current VWAP: {current['vwap']:.2f}")
        
        return buy_signal, signal_data

    def place_order(self, stock_name, order_type, price):
        """
        Place order through Angel One API
        """
        try:
            stock_config = self.stocks[stock_name]
            logger.info(stock_config)
            orderparams = {
                "variety": "NORMAL",
                "tradingsymbol": stock_config["symbol"],
                "symboltoken": stock_config["token"],
                "transactiontype": order_type,
                "exchange": stock_config["exchange"],
                "ordertype": "LIMIT",
                "producttype": "INTRADAY",
                "duration": "DAY",
                "price": str(price),
                "quantity": str(stock_config["quantity"])
            }
            
            response = self.smartApi.placeOrder(orderparams)
            if response['status']:
                logger.info(f"{order_type} order placed successfully for {stock_name}! Order ID: {response['data']['orderid']}")
                if order_type == "BUY":
                    self.positions[stock_name] = True
                elif order_type == "SELL":
                    self.positions[stock_name] = False
                return True
            else:
                logger.error(f"Failed to place {order_type} order for {stock_name}: {response['message']}")
                return False
        except Exception as e:
            logger.error(f"Error placing {order_type} order for {stock_name}: {str(e)}")
            return False
    def place_order_sell(self, stock_name, order_type, price):
        """
        Place order through Angel One API
        """
        try:
            stock_config = self.stocks[stock_name]
            logger.info(stock_config)
            orderparams = {
                "variety": "NORMAL",
                "tradingsymbol": stock_config["symbol"],
                "symboltoken": stock_config["token"],
                "transactiontype": order_type,
                "exchange": stock_config["exchange"],
                "ordertype": "LIMIT",
                "producttype": "INTRADAY",
                "duration": "DAY",
                "price": str(price),
                "quantity": str(stock_config["quantity"])
            }
            
            response = self.smartApi.placeOrderFullResponse(orderparams)
            if response['status']:
                logger.info(f"{order_type} order placed successfully for {stock_name}! Order ID: {response['data']['orderid']}")
                if order_type == "SELL":
                    self.positions[stock_name] = True
                return True
            else:
                logger.error(f"Failed to place {order_type} order for {stock_name}: {response['message']}")
                return False
        except Exception as e:
            logger.error(f"Error placing {order_type} order for {stock_name}: {str(e)}")
            return False
    def place_order_buy(self, stock_name, order_type, price):
        """
        Place order through Angel One API
        """
        try:
            logger.info(self.stocks)
            stock_config = self.stocks[stock_name]
            logger.info(stock_config["token"])
            orderparams = {
                "variety": "NORMAL",
                "tradingsymbol": stock_config["symbol"],
                "symboltoken": stock_config["token"],
                "transactiontype": order_type,
                "exchange": stock_config["exchange"],
                "ordertype": "LIMIT",
                "producttype": "DELIVERY",
                "duration": "DAY",
                "price": str(price),
                "quantity": str(stock_config["quantity"])
            }
            
            response = self.smartApi.placeOrderFullResponse(orderparams)
            logger.info(response['status'])
            if response['status']:
                logger.info(f"{order_type} order placed successfully for {stock_name}! Order ID: {response['data']['orderid']}")
                if order_type == "SELL":
                    self.positions[stock_name] = True
                return True
            else:
                logger.error(f"Failed to place {order_type} order for {stock_name}: {response['message']}")
                return False
        except Exception as e:
            logger.error(f"Error placing {order_type} order for {stock_name}: {str(e)}")
            return False
    def get_ltp(self, stock_name):
        """
        Get Last Traded Price for a specific stock
        """
        try:
            stock_config = self.stocks[stock_name]
            ltp_data = self.smartApi.ltpData(stock_config["exchange"], stock_config["symbol"], stock_config["token"])
            if ltp_data['status']:
                return float(ltp_data['data']['ltp'])
            else:
                logger.error(f"Failed to get LTP for {stock_name}: {ltp_data['message']}")
                return None
        except Exception as e:
            logger.error(f"Error getting LTP for {stock_name}: {str(e)}")
            return None

    def send_email_alert(self, subject, message):
        """
        Send email alert
        """
        try:
            if not self.smtp_server:
                if not self.setup_email():
                    return False
            
            full_message = f"Subject: {subject}\n\n{message}"
            self.smtp_server.sendmail(self.email_from, self.email_to, full_message)
            logger.info(f"Email sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            # Try to reconnect
            self.setup_email()
            return False

    def is_market_open(self):
        """
        Check if market is currently open
        """
        current_time = datetime.now(IST)
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        # Market hours: 9:15 AM to 3:30 PM IST
        market_start = 9 * 60 + 15  # 9:15 AM in minutes
        market_end = 15 * 60 + 30   # 3:30 PM in minutes
        current_time_minutes = current_hour * 60 + current_minute
        
        # Check if it's a weekday (Monday = 0, Sunday = 6)
        if current_time.weekday() >= 5:  # Saturday or Sunday
            return False
        
        return market_start <= current_time_minutes <= market_end

    def process_stock(self, stock_name):
        """
        Process a single stock for signals
        """
        try:
            print("Checking")
            stock_config = self.stocks[stock_name]
            
            # Get fresh historical data
            df = self.get_historical_data(stock_config["symbol"], stock_config["token"], 
                                        stock_config["exchange"], days=10)
            
            if not df.empty:
                # Calculate indicators
                df = self.calculate_indicators(df)
                self.stock_dataframes[stock_name] = df
                
                # Check for buy signal
                buy_signal, buy_data = self.check_buy_signal(df, stock_name)

                if buy_signal:
                    current_price = self.get_ltp(stock_name)
                    # if current_price:
                    #     if self.place_order_buy(stock_name, "BUY", current_price):
                    #         self.last_signal_times[stock_name] = datetime.now()
                    #         print("Placed order")
                    self.setup_email()
                    self.send_email_alert(f"BUY Signal - {stock_name}", 
                                                f"Buy signal detected for {stock_name} at price {current_price}")
                
                # Check for sell signal
                sell_signal, sell_data = self.check_sell_signal(df, stock_name)
                if sell_signal :
                    current_price = self.get_ltp(stock_name)
                    # if current_price:
                    #     if self.place_order_sell(stock_name, "SELL", current_price):
                    #         self.last_signal_times[stock_name] = datetime.now()
                    #         print("Placed order")
                    self.setup_email()
                    self.send_email_alert(f"SELL Signal - {stock_name}", 
                                                f"Sell signal detected for {stock_name} at price {current_price}")
                
                # Log current status
                current = df.iloc[-1]
                logger.info(f"{stock_name} - Close: {current['close']:.2f}, VWAP: {current['vwap']:.2f}, "
                           f"EMA5: {current['ema_5']:.2f}, EMA50: {current['ema_50']:.2f}")
                
        except Exception as e:
            logger.error(f"Error processing {stock_name}: {str(e)}")

    def run_strategy(self):
        """
        Main strategy execution loop for multiple stocks
        """
        logger.info("Starting multi-stock trading strategy...")
        
        # Initial data load for all stocks
        for stock_name in self.stocks:
            stock_config = self.stocks[stock_name]
            df = self.get_historical_data(stock_config["symbol"], stock_config["token"], 
                                        stock_config["exchange"])
            if not df.empty:
                df = self.calculate_indicators(df)
                self.stock_dataframes[stock_name] = df
            else:
                logger.error(f"Failed to load initial data for {stock_name}")
        
        # Main trading loop
        while True:
            try:
                if self.is_market_open():
                    current_time = datetime.now(IST)
                    logger.info(f"Waiting for 15 min interval... current time: {current_time}")
                    
                    # Update data every 5 minutes
                    current_bucket = (current_time.hour * 60 + current_time.minute)
                    if current_bucket != self.last_processed_minute:
                        self.last_processed_minute=current_bucket

                        logger.info(f"Processing signals for {len(self.stocks)} stocks...")
                        
                        # Process all stocks concurrently
                        futures = []
                        for stock_name in self.stocks:
                            future = self.thread_pool.submit(self.process_stock, stock_name)
                            futures.append(future)
                        
                        # Wait for all tasks to complete
                        for future in futures:
                            future.result()
                        
                        logger.info("Signal processing completed for all stocks")
                        time.sleep(60)  # Wait 1 minute to avoid multiple executions
                else:
                    logger.info("Market is closed. Waiting...")
                    time.sleep(300)  # Wait 5 minutes when market is closed
                
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                logger.info("Strategy stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in strategy execution: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying

    def start_trading(self):
        """
        Start the trading bot
        """
        if self.login():
            self.setup_email()
            self.run_strategy()
        else:
            logger.error("Failed to login. Cannot start trading.")
        

    def test_indicators(self):
        """
        Test function to verify indicator calculations for all stocks
        """
        if not self.login():
            return
        
        self.setup_email()
        
        try:
            while True:
                if self.is_market_open():
                    current_time = datetime.now(IST)
                    
                    # Update data every 5 minutes
                    if current_time.minute % 5 == 0:
                        logger.info(f"Testing indicators for {len(self.stocks)} stocks...")
                        
                        for stock_name in self.stocks:
                            stock_config = self.stocks[stock_name]
                            df = self.get_historical_data(stock_config["symbol"], stock_config["token"], 
                                                        stock_config["exchange"], days=5)
                            
                            if not df.empty:
                                df = self.calculate_indicators(df)
                                print(f"\n{stock_name} - Last 3 candles with indicators:")
                                print(df[['open', 'high', 'low', 'close', 'volume', 'vwap', 'ema_5', 'ema_50']].tail(3))
                                
                                # Check signals
                                sell_signal, _ = self.check_sell_signal(df, stock_name)
                                buy_signal, _ = self.check_buy_signal(df, stock_name)
                                
                                if sell_signal:
                                    message = f"SELL signal detected for {stock_name}"
                                    self.send_email_alert(f"SELL Signal - {stock_name}", message)
                                
                                if buy_signal:
                                    message = f"BUY signal detected for {stock_name}"
                                    self.send_email_alert(f"BUY Signal - {stock_name}", message)
                                
                                print(f"{stock_name} - Sell signal: {sell_signal}, Buy signal: {buy_signal}")
                        
                        time.sleep(60)  # Wait 1 minute to avoid multiple executions
                else:
                    current_time = datetime.now(IST)
                    print(f"Market closed, current time: {current_time}")
                    time.sleep(300)  # Wait 5 minutes when market is closed
                    
        except KeyboardInterrupt:
            logger.info("Testing stopped by user")
        except Exception as e:
            logger.error(f"Error in testing: {str(e)}")
        finally:
            if self.smtp_server:
                self.smtp_server.quit()

    def backtest_strategy(self, days=30):
        """
        Backtest the strategy on historical data
        """
        print(f"\n=== BACKTESTING STRATEGY ===")
        print(f"Analyzing last {days} days of data...")
        print(f"Timeframe: {self.timeframe}")
        print("=" * 50)
        
        if not self.login():
            return
        
        all_results = {}
        
        for stock_name in self.stocks:
            stock_config = self.stocks[stock_name]
            print(f"\n--- Backtesting {stock_name} ---")
            
            # Get historical data
            df = self.get_historical_data(stock_config["symbol"], stock_config["token"], 
                                        stock_config["exchange"], days=days)
            
            if df.empty:
                print(f"No historical data available for {stock_name}")
                continue
            
            # Calculate indicators
            df = self.calculate_indicators(df)
            df.to_csv("df.csv", index=False)
            # Find all signals
            buy_signals = []
            sell_signals = []
            
            for i in range(50, len(df)):  # Start from 50 to have enough data for indicators
                temp_df = df.iloc[:i+1]
                
                buy_signal, buy_data = self.check_buy_signal(temp_df, stock_name)
                if buy_signal:
                    buy_signals.append(buy_data)
                
                sell_signal, sell_data = self.check_sell_signal(temp_df, stock_name)
                print(sell_signal,temp_df)
                if sell_signal:
                    sell_signals.append(sell_data)
            
            # Store results
            all_results[stock_name] = {
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'dataframe': df
            }
            
            # Analyze results for this stock
            self.analyze_backtest_results(buy_signals, sell_signals, df, stock_name)
        
        return all_results

    def analyze_backtest_results(self, buy_signals, sell_signals, df, stock_name):
        """
        Analyze and display backtesting results
        """
        print(f"\n=== BACKTESTING RESULTS for {stock_name} ===")
        print(f"Total Buy Signals: {len(buy_signals)}")
        print(f"Total Sell Signals: {len(sell_signals)}")
        
        # Analyze buy signals
        if buy_signals:
            buy_df = pd.DataFrame(buy_signals)
            print(f"\nBuy Signal Analysis:")
            print(f"Average buy price: ₹{buy_df['close'].mean():.2f}")
            print(f"Highest buy price: ₹{buy_df['close'].max():.2f}")
            print(f"Lowest buy price: ₹{buy_df['close'].min():.2f}")
            
            # Show recent buy signals
            print(f"\nRecent Buy Signals (last 5):")
            for signal in buy_signals[-5:]:
                timestamp = signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                print(f"  {timestamp} | Price: ₹{signal['close']:.2f}")
        
        # Analyze sell signals
        if sell_signals:
            sell_df = pd.DataFrame(sell_signals)
            print(f"\nSell Signal Analysis:")
            print(f"Average sell price: ₹{sell_df['close'].mean():.2f}")
            print(f"Highest sell price: ₹{sell_df['close'].max():.2f}")
            print(f"Lowest sell price: ₹{sell_df['close'].min():.2f}")
            
            # Show recent sell signals
            print(f"\nRecent Sell Signals (last 5):")
            for signal in sell_signals[-5:]:
                timestamp = signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                print(f"  {timestamp} | Price: ₹{signal['close']:.2f}")
        
        # Save results
        self.save_backtest_results(buy_signals, sell_signals, stock_name)

    def save_backtest_results(self, buy_signals, sell_signals, stock_name):
        """
        Save backtest results to CSV files
        """
        try:
            timestamp = datetime.now(IST).strftime('%Y%m%d_%H%M%S')
            
            if buy_signals:
                buy_df = pd.DataFrame(buy_signals)
                buy_filename = f"buy_signals_{stock_name}_{timestamp}.csv"
                buy_df.to_csv(buy_filename, index=False)
                print(f"Buy signals saved to: {buy_filename}")
            
            if sell_signals:
                sell_df = pd.DataFrame(sell_signals)
                sell_filename = f"sell_signals_{stock_name}_{timestamp}.csv"
                sell_df.to_csv(sell_filename, index=False)
                print(f"Sell signals saved to: {sell_filename}")
                
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

    def add_stock(self, stock_name, symbol, token, exchange, quantity=1):
        """
        Add a new stock to monitor
        """
        self.stocks[stock_name] = {
            "symbol": symbol,
            "token": token,
            "exchange": exchange,
            "quantity": quantity
        }
        self.positions[stock_name] = False
        self.last_signal_times[stock_name] = None
        self.stock_dataframes[stock_name] = pd.DataFrame()
        logger.info(f"Added {stock_name} to monitoring list")

    def remove_stock(self, stock_name):
        """
        Remove a stock from monitoring
        """
        if stock_name in self.stocks:
            del self.stocks[stock_name]
            del self.positions[stock_name]
            del self.last_signal_times[stock_name]
            del self.stock_dataframes[stock_name]
            logger.info(f"Removed {stock_name} from monitoring list")


# Example usage
if __name__ == "__main__":
    # Angel One API credentials - Replace with your actual credentials
    API_KEY = "Ca1mkcLk"
    CLIENT_CODE = "U20441"
    PIN = "2657"
    TOKEN = "G5NV6C5WFPRFMUMDQLOUYUKUDA"
    TOTP_TOKEN=pyotp.TOTP(TOKEN).now()
    # Create multi-stock trading bot
    bot = MultiStockTradingBot(API_KEY, CLIENT_CODE, PIN, TOTP_TOKEN)
    
    print("=== Multi-Stock Angel One Trading Bot ===")
    print("Strategy: VWAP + EMA Crossover")
    print("Timeframe: 5 minutes")
    print("Indicators: VWAP, EMA(5), EMA(50)")
    print("Monitoring stocks:", list(bot.stocks.keys()))
    print("=========================================")
    
    # Add more stocks if needed
    # bot.add_stock("ICICIBANK", "ICICIBANK-EQ", "1330", "NSE", quantity=1)
    # bot.add_stock("AXISBANK", "AXISBANK-EQ", "5900", "NSE", quantity=1)
    
    def keep_alive():
        port = int(os.environ.get("PORT", 10000))  # Render will set PORT
        server = HTTPServer(("0.0.0.0", port), DummyHandler)
        print(f"Web server started on port {port}")
        server.serve_forever()

    threading.Thread(target=keep_alive, daemon=True).start()

    # Start your trading bot logic in a background thread
    def start_bot():
        try:
            bot.start_trading()  # Or use bot.start_trading() for real trading
        except Exception as e:
            logger.error(f"Bot crashed: {str(e)}")
        finally:
            if hasattr(bot, 'smtp_server') and bot.smtp_server:
                bot.smtp_server.quit()

    threading.Thread(target=start_bot).start()

    # Prevent script from exiting
    while True:
        time.sleep(60)
