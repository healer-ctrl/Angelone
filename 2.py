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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        
        # Multiple stocks configuration
        self.stocks = {
            "SBIN": {"symbol": "SBIN-EQ", "token": "3045", "quantity": 1},
            "NIFTY31JUL25FUT": {"symbol":"NIFTY31JUL25FUT","token":"53216","quantity": 1}
            # Add more stocks as needed
        }
        
        self.exchange = "NSE"
        self.timeframe = "FIVE_MINUTE"
        
        # Track positions and signals for each stock
        self.positions = {stock: False for stock in self.stocks}
        self.last_signal_times = {stock: None for stock in self.stocks}
        self.stock_dataframes = {stock: pd.DataFrame() for stock in self.stocks}
        
        
        # Thread pool for concurrent processing
        self.thread_pool = ThreadPoolExecutor(max_workers=len(self.stocks))
        self.signal_queue = queue.Queue()

    def setup_email(self):
        """Setup email configuration"""
        try:
            self.smtp_server = smtplib.SMTP('smtp.gmail.com', 587)
            self.smtp_server.starttls()
            self.smtp_server.login("madhu17702@gmail.com", "ndqx cnsi yqoc pgwm")
            self.email_from = "madhu17702@gmail.com"
            self.email_to = "msr459@gmail.com"
        except Exception as e:
            logger.error(f"Email setup failed: {str(e)}")
            self.smtp_server = None

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

    def get_historical_data(self, symbol, token, days=30):
        """
        Fetch historical data for calculating initial indicators
        """
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
                historicParam = {
                    "exchange": "NFO",
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
        """
        if len(df) < 2:
            return False
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Check if previous candle closed below VWAP
        prev_close_above_vwap = previous['close'] < previous['vwap']
        
        # Check for EMA crossover (5 EMA crossing above 50 EMA)
        current_5_above_50 = current['ema_5'] <= current['ema_50']
        ema_crossover = current_5_above_50
        
        # Check if closing price is above both crossover point and VWAP
        close_above_crossover = current['ema_50'] > current['close'] and current['ema_5'] > current['close']
        close_above_vwap = current['close'] < current['vwap']
        
        sell_signal = ema_crossover and close_above_vwap and close_above_crossover
        
        if sell_signal:
            logger.info(f"SELL SIGNAL DETECTED for {stock_name}!")
            logger.info(f"Previous close: {previous['close']:.2f}, Previous VWAP: {previous['vwap']:.2f}")
            logger.info(f"Current EMA5: {current['ema_5']:.2f}, Current EMA50: {current['ema_50']:.2f}")
            logger.info(f"Current close: {current['close']:.2f}")
        
        return sell_signal

    def check_buy_signal(self, df, stock_name):
        """
        Check for buy signal based on strategy
        """
        if len(df) < 2:
            return False, {}
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # Check if previous candle closed above VWAP
        prev_close_above_vwap = previous['close'] > previous['vwap']
        
        # Check for EMA crossover (5 EMA crossing above 50 EMA)
        current_5_above_50 = current['ema_5'] > current['ema_50']
        previous_5_below_50 = previous['ema_5'] <= previous['ema_50']
        ema_crossover = current_5_above_50 and previous_5_below_50
        ema_crossover = current_5_above_50
        
        # Check if closing price is above both crossover point and VWAP
        close_above_crossover = current['ema_50'] < current['close'] and current['ema_5'] < current['close']
        close_above_vwap = current['close'] > current['vwap']
        
        buy_signal = ema_crossover and close_above_vwap and close_above_crossover
        
        signal_data = {
            'timestamp': current.name,
            'close': current['close'],
            'vwap': current['vwap'],
            'ema_5': current['ema_5'],
            'ema_50': current['ema_50'],
            'prev_close_above_vwap': prev_close_above_vwap,
            'ema_crossover': ema_crossover,
            'close_above_crossover': close_above_crossover,
            'close_above_vwap': close_above_vwap,
            'buy_signal': buy_signal
        }
        
        if buy_signal:
            logger.info(f"BUY SIGNAL DETECTED for {stock_name}!")
            logger.info(f"Time: {current.name}")
            logger.info(f"Previous close: {previous['close']:.2f}, Previous VWAP: {previous['vwap']:.2f}")
            logger.info(f"Current EMA5: {current['ema_5']:.2f}, Current EMA50: {current['ema_50']:.2f}")
            logger.info(f"Current close: {current['close']:.2f}, Current VWAP: {current['vwap']:.2f}")
        
        return buy_signal, signal_data

    def place_order(self, stock_name, order_type, price):
        """
        Place order through Angel One API
        """
        try:
            stock_config = self.stocks[stock_name]
            
            orderparams = {
                "variety": "NORMAL",
                "tradingsymbol": stock_config["symbol"],
                "symboltoken": stock_config["token"],
                "transactiontype": order_type,
                "exchange": self.exchange,
                "ordertype": "LIMIT",
                "producttype": "INTRADAY",
                "duration": "DAY",
                "price": str(price),
                "quantity": str(stock_config["quantity"])
            }
            
            response = self.smartApi.placeOrder(orderparams)
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
            ltp_data = self.smartApi.ltpData(self.exchange, stock_config["symbol"], stock_config["token"])
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
            if self.smtp_server:
                full_message = f"Subject: {subject}\n\n{message}"
                self.smtp_server.sendmail(self.email_from, self.email_to, full_message)
                logger.info(f"Email sent: {subject}")
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")

    def process_stock(self, stock_name):
        """
        Process a single stock for signals
        """
        try:
            stock_config = self.stocks[stock_name]
            
            # Get fresh historical data
            df = self.get_historical_data(stock_config["symbol"], stock_config["token"], days=10)
            
            if not df.empty:
                # Calculate indicators
                df = self.calculate_indicators(df)
                self.stock_dataframes[stock_name] = df
                
                # Check for sell signal
                if self.check_sell_signal(df, stock_name) and not self.positions[stock_name]:
                    current_price = self.get_ltp(stock_name)
                    if current_price:
                        if self.place_order(stock_name, "SELL", current_price):
                            self.last_signal_times[stock_name] = datetime.now()
                            self.send_email_alert(f"SELL Signal - {stock_name}", 
                                                f"Sell signal detected for {stock_name} at price {current_price}")
                
                # Check for buy signal
                buy_signal, signal_data = self.check_buy_signal(df, stock_name)
                if buy_signal:
                    current_price = self.get_ltp(stock_name)
                    if current_price:
                        self.send_email_alert(f"BUY Signal - {stock_name}", 
                                            f"Buy signal detected for {stock_name} at price {current_price}")
                
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
            df = self.get_historical_data(stock_config["symbol"], stock_config["token"])
            if not df.empty:
                df = self.calculate_indicators(df)
                self.stock_dataframes[stock_name] = df
            else:
                logger.error(f"Failed to load initial data for {stock_name}")
        
        # Main trading loop
        while True:
            try:
                current_time = datetime.now()
                
                # Check if market is open (9:15 AM to 3:30 PM IST)
                if current_time.hour >= 9 and current_time.hour < 15:
                    if current_time.hour == 9 and current_time.minute < 15:
                        pass  # Market not open yet
                    else:
                        # Update data every 15 minutes
                        if current_time.minute % 15 == 0:
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
            self.run_strategy()
        else:
            logger.error("Failed to login. Cannot start trading.")

    def test_indicators(self):
        """
        Test function to verify indicator calculations for all stocks
        """
        if not self.login():
            return
        
        while True:
            try:
                current_time = datetime.now()
                
                # Check if market is open (9:15 AM to 3:30 PM IST)
                if current_time.hour >= 9 and current_time.hour < 15:
                    if current_time.hour == 9 and current_time.minute < 15:
                        pass  # Market not open yet
                    else:
                        # Update data every 15 minutes
                        if current_time.minute % 5 == 0:
                            logger.info(f"Testing indicators for {len(self.stocks)} stocks...")
                            
                            for stock_name in self.stocks:
                                stock_config = self.stocks[stock_name]
                                df = self.get_historical_data(stock_config["symbol"], stock_config["token"], days=5)
                                
                                if not df.empty:
                                    df = self.calculate_indicators(df)
                                    print(f"\n{stock_name} - Last 3 candles with indicators:")
                                    print(df[['open', 'high', 'low', 'close', 'volume', 'vwap', 'ema_5', 'ema_50']].tail(3))
                                    
                                    # Check signals
                                    sell_signal = self.check_sell_signal(df, stock_name)
                                    buy_signal, _ = self.check_buy_signal(df, stock_name)
                                    self.setup_email()
                                    # self.send_email_alert(f"BUY Signal - {stock_name}", "hello")
                                    if sell_signal:
                                        message = f"SELL signal detected for {stock_name}"
                                        self.send_email_alert(f"SELL Signal - {stock_name}", message)
                                    
                                    if buy_signal:
                                        message = f"BUY signal detected for {stock_name}"
                                        self.send_email_alert(f"BUY Signal - {stock_name}", message)
                                    bot.smtp_server.quit()
                                    print(f"{stock_name} - Sell signal: {sell_signal}, Buy signal: {buy_signal}")
                            
                            time.sleep(60)  # Wait 1 minute to avoid multiple executions
                else:
                    print(f"Market closed, current time: {current_time}")
                    time.sleep(30)
                    
            except KeyboardInterrupt:
                logger.info("Testing stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in testing: {str(e)}")
                time.sleep(60)

    def add_stock(self, stock_name, symbol, token, quantity=1):
        """
        Add a new stock to monitor
        """
        self.stocks[stock_name] = {
            "symbol": symbol,
            "token": token,
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

    def get_portfolio_status(self):
        """
        Get current status of all monitored stocks
        """
        status = {}
        for stock_name in self.stocks:
            if stock_name in self.stock_dataframes and not self.stock_dataframes[stock_name].empty:
                df = self.stock_dataframes[stock_name]
                current = df.iloc[-1]
                status[stock_name] = {
                    'close': current['close'],
                    'vwap': current['vwap'],
                    'ema_5': current['ema_5'],
                    'ema_50': current['ema_50'],
                    'position_open': self.positions[stock_name],
                    'last_signal_time': self.last_signal_times[stock_name]
                }
        return status


# Example usage
if __name__ == "__main__":
    # Angel One API credentials - Replace with your actual credentials
    API_KEY = "Ca1mkcLk"
    CLIENT_CODE = "U20441"
    PIN = "2657"
    TOTP_TOKEN = "921710"
    
    # Create multi-stock trading bot
    bot = MultiStockTradingBot(API_KEY, CLIENT_CODE, PIN, TOTP_TOKEN)
    
    print("=== Multi-Stock Angel One Trading Bot ===")
    print("Strategy: VWAP + EMA Crossover")
    print("Timeframe: 15 minutes")
    print("Indicators: VWAP, EMA(5), EMA(50)")
    print("Monitoring stocks:", list(bot.stocks.keys()))
    print("=========================================")
    
    # Add more stocks if needed
    # bot.add_stock("ICICIBANK", "ICICIBANK-EQ", "1330", quantity=1)
    # bot.add_stock("AXISBANK", "AXISBANK-EQ", "5900", quantity=1)
    
    try:
        # Start testing indicators for all stocks
        bot.test_indicators()
        
        # Uncomment to start actual trading
        # bot.start_trading()
        
    except Exception as e:
        logger.error(f"Bot crashed: {str(e)}")
    finally:
        # Clean up email connection
        if hasattr(bot, 'smtp_server') and bot.smtp_server:
            bot.smtp_server.quit()