import time
import datetime
import pyotp
import requests
from SmartApi import SmartConnect
from SmartApi.smartExceptions import SmartAPIException

# --- CONFIGURATION ---
# Replace these with your actual credentials and trading parameters

API_KEY = "Ca1mkcLk"
CLIENT_CODE = "U20441"
PIN = "2657"
TOTP_SECRET = "G5NV6C5WFPRFMUMDQLOUYUKUDA"  # Get this from the SmartAPI portal
SYMBOL_TOKEN = "3045"  # Example: "3045" for SBIN
EXCHANGE = "NSE"
PRODUCT_TYPE = "INTRADAY"
ORDER_VARIETY = "NORMAL"
QUANTITY = 1  # Number of shares to trade
PROFIT_TARGET = 3  # Profit target in Rupees


def get_session():
    """
    Authenticates with the Angel One SmartAPI using TOTP and returns a SmartConnect object.
    """
    try:
        # Initialize the SmartConnect object with the API key
        obj = SmartConnect(api_key=API_KEY)

        # Generate TOTP
        totp = pyotp.TOTP(TOTP_SECRET).now()

        # Authenticate and generate a session token
        data = obj.generateSession(CLIENT_CODE, PIN, totp)

        if data.get("status") is False:
            print(f"Authentication failed: {data.get('message')}")
            return None

        print("Authentication successful.")
        return obj

    except Exception as e:
        print(f"An error occurred during authentication: {e}")
        return None


def get_candle_data(obj, symbol_token, exchange, interval, from_date, to_date):
    """
    Fetches historical candle data for a given instrument.

    Args:
        obj (SmartConnect): The authenticated SmartConnect object.
        symbol_token (str): The token of the trading symbol.
        exchange (str): The exchange (e.g., "NSE").
        interval (str): The candle interval (e.g., "FIFTEEN_MINUTE").
        from_date (str): The start date and time (format: YYYY-MM-DD HH:MM).
        to_date (str): The end date and time (format: YYYY-MM-DD HH:MM).

    Returns:
        dict or None: A dictionary containing the candle data, or None on failure.
    """
    try:
        historic_param = {
            "exchange": exchange,
            "symboltoken": symbol_token,
            "interval": interval,
            "fromdate": from_date,
            "todate": to_date
        }
        historic_data = obj.getCandleData(historic_param)

        if not historic_data.get("data"):
            print("No candle data found for the specified period.")
            return None

        # The API returns a list of lists. We'll use the last one for the latest candle.
        # Format: [timestamp, open, high, low, close, volume]
        return historic_data['data'][-1]

    except SmartAPIException as e:
        print(f"Error fetching candle data: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching candle data: {e}")
        return None


def place_order(obj, symbol_token, exchange, transaction_type, price, quantity):
    """
    Places an order.

    Args:
        obj (SmartConnect): The authenticated SmartConnect object.
        symbol_token (str): The token of the trading symbol.
        exchange (str): The exchange (e.g., "NSE").
        transaction_type (str): "BUY" or "SELL".
        price (float): The price at which to place the order.
        quantity (int): The number of shares.

    Returns:
        bool: True if the order was placed successfully, False otherwise.
    """
    try:
        order_params = {
            "variety": ORDER_VARIETY,
            "tradingsymbol": "SBIN-EQ",  # Use trading symbol for the instrument
            "symboltoken": symbol_token,
            "transactiontype": transaction_type,
            "exchange": exchange,
            "ordertype": "LIMIT",
            "producttype": PRODUCT_TYPE,
            "duration": "DAY",
            "price": price,
            "quantity": quantity
        }

        order_id = obj.placeOrder(order_params)

        if order_id:
            print(f"Successfully placed {transaction_type} order with order ID: {order_id}")
            return True
        else:
            print(f"Failed to place {transaction_type} order.")
            return False

    except SmartAPIException as e:
        print(f"Error placing {transaction_type} order: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while placing order: {e}")
        return False


def morning_trade(obj):
    """
    Executes the morning trading strategy.
    """
    print("Executing morning trade strategy...")

    # Define the time range for the 9:15 to 9:30 candle
    today = datetime.date.today()
    from_time_str = f"{today} 09:15"
    to_time_str = f"{today} 09:30"

    candle = get_candle_data(obj, SYMBOL_TOKEN, EXCHANGE, "FIFTEEN_MINUTE", from_time_str, to_time_str)

    if not candle:
        return

    open_price = candle[1]
    close_price = candle[4]

    sell_price = None

    # Condition 1: if close < open, place sell order at close - 0.5
    if close_price < open_price:
        sell_price = close_price - 0.5
        print(f"Condition 1 met: close ({close_price}) < open ({open_price}). Placing sell order.")
    # Condition 2: if open < close, place sell order at open - 0.5
    elif open_price < close_price:
        sell_price = open_price - 0.5
        print(f"Condition 2 met: open ({open_price}) < close ({close_price}). Placing sell order.")
    else:
        print("No condition met for the morning trade. Candle open and close are equal.")
        return

    if sell_price:
        # Place the sell order
        if place_order(obj, SYMBOL_TOKEN, EXCHANGE, "SELL", sell_price, QUANTITY):
            # Place the corresponding profit-booking buy order
            buy_price = sell_price - PROFIT_TARGET
            print(f"Placing profit-booking buy order at {buy_price}...")
            place_order(obj, SYMBOL_TOKEN, EXCHANGE, "BUY", buy_price, QUANTITY)


def afternoon_trade(obj):
    """
    Executes the afternoon trading strategy.
    """
    print("Executing afternoon trade strategy...")

    # Define the time range for the 1:00 to 1:15 candle
    today = datetime.date.today()
    from_time_str = f"{today} 13:00"
    to_time_str = f"{today} 13:15"

    candle = get_candle_data(obj, SYMBOL_TOKEN, EXCHANGE, "FIFTEEN_MINUTE", from_time_str, to_time_str)

    if not candle:
        return

    open_price = candle[1]
    close_price = candle[4]

    sell_price = None

    # Same conditions as the morning trade
    if close_price < open_price:
        sell_price = close_price - 0.5
        print(f"Condition 1 met: close ({close_price}) < open ({open_price}). Placing sell order.")
    elif open_price < close_price:
        sell_price = open_price - 0.5
        print(f"Condition 2 met: open ({open_price}) < close ({close_price}). Placing sell order.")
    else:
        print("No condition met for the afternoon trade. Candle open and close are equal.")
        return

    if sell_price:
        # Place the sell order
        if place_order(obj, SYMBOL_TOKEN, EXCHANGE, "SELL", sell_price, QUANTITY):
            # The prompt is slightly ambiguous about the afternoon buy order.
            # Assuming it's a profit-booking order with the same logic as the morning trade.
            buy_price = sell_price - PROFIT_TARGET
            print(f"Placing profit-booking buy order at {buy_price}...")
            place_order(obj, SYMBOL_TOKEN, EXCHANGE, "BUY", buy_price, QUANTITY)


def main():
    """
    The main function to run the trading bot.
    """
    angel = get_session()
    if not angel:
        print("Failed to get a trading session. Exiting.")
        return

    morning_executed = False
    afternoon_executed = False

    while True:
        now = datetime.datetime.now().time()

        # Check for morning trade time (9:30:05)
        if now.hour == 9 and now.minute == 30 and now.second == 5 and not morning_executed:
            print("It's 9:30:05. Time to execute the morning trade.")
            morning_trade(angel)
            morning_executed = True

        # Check for afternoon trade time (13:15:02)
        elif now.hour == 13 and now.minute == 15 and now.second == 2 and not afternoon_executed:
            print("It's 13:15:02. Time to execute the afternoon trade.")
            afternoon_trade(angel)
            afternoon_executed = True

        # Exit the loop after both trades have been attempted
        if morning_executed and afternoon_executed:
            print("Both trades for the day have been executed. Exiting the script.")
            break

        time.sleep(1)  # Wait for 1 second before checking the time again


if __name__ == "__main__":
    main()

