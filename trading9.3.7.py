import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from scipy.signal import argrelextrema
from enum import Enum
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

class BotLogger:
    def __init__(self):
        self.log = []

    def record(self, event, msg):
        self.log.append((event, msg))
    
    def output_log(self, print_log=True, save_to_file="trade_log.txt"):
        output = '\n'.join([f"{event}: {msg}" for event, msg in self.log])
        if print_log:
            print(output)
        if save_to_file:
            with open(save_to_file, 'w') as f:
                f.write(output)

class Decision(Enum):
    BUY = 1
    SELL = 2
    HOLD = 3

class MarketDataCollector:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def fetch_market_indices(self):
        # Define the market indices you want to track, e.g., S&P 500, NASDAQ, etc.
        indices = ["^GSPC", "^IXIC", "^DJI", "TQQQ", "SQQQ", "^VIX"]  # S&P 500, NASDAQ, Dow Jones
        indices_data = yf.download(indices, start=self.start_date, end=self.end_date)['Close']
        return indices_data

    def fetch_economic_indicators(self):
        # Define the symbols for the S&P 500, NASDAQ, Dow Jones, TQQQ, and SQQQ
        symbols = ['^GSPC', '^IXIC', '^DJI', 'TQQQ', 'SQQQ', '^VIX']
        economic_data = yf.download(symbols, start=self.start_date, end=self.end_date)['Close']
        return economic_data.ffill()

    def aggregate_data(self):
        market_indices = self.fetch_market_indices()
        economic_indicators = self.fetch_economic_indicators()

        # Create a composite market index (e.g., average of the indices)
        market_indices['market_index'] = market_indices.mean(axis=1)

        aggregated_data = pd.concat([market_indices, economic_indicators], axis=1).ffill()
        return aggregated_data
    

class MarketTrendComparator:
    def continuous_learning(self, new_market_data, new_stock_data, retrain_interval, date_index):
        """
        Continuously update the model with new data and retrain it at specified intervals.

        :param new_market_data: New market-wide data to be added for training.
        :param new_stock_data: New stock-specific data to be added for training.
        :param retrain_interval: Interval at which the model should be retrained.
        :param date_index: The current date index in the trading simulation.
        """
        if date_index % retrain_interval == 0:
            print(f"Retraining the model at date index: {date_index}")

            # Update historical data with new data
            self.update_data(new_market_data, new_stock_data)

            # Prepare combined data for retraining
            combined_data = self.combine_data()
            features, labels = self.prepare_data(combined_data)

            # Retrain the model with the updated data
            self.train_model(features, labels)

            print(f"Model retrained with updated data at index {date_index}.")
        else:
            print(f"No retraining at index {date_index}. Next retraining at index {date_index + (retrain_interval - (date_index % retrain_interval))}")

    def __init__(self, market_data, stock_data):
        self.market_data = market_data
        self.historical_data = stock_data
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def combine_data(self):
    # Check for duplicate dates in market_data
        if self.market_data.index.duplicated().any():
            print("Duplicate dates found in market_data:")
            print(self.market_data[self.market_data.index.duplicated(keep=False)])

    # Check for duplicate dates in stock_data
        if self.historical_data.index.duplicated().any():
            print("Duplicate dates found in stock_data:")
            print(self.historical_data[self.historical_data.index.duplicated(keep=False)])

        market_data_multi_index = self.market_data.copy()
        market_data_multi_index['Ticker'] = 'Market'
        market_data_multi_index.set_index('Ticker', append=True, inplace=True)

    # Verify that both DataFrames have unique indices
        assert market_data_multi_index.index.is_unique, "Market data index is not unique."
        assert self.historical_data.index.is_unique, "Stock data index is not unique."

    # Concatenate along the columns
        combined_data = pd.concat([market_data_multi_index, self.historical_data], axis=1)
        combined_data.index.names = ['Date', 'Ticker']  # Set index names
        return combined_data

    def calculate_features(self, combined_data):
        # Initialize a container for the new features
        feature_data = []

        # Market-Wide Indicators
        combined_data = combined_data.copy()
        combined_data.loc[:, 'market_change'] = combined_data['market_index'].pct_change(fill_method=None)

        # Iterate over each stock ticker in the index
        for ticker in combined_data.index.get_level_values(1).unique():
            # Access data for the current ticker
            ticker_data = combined_data.xs(ticker, level=1)

            # Stock-Specific Indicators
            price_change = ticker_data['Close'].pct_change(fill_method=None)
            volume_change = ticker_data['Volume'].pct_change(fill_method=None)
            ma_50 = ticker_data['Close'].rolling(window=50).mean()
            ma_200 = ticker_data['Close'].rolling(window=200).mean()

            # Calculate Stochastic Oscillator
            low_min = ticker_data['Low'].rolling(window=6).min()
            high_max = ticker_data['High'].rolling(window=6).max()
            k = 100 * ((ticker_data['Close'] - low_min) / (high_max - low_min))

            # Calculate MACD
            short_ema = ticker_data['Close'].ewm(span=12, adjust=False).mean()
            long_ema = ticker_data['Close'].ewm(span=26, adjust=False).mean()
            macd = short_ema - long_ema
            macd_signal = macd.ewm(span=9, adjust=False).mean()

            # Calculate RSI
            delta = ticker_data['Close'].diff(1)
            gain = (delta.where(delta > 0, 0)).fillna(0)
            loss = (-delta.where(delta < 0, 0)).fillna(0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # Calculate ATR (Average True Range)
            high_low = ticker_data['High'] - ticker_data['Low']
            high_close = (ticker_data['High'] - ticker_data['Close'].shift()).abs()
            low_close = (ticker_data['Low'] - ticker_data['Close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()

            # Calculate OBV (On-Balance Volume)
            obv = (ticker_data['Volume'] * (~ticker_data['Close'].diff().le(0) * 2 - 1)).cumsum()

            # Bollinger Bands Calculation
            sma_20 = ticker_data['Close'].rolling(window=20).mean()
            rolling_std_20 = ticker_data['Close'].rolling(window=20).std()
            upper_band = sma_20 + (2 * rolling_std_20)
            lower_band = sma_20 - (2 * rolling_std_20)

            # Combine the indicators into a DataFrame
            indicators = pd.DataFrame({
                f'{ticker}_price_change': price_change,
                f'{ticker}_volume_change': volume_change,
                f'{ticker}_50_day_MA': ma_50,
                f'{ticker}_200_day_MA': ma_200,
                f'{ticker}_Stochastic_Oscillator': k,
                f'{ticker}_MACD': macd,
                f'{ticker}_MACD_Signal': macd_signal,
                f'{ticker}_RSI': rsi,
                f'{ticker}_ATR': atr,
                f'{ticker}_OBV': obv,
                f'{ticker}_Upper_Band': upper_band,
                f'{ticker}_Lower_Band': lower_band
            })

            # Add ticker level to the indicators DataFrame
            indicators['Ticker'] = ticker
            indicators.set_index('Ticker', append=True, inplace=True)

            # Append the indicators to the feature_data list
            feature_data.append(indicators)

        # Combine all the feature data
        all_features = pd.concat(feature_data).sort_index()

        # Ensure index names are set correctly
        all_features.index.names = ['Date', 'Ticker']

        # Merge the new features back into the original combined_data
        combined_data = combined_data.merge(all_features, left_index=True, right_index=True, how='left')

        # Normalize/Standardize Features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(combined_data.fillna(0))
        scaled_combined_data = pd.DataFrame(scaled_features, index=combined_data.index, columns=combined_data.columns)

        return scaled_combined_data

    def define_labels(self, combined_data, threshold=0.02):
        """Define labels based on future price movement."""
        future_return = combined_data['Close'].shift(-1) / combined_data['Close'] - 1
        labels = pd.cut(future_return, bins=[-np.inf, -threshold, -threshold/2, threshold/2, threshold, np.inf], labels=['strong_sell', 'sell', 'neutral', 'buy', 'strong_buy'])
        return labels.fillna('neutral')

    def prepare_data(self, combined_data):
        """Prepare the dataset for training."""
        features = self.calculate_features(combined_data)
        labels = self.define_labels(combined_data)

        # Align features and labels
        features = features[1:-1]  # Adjust for lag
        labels = labels[1:-1]

        return features, labels

    def train_model(self, features, labels):
        """Train the model using the provided features and labels."""
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        print(f"Number of features used for training: {features.shape[1]}")  # Print feature count
        print(f"Model trained with accuracy: {self.model.score(X_test, y_test)}")

    def predict(self, new_data):
        """Make a prediction based on new combined data."""
        features = self.calculate_features(new_data)
        print(f"Number of features used for prediction: {features.shape[1]}")  # Print feature count
        return self.model.predict(features)

    def update_data(self, new_market_data, new_stock_data):
        """Update the data with new market and stock information."""
        self.market_data = pd.concat([self.market_data, new_market_data])
        self.historical_data = pd.concat([self.historical_data, new_stock_data])

    def save_model(self, filename):
        """Save the trained model to a file."""
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")

    def load_model(self, filename):
        """Load a trained model from a file."""
        self.model = joblib.load(filename)
        print(f"Model loaded from {filename}")


class BacktestTradingBot:
    def __init__(self, tickers, start_date, end_date, lookback_period=14):
        self.lookback_period = lookback_period
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers
        self.historical_data = self._fetch_historical_data(tickers, start_date, end_date)  # Stock-specific data
        self.market_data_collector = MarketDataCollector(start_date, end_date)  # Market-wide data collector
        self.market_data = self.market_data_collector.aggregate_data()  # Aggregate market-wide data
        self.market_trend_comparator = MarketTrendComparator(self.market_data, self.historical_data)  # Initialize the comparator

        self.portfolio = {'cash': 100000}
        self.initial_balance = 100000
        self.stop_loss_percentage = 0.05
        self.trailing_stop_loss_percentage = 0.05
        self.position_size_percentage = 0.05
        self.max_loss_percentage = 0.10
        self.trade_limit = 4  # Limit trades per week
        self.trade_count = 0  # Weekly trade counter
        self.cooldown_period = 0  # Days to wait after a trade before trading again
        self.last_trade_day = {}  # Dictionary to store the last trade day for each ticker
        self.logger = BotLogger()

        # Train the comparator with initial data
        combined_data = self.market_trend_comparator.combine_data()
        features, labels = self.market_trend_comparator.prepare_data(combined_data)
        self.market_trend_comparator.train_model(features, labels)

        for ticker in tickers:
            ticker_data_length = len(self.historical_data.xs(ticker, level='Ticker'))
            print(f"Length of data for {ticker}: {ticker_data_length}")

    def _fetch_historical_data(self, tickers, start_date, end_date):
        all_data = []
        for ticker in tickers:
            data = yf.download(ticker, start=start_date, end=end_date)
            data['Ticker'] = ticker  
            data_with_index = data.set_index('Ticker', append=True)
            all_data.append(data_with_index)

        # Combine all the data into a single DataFrame with multi-index (Date, Ticker)
        combined_stock_data = pd.concat(all_data)
        combined_stock_data = combined_stock_data.swaplevel(0, 1).sort_index()
        return combined_stock_data
    
    def _determine_market_condition(self, ticker):
        """
        Determine the market condition (bullish, bearish, neutral) based on indicators.
        """
        data = self.historical_data[ticker]
        short_ma = data['Close'].rolling(window=50).mean()  # Short-term moving average
        long_ma = data['Close'].rolling(window=200).mean()  # Long-term moving average
        macd, macd_signal = self._compute_macd(data['Close'])
        rsi = self._compute_rsi(data['Close'])

        # Check for bearish market condition
        if short_ma.iloc[-1] < long_ma.iloc[-1] and macd.iloc[-1] < macd_signal.iloc[-1] and rsi.iloc[-1] > 70:
            return 'bearish'
        # Check for bullish market condition
        elif short_ma.iloc[-1] > long_ma.iloc[-1] and macd.iloc[-1] > macd_signal.iloc[-1] and rsi.iloc[-1] < 30:
            return 'bullish'
        else:
            return 'neutral'

    def _init_optimizer(self):
        # Initialize MarketDataCollector to collect market-wide data
        market_data_collector = MarketDataCollector(self.start_date, self.end_date)
        market_data = market_data_collector.aggregate_data()

        # Initialize MarketTrendComparator with both market and stock data
        self.market_trend_comparator = MarketTrendComparator(market_data, self.historical_data)

        # Prepare combined data for initial training
        combined_data = self.market_trend_comparator.combine_data()
        features, labels = self.market_trend_comparator.prepare_data(combined_data)

        # Train the model with combined data
        self.market_trend_comparator.train_model(features, labels)
        print("Optimizer initialized and model trained with combined market and stock data.")
        
    def _analyze_stock(self, ticker, combined_data, date_index):
        print(f"Analyzing {ticker} for date index: {date_index}, corresponding to date: {combined_data.index[date_index]}")

        if date_index < 0 or date_index >= len(combined_data):
            print(f"Warning: date_index {date_index} is out of range for {ticker}.")
            return Decision.HOLD

        # Use the MarketTrendComparator to get a prediction
        prediction = self.market_trend_comparator.predict(combined_data.iloc[date_index:date_index + 1])

        if prediction == 'buy' or prediction == 'strong_buy':
            decision = Decision.BUY
        elif prediction == 'sell' or prediction == 'strong_sell':
            decision = Decision.SELL
        else:
            decision = Decision.HOLD

        print(f"Decision for {ticker} on index {date_index}: {decision}")   
        return decision

    def reset_trade_count(self, date):
        # Check if it's the start of the week, and reset trade_count
        if date.weekday() == 0:  # Monday
            self.trade_count = 0      
    
    def _can_trade(self, ticker, date_index):
        # Check if a trade can be executed based on the cooldown period
        last_trade_day = self.last_trade_day.get(ticker, None)
        if last_trade_day is not None:
            if date_index - last_trade_day < self.cooldown_period:
                if self.last_trade_day.get(f"{ticker}_stop_loss_triggered", False):   
                        return True 
                return False
        return True
    


    def _check_trailing_stop_loss(self, ticker, date_index):
        current_price = self.historical_data[ticker]['Close'].iloc[date_index]
    
    # Update the highest observed price
        highest_price_key = f"{ticker}_highest_price"
        if highest_price_key not in self.portfolio:
            self.portfolio[highest_price_key] = current_price
        else:
            self.portfolio[highest_price_key] = max(self.portfolio[highest_price_key], current_price)
        
        trailing_key = f"{ticker}_trailing_stop"
        if trailing_key in self.portfolio:
        # If current price drops below the trailing stop, sell the asset
            if current_price < self.portfolio[trailing_key]:
                self._place_order(ticker, "sell", self.portfolio.get(ticker, 0), date_index)
                self.last_trade_day[f"{ticker}_stop_loss_triggered"] = date_index  # Mark that stop loss was triggered for cooldown
        # Update the trailing stop based on the highest observed price
            else:
                updated_trailing_stop = self.portfolio[highest_price_key] - (self.trailing_stop_loss_percentage * self.portfolio[highest_price_key])
                self.portfolio[trailing_key] = max(self.portfolio[trailing_key], updated_trailing_stop)

    
    def _calculate_position_size(self, price):
    #Calculate the number of shares based on the position size percentage.
        amount_to_invest = self.portfolio['cash'] * self.position_size_percentage
        number_of_shares = amount_to_invest / price  # Use // for integer division
        return int(number_of_shares)

    def _place_order(self, ticker, action, quantity, price, date_index):
        print(f"Placing order: {action} {quantity} of {ticker} at {price} on index {date_index}")
        if self.trade_count >= self.trade_limit or not self._can_trade(ticker, date_index):
            print(f"Order for {ticker} not placed. Trade count or cooldown period restriction.")
            return

        if action == "buy":
            # Buying logic
            self._execute_buy(ticker, quantity, price, date_index)
        elif action == "sell":
            # Selling logic
            self._execute_sell(ticker, quantity, price, date_index)

    def _execute_buy(self, ticker, quantity, price, date_index):
        order_quantity = min(quantity, self._calculate_position_size(price))
        if order_quantity > 0:
            self.portfolio[ticker] = self.portfolio.get(ticker, 0) + order_quantity
            self.portfolio['cash'] -= price * order_quantity
            print(f"Executed buy: {order_quantity} of {ticker} at {price} on index {date_index}")
            # Setting up stop loss and trailing stop for the purchased stock
            self._check_stop_loss(ticker, date_index)
        else:
            print(f"Buy order not executed for {ticker}. Insufficient quantity or cash.")

    def _execute_sell(self, ticker, quantity, price, date_index):
        current_quantity = self.portfolio.get(ticker, 0)
        sell_quantity = min(quantity, current_quantity)
        if sell_quantity > 0:
            self.portfolio[ticker] -= sell_quantity
            self.portfolio['cash'] += price * sell_quantity
            print(f"Executed sell: {sell_quantity} of {ticker} at {price} on index {date_index}")
            # Removing the stop loss and trailing stop after selling
            self._check_stop_loss(ticker, date_index)
        else:
            print(f"Sell order not executed for {ticker}. Insufficient quantity.")

    def _check_max_loss(self, initial_balance, date_index):
        if self.portfolio['cash'] < initial_balance * (1 - self.max_loss_percentage):
            for ticker in self.historical_data:
                if ticker in self.portfolio and self.portfolio[ticker] > 0:
                    current_price = self.historical_data[ticker]['Close'].iloc[date_index]  # Get the current price
                    self._place_order(ticker, "sell", self.portfolio.get(ticker, 0), current_price, date_index)

    
    def _check_stop_loss(self, ticker, date_index):
        current_price = self.historical_data[ticker]['Close'].iloc[date_index]
        
        # Check for traditional stop-loss
        stop_loss_key = f"{ticker}_stop_loss"
        if stop_loss_key in self.portfolio:
            if current_price < self.portfolio[stop_loss_key]:
                self._place_order(ticker, "sell", self.portfolio.get(ticker, 0), date_index)
                self.last_trade_day[ticker] = date_index  # Updating the last trade day to start the cooldown
                return  # Exit the function after selling
        
        # Check for trailing stop-loss
        trailing_key = f"{ticker}_trailing_stop"
        if trailing_key in self.portfolio:
            # If current price drops below the trailing stop, sell the asset
            if current_price < self.portfolio[trailing_key]:
                self._place_order(ticker, "sell", self.portfolio.get(ticker, 0), date_index)
                self.last_trade_day[ticker] = date_index  # Updating the last trade day to start the cooldown
            # Update the trailing stop if the current price is higher than the previous trailing stop + a certain percentage
            elif current_price > self.portfolio[trailing_key] * (1 + self.trailing_stop_loss_percentage):
                self.portfolio[trailing_key] = current_price * (1 - self.trailing_stop_loss_percentage)
            
    def run(self, model_filename='model_filename.pkl'):
        market_data_collector = MarketDataCollector(self.start_date, self.end_date)
        market_data = market_data_collector.aggregate_data()

        self.market_trend_comparator = MarketTrendComparator(market_data, self.historical_data)
        self.market_trend_comparator.load_model(model_filename)

        total_days = len(self.historical_data[next(iter(self.historical_data))]) - self.lookback_period
        retrain_interval = 500  # Example interval for retraining the model

        for date_index in range(self.lookback_period, total_days):
            # Correctly extract the date from the multi-index (second level)
            date = self.historical_data[next(iter(self.historical_data))].index.get_level_values(1)[date_index]

            # Ensure date is a datetime object
            if not isinstance(date, pd.Timestamp):
                date = pd.to_datetime(date)

            self.reset_trade_count(date)

            # Continuously learn and update the model
            self.market_trend_comparator.continuous_learning(market_data, self.historical_data, retrain_interval, date_index)

            print(f"Processing date: {date}")

            # Check portfolio for max loss
            self._check_max_loss(self.initial_balance, date_index)

            # Iterate over each ticker and make trading decisions
            for ticker in self.historical_data:
                combined_data = self.market_trend_comparator.combine_data()
                decision = self._analyze_stock(ticker, combined_data, date_index)
                current_price = self.historical_data[ticker]['Close'].iloc[date_index]

                # Execute trades based on decisions
                if decision == Decision.BUY:
                    quantity = self._calculate_position_size(current_price)
                    self._place_order(ticker, "buy", quantity, current_price, date_index)
                elif decision == Decision.SELL:
                    quantity = self.portfolio.get(ticker, 0)
                    self._place_order(ticker, "sell", quantity, current_price, date_index)

                # Check for trailing stop loss
                self._check_trailing_stop_loss(ticker, date_index)

            # Calculate and display end-of-day portfolio value
            self.calculate_portfolio_value(date_index)

            print(f"Completed processing day {date_index - self.lookback_period + 1} of {total_days}")

        self.market_trend_comparator.save_model('updated_model.pkl')
        print(f"Model Accuracy after training: {self.market_trend_comparator.model_performance}")


        # Debug: Print progress
        print(f"Completed processing day {date_index - self.lookback_period + 1} of {total_days}")
        print(f"Model Accuracy after training: {self.optimizer.model_performance}")

# Rest of your code for setting up and running the bot

        
    def calculate_portfolio_value(self, date_index):
            last_day_index = len(self.historical_data[next(iter(self.historical_data))]) - 1
            closing_prices = {ticker: self.historical_data[ticker]['Close'].iloc[last_day_index] for ticker in self.historical_data}

            total_shares_value = sum(self.portfolio.get(ticker, 0) * closing_prices[ticker] for ticker in self.historical_data)
            total_portfolio_value = self.portfolio['cash'] + total_shares_value

            print(f"Closing Prices: {closing_prices}")
            print(f"Total Portfolio Value: {total_portfolio_value}")
            print(self.portfolio)
        

# Tickers and date range
tickers = ["TQQQ", "SQQQ"]
start_date = "2021-01-01"
end_date = "2022-01-01"
data_collector = MarketDataCollector(start_date, end_date)
economic_indicators = data_collector.fetch_economic_indicators()

print(economic_indicators.head())

bot = BacktestTradingBot(tickers, start_date, end_date)
bot.logger.output_log(print_log=True, save_to_file='trading_bot_log.txt')
bot.run(model_filename='updated_model.pkl')