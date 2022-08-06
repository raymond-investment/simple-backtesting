
import pandas as pd
import numpy as np
import yfinance

def get_price_history(symbol: str) -> pd.DataFrame:
    """
    Using Yahoo Finance third party package to fetch historical candlestick data which includes Open, High, Low, Close,
    Volume, Dividend, Split. Default setting of period is "Max", we want to fetch the data as long as we can.
    :param symbol:
    :return: pd.DataFrame
    """
    price_history_df = yfinance.Ticker(symbol).history(period="max")
    return price_history_df

def backtest(symbol: str, parameters_1: int, parameters_2: int, commisson = 0) -> dict:
    # Get historical candlestick data from Yahoo Finance
    price_history_df = get_price_history(symbol=symbol)

    # Calculate daily return on Close price
    price_history_df['daily return'] = price_history_df['Close'].pct_change()

    # Calculate technical indicators
    price_history_df['Indicator 1'] = price_history_df['Close'].rolling(parameters_1).mean()
    price_history_df['Indicator 2'] = price_history_df['Close'].rolling(parameters_2).mean()

    # Calculate trading signal
    price_history_df['signal'] = np.where(price_history_df['Indicator 1'] > price_history_df['Indicator 2'], 1, 0)

    # Backtest
    """
    Assuming execute single trade at next bar close.
    Consider continuously compound in cumulative return, it means invest all of the position in the all time. 
    Portfolio investment: continuously compound
    Futures trading: simple interest .cumsum()
    """
    price_history_df['strategic return'] = price_history_df['daily return'] * price_history_df['signal'].shift(1)
    price_history_df['equity curve'] = price_history_df['strategic return'].cumsum().apply(np.exp)

    trade_list = price_history_df[price_history_df['signal'] != price_history_df['signal'].shift(1)][['signal','Close']]

    if trade_list['signal'].iloc[0] == 0:
        trade_list = trade_list[1:]

    if trade_list['signal'].iloc[-1] == 1:
        trade_list = trade_list[:-1]

    buy_trade_list = trade_list[trade_list['signal'] == 1].reset_index()
    sell_trade_list = trade_list[trade_list['signal'] == 0].reset_index()

    all_trades = pd.concat([buy_trade_list, sell_trade_list], axis = 1)
    all_trades.columns = ['EntryDate','signal','EntryPrice','ExitDate','signal','ExitPrice']

    all_trades['profit'] = all_trades['ExitPrice'] / all_trades['EntryPrice'] - 1 - commisson

    trades_num = len(all_trades)
    win_percent = np.where(all_trades['profit'] > 0, 1, 0).mean()
    profit_factor = -np.where(all_trades['profit'] > 0, all_trades['profit'], 0).sum() / np.where(all_trades['profit'] < 0, all_trades['profit'], 0).sum()
    annual_return = (1 + price_history_df['strategic return'].mean() ** 252) -1
    volatility = price_history_df['strategic return'].std() * np.sqrt(252)
    sharpe_ratio = annual_return / volatility

    return_dict = {'strategic daily return': price_history_df['strategic return'],
                   'all trades': all_trades,
                   'performance': {'WinPercent': win_percent,
                                   'Profit Factor': profit_factor,
                                   'Number of traders': trades_num,
                                   'Annual Return': annual_return,
                                   'Volatility': volatility,
                                   'Sharpe Ratio': sharpe_ratio}}

    return return_dict



if __name__ == '__main__':
    results = backtest('SPY', 5, 34)
    print(results)



