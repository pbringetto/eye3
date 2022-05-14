import api.ftx as f
import strategy as s
import cfg_load
import twitter as t
alpha = cfg_load.load('alpha.yaml')
import pandas as pd
import helpers.util as u
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
import models.signal_model as sig
signal_data = sig.SignalDataModel()

def signal_keys():
   return ['at_bollinger_low', 'below_bollinger_low', 'rsi_oversold', 'bullish_regular', 'bullish_hidden', 'macd_over_signal', 'macd_over_centerline', 'macd_rising'], ['above_bollinger_high', 'at_bollinger_high', 'rsi_overbought', 'bearish_regular', 'bearish_regular', 'macd_under_signal', 'macd_under_centerline', 'macd_dropping']

def get_signals(pair, tf, data, df):
    signals = signal_data.get_signals(pair, tf)
    if not signals:
        for key, value in data.items():
            if value:
                signal_data.insert_signal(pair, tf, key, value, df['startTime'].iloc[-1])
                signals = signal_data.get_signals(pair, tf)
    return signals

def split_signals(signals, data, pair, tf, df):
    buy_signal_keys, sell_signal_keys = signal_keys()
    buy_signals, sell_signals = [], []
    update = False
    for key, value in data.items():
        if value:
            if key in buy_signal_keys:
                buy_signals.append({key : value})
            if key in sell_signal_keys:
                sell_signals.append({key : value})
            if structure_change(signals, data):
                signal_data.insert_signal(pair, tf, key, value, df['startTime'].iloc[-1])
                update = True
    return buy_signals, sell_signals, update

def signals(data, pair, tf, df):
    signals = get_signals(pair, tf, data, df)
    return split_signals(signals, data, pair, tf, df)

def structure_change(signals, data):
    previous_signals = list(filter(lambda x: x['created_at'] == signals[0]['created_at'], signals))
    previous_signals_dict = {}
    for value in previous_signals:
        previous_signals_dict[value['key']] = value['value']
    current_signals = {}
    for key, value in data.items():
        if value:
            current_signals[key] = value
    shared_items = {k: previous_signals_dict[k] for k in previous_signals_dict if k in current_signals and previous_signals_dict[k] == current_signals[k]}
    return len(previous_signals_dict) != len(shared_items)

def go():
    strategy = s.Strategy()

    ftx = f.FtxClient(alpha["ftx_key"], alpha["ftx_secret"])
    tf = 86400
    for pair in alpha["pairs"]:
        for tf in alpha["timeframes"]:
            df = pd.DataFrame(ftx.get_historical_prices(pair['pair'], tf['seconds']))
            data, df = strategy.setup(df, tf['seconds'], pair['pair'])
            buy_signals, sell_signals, update = signals(data, pair['pair'], tf['seconds'], df)

            if update:
                tweet(buy_signals, sell_signals)

            u.show('Market', pair['label'])
            u.show('Timeframe', tf['label'])
            u.show('Price', df['close'].iloc[-1])
            u.show('Open Price', df['open'].iloc[-1])
            u.show('High', df['high'].iloc[-1])
            u.show('Low', df['low'].iloc[-1])

            print('bullish case------------------------------------')
            for value in buy_signals:
                print(value)

            print('bearish case------------------------------------')
            for value in sell_signals:
                print(value)



def tweet(buy_signals, sell_signals):
    twitter = t.Twitter()
    twitter.tweet('hello')

go()


