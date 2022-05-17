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
import models.signal_model as sm
signal_data = sm.SignalDataModel()

def signal_keys():
   return ['at_bollinger_low', 'below_bollinger_low', 'rsi_oversold', 'bullish_regular', 'bullish_hidden', 'macd_over_signal', 'macd_over_centerline', 'macd_rising'], ['above_bollinger_high', 'at_bollinger_high', 'rsi_overbought', 'bearish_regular', 'bearish_regular', 'macd_under_signal', 'macd_under_centerline', 'macd_dropping']

def get_signals(pair, tf, data, df):
    signals = signal_data.get_signals(pair, tf)
    #print('get_signals from db')
    #print(signals)
    #active_signals = []
    #if not signals:
        #print('no signals')
        #for key, value in data.items():
            #if value:
                #active_signals.append({key : value})
                #signal_data.insert_signal(pair, tf, key, value, df['startTime'].iloc[-1])
                #signals = signal_data.get_signals(pair, tf)
        #print('first insert')
        #print(active_signals)
    return signals

def split_signals(signals, data, pair, tf, df):
    buy_signal_keys, sell_signal_keys = signal_keys()
    buy_signals, sell_signals = [], []
    update = False
    active_signals = []
    for key, value in data.items():
        if value:
            active_signals.append({key : value})
            if key in buy_signal_keys:
                buy_signals.append({key : value})
            if key in sell_signal_keys:
                sell_signals.append({key : value})

    #print('other insert')
    #print(active_signals)

    if structure_change(signals, data):
        update = True
        #print(update)
        #signal_data.insert_signal(pair, tf, key, value, df['startTime'].iloc[-1])
    return buy_signals, sell_signals, update

def signals(data, pair, tf, df):
    signals = get_signals(pair, tf, data, df)
    return split_signals(signals, data, pair, tf, df)

def structure_change(signals, data):
    print('8888888888888888888888888888888888')
    print(data)
    print(signals)
    previous_signals = list(filter(lambda x: x['created_at'] == signals[0]['created_at'], signals))
    previous_signals_dict = {}
    for value in previous_signals:
        previous_signals_dict[value['key']] = value['value']
    current_signals = {}
    for key, value in data.items():
        if value:
            current_signals[key] = value
    shared_items = {k: previous_signals_dict[k] for k in previous_signals_dict if k in current_signals and previous_signals_dict[k] == current_signals[k]}

    print('00000000000000000000000000000000000000')

    print(len(previous_signals_dict))
    print(len(shared_items))

    return (len(previous_signals_dict) != len(shared_items)) or len(signals) == 0

def go():
    strategy = s.Strategy()

    ftx = f.FtxClient(alpha["ftx_key"], alpha["ftx_secret"])
    tf = 86400
    for pair in alpha["pairs"]:
        print(pair['pair'])
        for tf in alpha["timeframes"]:

            print(tf['seconds'])
            df = None
            data = None
            df = pd.DataFrame(ftx.get_historical_prices(pair['pair'], tf['seconds']))
            data, df = strategy.setup(df, tf['seconds'], pair['pair'])

            #print(data)


            buy_signals, sell_signals, update = signals(data, pair['pair'], tf['seconds'], df)


            if update:
                for item in buy_signals + sell_signals:
                    for key, value in item.items():
                        #print(value)
                        signal_data.insert_signal(pair['pair'], tf['seconds'], key, value, df['startTime'].iloc[-1])


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


