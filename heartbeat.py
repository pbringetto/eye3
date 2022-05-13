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
   return ['rsi_oversold', 'bullish_regular', 'bullish_hidden', 'macd_over_signal', 'macd_over_centerline', 'macd_rising'], ['rsi_overbought', 'bearish_regular', 'bearish_regular', 'macd_under_signal', 'macd_under_centerline', 'macd_dropping']

def signals(data, pair, tf):
    buy_signal_keys, sell_signal_keys = signal_keys()
    buy_signals, sell_signals = [], []

    signals = signal_data.get_signals(pair, tf)
    if signals:
        previous_signals = list(filter(lambda x: x['created_at'] == signals[0]['created_at'], signals))
        previous_signals_dict = {}
        for value in previous_signals:
            previous_signals_dict[value['key']] = value['value']
        current_signals = {}
        for key, value in data.items():
            if value:
                current_signals[key] = value
        shared_items = {k: previous_signals_dict[k] for k in previous_signals_dict if k in current_signals and previous_signals_dict[k] == current_signals[k]}

        for key, value in data.items():
            if value:
                if key in buy_signal_keys:
                    buy_signals.append({key : value})
                if key in sell_signal_keys:
                    sell_signals.append({key : value})
                print(len(previous_signals_dict))
                print(len(shared_items))
                if len(previous_signals_dict) != len(shared_items):
                    signal_data.insert_signal(pair, tf, key, value)
                    print('update')
                else:
                    print('no change')
    return buy_signals, sell_signals

def go():
    strategy = s.Strategy()
    twitter = t.Twitter()
    ftx = f.FtxClient(alpha["ftx_key"], alpha["ftx_secret"])
    tf = 86400
    for pair in alpha["pairs"]:
        for tf in alpha["timeframes"]:
             df = pd.DataFrame(ftx.get_historical_prices(pair['pair'], tf['seconds']))
             data, df = strategy.setup(df, tf['seconds'], pair['pair'])

             u.show('Market', pair['label'])
             u.show('Timeframe', tf['label'])
             u.show('Price', df['close'].iloc[-1])
             u.show('Open Price', df['open'].iloc[-1])
             u.show('High', df['high'].iloc[-1])
             u.show('Low', df['low'].iloc[-1])

             buy_signals, sell_signals = signals(data, pair['pair'], tf['seconds'])

             print('bullish case------------------------------------')
             for value in buy_signals:
                 print(value)

             print('bearish case------------------------------------')
             for value in sell_signals:
                 print(value)



             '''
             #get signal table row for tf pair since n
             if data['rsi_oversold']:
                 #if tf pair signal table row not exist since n
                 #create signal table row

             if data['rsi_oversold']:

             rsi oversold
             bullish divergence
             macd crosses
             buy

             if data['bullish_regular'] or data['bullish_regular']:
                 #if tf pair signal table row exist since n
                 #update signal table row

             if data['bullish_regular'] or data['bullish_regular']:


            'macd_over_signal': 'MACD Crossover soon' if ohlc['MACD_12_26_9'].iloc[-1] > ohlc['MACDs_12_26_9'].iloc[-1] else False,
            'macd_over_centerline':

             rsi overbought
             bearish divergence
             macd crosses
             sell
             '''

             twitter.tweet()

go()


