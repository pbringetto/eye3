from flask import Flask, request, render_template
import drop as d

app = Flask(__name__)

@app.route("/")
def hello():
    drop = d.Drop()
    data = drop.signals()
    #print(buy_signals)
    #return render_template('index.html', buy_signals=buy_signals, sell_signals=sell_signals )
    return 'fhjgfjgjh'

if __name__ == "__main__":
    app.run(host='0.0.0.0')
