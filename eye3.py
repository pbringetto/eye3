from flask import Flask, request, render_template
import drop as d

app = Flask(__name__)

@app.route("/")
def hello():
    drop = d.Drop()
    data = drop.alpha()
    return render_template('index.html', data=data)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
