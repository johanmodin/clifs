from flask import Flask

from clifs import CLIFS
from flask import jsonify

app = Flask(__name__)

clifs = CLIFS()

# This is not fully implemented yet
@app.route('/add_video/<path>')
def add_video():
    return "Hello World!"


@app.route('/search/<query>')
def search(query):
    query_results = clifs.search(query)
    return jsonify(query_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
