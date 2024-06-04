from flask import Flask, jsonify, request
import random
import string

# insert flask annotation here
app = Flask(__name__)

@app.route('/get')
def _get():
    # s: number of bytes to generate
    s = request.args.get('s', default=100, type=int)
    num_chars = int(s)
    response_body = ''.join(random.choices(string.ascii_letters,k=num_chars))
    return response_body

if __name__ == "__main__":
    app.run(host='0.0.0.0')