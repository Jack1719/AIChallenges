from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/audio-data', methods=['GET'])
def get_audio_data():
    # Example of sending JSON data
    data = {
        "message": "Hello from Flask!",
        "data": [1, 2, 3, 4]
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
