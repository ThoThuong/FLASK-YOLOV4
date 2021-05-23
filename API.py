from flask_cors import CORS
import json
from flask import Flask, request, Response
from absl.flags import FLAGS
from absl import app, flags, logging
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/health', methods=['GET'])
def health():
    return Response(response='api ok')


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)
