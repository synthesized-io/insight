from io import StringIO

import pandas as pd
import werkzeug
from flask import Flask, Response
from flask_restful import Resource, Api, reqparse

from synthesized.core import BasicSynthesizer

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

api = Api(app)


class Synthesize(Resource):
    def post(self):
        parse = reqparse.RequestParser()
        parse.add_argument('file', type=werkzeug.FileStorage, location='files', required=True)
        args = parse.parse_args()
        file = args['file']
        with file.stream as stream:
            data = pd.read_csv(stream).dropna()
            print(data.head(5))
            with BasicSynthesizer(data=data) as synthesizer:
                synthesizer.learn(num_iterations=2000, data=data, verbose=200)
                synthesized = synthesizer.synthesize(len(data))
                out = StringIO()
                synthesized.to_csv(out)
                out.seek(0)
                return Response(response=out, status=200, mimetype='text/csv')


api.add_resource(Synthesize, '/synthesize')

if __name__ == '__main__':
    app.run(port='5002')
