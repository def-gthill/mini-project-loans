import flask
from flask import request
import flask_restful as fr
from flask_restful import reqparse

import numpy as np
import pandas as pd

from bobs import learn


app = flask.Flask(__name__)
api = fr.Api(app)


model = learn.load('everything_logistic.pkl')
must_have_fields = [
    'ApplicantIncome', 'LoanAmount', 'Credit_History',
]


class LoanApplication(fr.Resource):
    def post(self):
        json_data = request.get_json()
        for field in must_have_fields:
            if json_data[field] is None:
                return {'status': 'incomplete'}
        df = pd.DataFrame(
            json_data.values(), index=json_data.keys()
        ).transpose()
        prediction = model.predict(df)[0]
        result = 'approved' if prediction == 'Y' else 'denied'
        return {'status': result}


api.add_resource(LoanApplication, '/loan')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)