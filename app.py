import pandas as pd
from flask import Flask, jsonify, request
import pickle
from flask_cors import CORS

# load model
model = pickle.load(open('model.pkl','rb'))

# app
app = Flask(__name__)
CORS(app)
# routes
@app.route('/', methods=['POST'])

def predict():
        # get data
        data = request.get_json(force=True)

        # convert data into dataframe
        data_df = pd.DataFrame(data)

        # predictions
        result = model.predict(data_df)

        output = {'results': int(result[0])}

        # return data
        #return jsonify({'prediction': str(result)})

        return jsonify(output)

if __name__ == '__main__':
        print('model loaded')
        app.run(port = 5000, debug=True)
