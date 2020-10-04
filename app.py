from flask import Flask, jsonify, request
import lightgbm as lgb

# load model
model = lgb.Booster(model_file='model.txt')

# app
app = Flask(__name__)

# routes


@app.route('/', methods=['POST'])
def predict():
    # get data
    data = request.get_json(force=True)

   # create list from data
    model_data = []
    model_data.append(float(data['income']))
    model_data.append(float(data['loanAmt']))
    model_data.append(float(data['propertyVal']))
    model_data.append(int(data['type']))
    model_data.append(int(data['occupiedUnits']))
    model_data.append(int(data['purpose']))
    model_data.append(int(data['businessOrCommercial']))

    # predictions
    result = model.predict([model_data])

    # send back to browser
    output = {'results': result[0]}

    # return data
    # return jsonify(results=output)
    return jsonify(results=output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)
