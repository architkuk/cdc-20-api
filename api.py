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
    model_data.append(data['income'])
    model_data.append(data['loanAmt'])
    model_data.append(data['propertyVal'])
    model_data.append(data['type'])
    model_data.append(data['occupiedUnits'])
    model_data.append(data['purpose'])
    model_data.append(data['businessOrCommercial'])
    
    # predictions
    result = model.predict([model_data])

    # send back to browser
    output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
