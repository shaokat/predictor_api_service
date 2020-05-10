from app import app
from flask import jsonify,request
import pickle
import numpy as np


@app.route("/api/prediction/thunder", methods=['POST'])
def get_predictaion():
    feature_list_pr = []
    feature_order = ['dew_temp', 'dry_temp', 'relative_humidity', 'pressure']
    feature_info = request.get_json() or {}
    #feature_info = json.load(data)
    if 'model' not in feature_info:
        model_selected = "random_forest.mod"
    else:
        model_name = feature_info["model"]
        if model_name == 'rf':
            model_selected = "random_forest.mod"
        else:
            model_selected = "neural_network.mod"
    time_info = feature_info['time']
    feature_list = feature_info['features']

    feature_list_pr.append(time_info['month'])
    feature_list_pr.append(time_info['time'])
    feature_list_pr.append(time_info['date'])

    for index, features in enumerate(feature_list):
        values = features[feature_order[index]]
        #    print(index,values)

        last_value = values[1]
        rof = (values[0] - values[1]) / 3
        #    print(last_value,rof)
        feature_list_pr.append(last_value)
        feature_list_pr.append(rof)

    feature_for_pred = np.array(feature_list_pr).reshape(1, -1)
    from sklearn.preprocessing import StandardScaler
    model = pickle.load(open(model_selected, 'rb'))
    features_scaled = StandardScaler().fit_transform(feature_for_pred)
    prediction = model.predict(features_scaled.reshape(1, -1))[0]

    start_time = time_info['time']
    end_time = (time_info['time'] + 3) % 24
    results = {
        'start_time': start_time,
        'end_time': end_time,
        'prediction': int(prediction)

    }

    return jsonify(results)
