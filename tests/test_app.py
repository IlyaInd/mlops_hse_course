import json

import numpy as np
import sklearn.linear_model
from dataclasses import dataclass
from core.models import ModelList, PREDEFINED_MODELS, app, Fit, Predict

# @pytest.fixture
# def client():
#     with app.test_client() as client:
#         yield client

def test_initial_models():
    assert len(PREDEFINED_MODELS) == 2
    assert isinstance(PREDEFINED_MODELS[0], sklearn.linear_model.Ridge)
    assert isinstance(PREDEFINED_MODELS[1], sklearn.ensemble.RandomForestRegressor)


@dataclass
class DatabaseTableRow:
    model_id: int
    model_name: str


def test_mock_database(mocker):
    mocker.patch('core.models.db')
    mocker.patch('core.models.session.query', return_value=[DatabaseTableRow(0, 'Ridge'),
                                                            DatabaseTableRow(1, 'RandomForest')])
    model_list = ModelList().get()
    print(model_list)
    assert isinstance(model_list, list)
    assert isinstance(model_list[0], dict)
    assert list(model_list[0].keys()) == ['model_id', 'model']


def test_types_of_models(mocker):
    mocker.patch('core.models.session')
    model_list = ModelList()
    # mocker.patch('flask.request.get_json', return_value={'model_type': 'ridge'})
    # app.testing = True

    with app.test_request_context('/model', json={'model_type': 'ridge'}):
        response = model_list.post()
        print(response)
    assert response[1] == 200

    with app.test_request_context('/model', json={'model_type': 'gachi_regression'}):
        response = model_list.post()
        print(response)
    assert response[1] == 400


def test_ridge_fit(mocker):
    model = sklearn.base.clone(PREDEFINED_MODELS[0])
    mocker.patch('pickle.loads', return_value=model)
    mocker.patch('core.models.session')

    train_data = [[1], [2], [5]]
    target = [1, 3, 8]
    fit_model = Fit()
    with app.test_request_context('/model/1/fit', json={"train_data": train_data, "target": target}):
        fit_model.put(model_id=1)
    assert round(model.coef_[0], 6) == 1.551724
    assert round(model.intercept_, 6) == -0.137931


def test_ridge_predict(mocker):
    model = sklearn.base.clone(PREDEFINED_MODELS[0])
    mocker.patch('pickle.loads', return_value=model)
    mocker.patch('core.models.session')

    data = [[1], [2], [5]]
    target = [1, 3, 8]
    predict_model = Predict()
    with app.test_request_context('/model/1/predict', json={"data": data, "target": target}):
        response = predict_model.put(model_id=1)
        assert response[1] == 424  # Not fitted yet

    model.fit(data, target)
    with app.test_request_context('/model/1/predict', json={"data": data, "target": target}):
        response = predict_model.put(model_id=1)
        assert response[1] == 200
        assert np.allclose(response[0]['y_pred'], [[1.413793, 2.965517, 7.620689]])

    incorrect_data = json.dumps({0: [1], 1: [2], 2: [5]})
    print(incorrect_data)
    with app.test_request_context('/model/1/predict', json={"data": incorrect_data, "target": target}):
        response = predict_model.put(model_id=1)
        print(response)
        assert response[1] == 400  # wrong data format
