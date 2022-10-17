from flask import Flask, request
from flask_restx import Api, Resource, fields, reqparse
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import json
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

# parser = reqparse.RequestParser()
# parser.add_argument("username", type=str)

app = Flask(__name__)
api = Api(app, title='API for ML models fit and predict')

models = {1: Ridge(), 2: RandomForestRegressor()}

@api.route("/models/<int:model_id>")
class Model(Resource):
    @api.doc(params={'model_id': {'description': 'id of the model', 'type': int, 'default': 1}})
    # @api.expect(parser)
    def get(self, model_id):
        """Return information about the model"""
        # model_type = request.args.get('type')
        # model_type = parser.parse_args()
        return {'model': str(models[model_id]), 'params': models[model_id].get_params()}

    def put(self, model_id):
        """Configure model with hyperparameters passed in JSON format"""
        params = request.get_json(force=True)
        models[model_id].set_params(**params)
        return {'model': str(models[model_id]), 'params': models[model_id].get_params()}

    def delete(self, model_id):
        """Delete exist model and create the new one"""
        if model_id == 1:
            models[model_id] = Ridge()
        elif model_id == 2:
            models[model_id] = RandomForestRegressor()
        else:
            raise ValueError("Incorrect model type specified")


@api.route("/models/<int:model_id>/fit")
class Fit(Resource):
    def put(self, model_id):
        """Fit model with previously passed train data"""
        # mode = request.form['mode']
        X, y = dataset['train'].X, dataset['train'].y
        models[model_id].fit(X, y)
        rmse = mean_squared_error(y, models[model_id].predict(X)) ** 0.5
        return {"status": "Fitted on train data", "RMSE on train": round(rmse, 4)}


@api.route("/models/<int:model_id>/validate")
class Validate(Resource):
    def get(self, model_id):
        """Perform 5-fold cross-validation on previously passed train data"""
        X, y = dataset['train'].X, dataset['train'].y
        cv_score = -cross_val_score(models[model_id], X, y, cv=5, scoring='neg_root_mean_squared_error').mean()
        return {"status": "Fitted on train data", "RMSE on 5-fold CV": round(cv_score, 4)}


# @api.route("/models/<int:model_id>/predict")
# class Predict(Resource):
#     def get(self, model_id):
#         data = request.form['data']


class Dataset:
    def __init__(self, target_column: str):
        self.target_column = target_column
        self.df = None
        self.X = None
        self.y = None

    def read_data(self, raw_data):
        self.df = pd.read_csv(raw_data)
        self.X = self.df.drop(columns=[self.target_column])
        self.y = self.df[self.target_column]
        print(f'Shape of the data is {self.df.shape}')

    def get_data(self):
        return self.df.iloc[0].to_dict()

dataset = {}

@api.route("/data/<dataset_name>")
class Data(Resource):
    @api.doc(params={'filename': {'description': 'Path to file with your data in csv format',
                                  'type': 'str', 'required': True},
                     'target_name': {'description': 'Name of the column containing target',
                                     'type': 'str', 'required': True}},
             responses={200: 'Data loaded'})

    def put(self, dataset_name):
        """Load and save your csv dataset"""
        filename = request.form['filename']
        target_name = request.form['target_name']
        data = Dataset(target_name)
        data.read_data(filename)
        dataset[dataset_name] = data
        return {"status": "Dataset successfully loaded", "shape": str(data.df.shape)}, 200

    def get(self, dataset_name):
        """Return the first row of the data in json format"""
        return dataset[dataset_name].get_data()


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)