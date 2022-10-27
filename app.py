from flask import Flask
from flask_restx import Api, Resource, fields
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import NotFittedError


app = Flask(__name__)
api = Api(app, title='API for ML models fit and predict')

models = {1: Ridge(), 2: RandomForestRegressor()}
dataset = {}


@api.route("/models")
class ModelList(Resource):
    def get(self):
        """Return all available models"""
        return [{"model_id": model_id, "model": str(model)} for model_id, model in models.items()]


params_fields = api.model('Hyperparameters', {'params': fields.Raw(default={"alpha": 1.0},
                                                                   description="Hyperparameters of the model in json format")})
@api.route("/models/<int:model_id>")
class Model(Resource):
    @api.doc(params={'model_id': {'description': 'id of the model', 'type': int, 'default': 1}})
    def get(self, model_id):
        """Return information about the model"""
        return {'model': str(models[model_id]), 'params': models[model_id].get_params()}

    @api.expect(params_fields)
    def put(self, model_id):
        """Configure model with hyperparameters passed in JSON format"""
        params = api.payload['params']
        models[model_id].set_params(**params)
        return {'model': str(models[model_id]), 'params': models[model_id].get_params()}

    def delete(self, model_id):
        """Delete exist model and create the new one, ie reset it"""
        if model_id == 1:
            models[model_id] = Ridge()
            return "Model successfully reset", 200
        elif model_id == 2:
            models[model_id] = RandomForestRegressor()
            return "Model successfully reset", 200
        else:
            return "Incorrect model type specified", 404


fit_fields = api.model('Train data', {'X': fields.List(fields.Float(), default=[[1], [2], [5]]),
                                'y': fields.List(fields.Float(), default=[1, 3, 8])})
@api.route("/models/<int:model_id>/fit")
class Fit(Resource):
    @api.expect(fit_fields)
    def put(self, model_id):
        """Fit model with data passed in JSON format"""
        X, y = np.array(api.payload["X"]), np.array(api.payload["y"])
        models[model_id].fit(X, y)
        rmse = mean_squared_error(y, models[model_id].predict(X)) ** 0.5
        return {"status": "Fitted on train data", "RMSE on train": round(rmse, 4)}


predict_fields = api.model('Predict data', {'X': fields.List(fields.Float(), default=[[1], [2], [5]])})
@api.route("/models/<int:model_id>/predict")
class Predict(Resource):
    @api.expect(predict_fields)
    def put(self, model_id):
        """Return prediction by data passed in JSON format"""
        X = np.array(api.payload["X"])
        try:
            y_pred = models[model_id].predict(X)
            return {"y_pred": list(y_pred)}
        except NotFittedError as e:
            return repr(e)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
