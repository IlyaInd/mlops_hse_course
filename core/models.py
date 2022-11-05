from flask import Flask
from flask_restx import Api, Resource, fields
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import NotFittedError

app = Flask(__name__)
api = Api(app, title='API for ML core fit and predict')

models = [Ridge(), RandomForestRegressor()]

model_fields = api.model('Model', {'model_type': fields.String(enum=['ridge', 'random_forest'], validate=True)})
@api.route("/model")
class ModelList(Resource):
    def get(self):
        """Return list of all current core"""
        return [{"model_id": model_id, "model": str(model)} for model_id, model in enumerate(models)]

    @api.doc(responses={200: "Add model successfully", 400: "Incorrect type of model specified"})
    @api.expect(model_fields)
    def post(self):
        """Add new untrained model"""
        model_type = api.payload['model_type']
        if model_type == 'ridge':
            models.append(Ridge())
        elif model_type == 'random_forest':
            models.append(RandomForestRegressor())
        else:
            return 'Incorrect type of model, please choose one of these types: [ridge, random_forest]', 400
        return f'Added new model with model_id={len(models)-1}', 200


params_fields = api.model('Hyperparameters', {'params': fields.Raw(default={"alpha": 1.0},
                                                                   description="Hyperparameters of the model in json format")})

@api.route("/model/<int:model_id>")
class Model(Resource):
    @api.doc(params={'model_id': {'description': 'id of the model', 'type': int, 'default': 1}},
             responses={200: "Find and show model", 404: "Incorrect model_id specified"})
    def get(self, model_id):
        """Return information about the model"""
        try:
            return {'model': str(models[model_id]), 'params': models[model_id].get_params()}, 200
        except IndexError:
            return f"Incorrect model_id specified, must be in range [0, {len(models)-1}]", 404

    @api.expect(params_fields)
    def put(self, model_id):
        """Configure model with hyperparameters passed in JSON format"""
        params = api.payload['params']
        models[model_id].set_params(**params)
        return {'model': str(models[model_id]), 'params': models[model_id].get_params()}

    @api.doc(responses={200: "Delete model successfully", 404: "Model with specified model_id not found"})
    def delete(self, model_id):
        """Drop the model"""
        if model_id <= len(models) - 1:
            m = models[model_id].__repr__()
            del models[model_id]
            return f"Model {m} was successfully deleted", 200
        else:
            return f"Incorrect model_id specified, must be in range [0, {len(models)-1}]", 404


fit_fields = api.model('Train data', {'train_data': fields.List(fields.List(fields.Float()), default=[[1], [2], [5]]),
                                      'target': fields.List(fields.Float(), default=[1, 3, 8])})
@api.route("/model/<int:model_id>/fit")
class Fit(Resource):
    @api.expect(fit_fields)
    def put(self, model_id):
        """Fit model with data passed in JSON format"""
        train_data, target = np.array(api.payload["train_data"]), np.array(api.payload["target"])
        models[model_id].fit(train_data, target)
        rmse = mean_squared_error(target, models[model_id].predict(train_data)) ** 0.5
        return {"status": "Fitted on train data", "RMSE on train": round(rmse, 4)}


predict_fields = api.model('Predict data', {'data': fields.List(fields.List(fields.Float()), default=[[1], [2], [5]])})
@api.route("/model/<int:model_id>/predict")
class Predict(Resource):
    @api.doc(responses={200: "Success inference", 400: "Incorrect data format", 424: "Model not fitted yet"})
    @api.expect(predict_fields)
    def put(self, model_id):
        """Return prediction by data passed in JSON format"""
        data = np.array(api.payload["data"])
        try:
            y_pred = models[model_id].predict(data)
            return {"y_pred": list(y_pred)}
        except NotFittedError as e:
            return repr(e), 424
        except:
            return "Incorrect data format, must be list of list with floats or integers", 400
