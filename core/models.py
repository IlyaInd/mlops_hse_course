from flask import Flask
from flask_restx import Api, Resource, fields
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import NotFittedError
from sqlalchemy import func
import db
import pickle

app = Flask(__name__)
api = Api(app, title='API for ML core fit and predict')
PREDEFINED_MODELS = [Ridge(), RandomForestRegressor()]

db.engine.connect()
session = db.Session()

def init_models_in_db():
    """Create and write in database default versions of each available models"""
    for i, model in enumerate(PREDEFINED_MODELS):
        m = db.Models(i, str(model), pickle.dumps(model))
        session.add(m)
    session.commit()

model_fields = api.model('Model', {'model_type': fields.String(enum=['ridge', 'random_forest'], validate=True)})
@api.route("/model")
class ModelList(Resource):
    def get(self):
        """Return list of all current core"""
        return [{"model_id": row.model_id, "model": row.model_name} for row in session.query(db.Models)]

    @api.doc(responses={200: "Add model successfully", 400: "Incorrect type of model specified"})
    @api.expect(model_fields)
    def post(self):
        """Add new untrained model"""
        model_type = api.payload['model_type']
        max_id = session.query(func.max(db.Models.model_id)).scalar()
        if model_type == 'ridge':
            model = Ridge()
        elif model_type == 'random_forest':
            model = RandomForestRegressor()
        else:
            return 'Incorrect type of model, please choose one of these types: [ridge, random_forest]', 400
        session.add(db.Models(max_id + 1, str(model), pickle.dumps(model)))
        session.commit()
        return f'Added new model with model_id={max_id + 1}', 200


params_fields = api.model('Hyperparameters', {'params': fields.Raw(default={"alpha": 1.0},
                                                                   description="Hyperparameters of the model in json format")})

@api.route("/model/<int:model_id>")
class Model(Resource):
    @api.doc(params={'model_id': {'description': 'id of the model', 'type': int, 'default': 1}},
             responses={200: "Find and show model", 404: "Incorrect model_id specified"})
    def get(self, model_id):
        """Return information about the model"""
        max_id = session.query(func.max(db.Models.model_id)).scalar()
        try:
            row = session.query(db.Models).get(model_id)
            model = pickle.loads(row.model_binary)
            return {'model': row.model_name, 'params': model.get_params()}, 200
        except IndexError:
            return f"Incorrect model_id specified, must be in range [0, {max_id}]", 404

    @api.expect(params_fields)
    def put(self, model_id):
        """Configure model with hyperparameters passed in JSON format"""
        params = api.payload['params']
        row = session.query(db.Models).get(model_id)
        model = pickle.loads(row.model_binary)
        session.query(db.Models).filter(db.Models.model_id == model_id).update(
            {"model_binary": pickle.dumps(model.set_params(**params))}, synchronize_session="fetch")
        session.commit()
        return {'model': row.model_name, 'params': model.get_params()}

    @api.doc(responses={200: "Delete model successfully", 404: "Model with specified model_id not found"})
    def delete(self, model_id):
        """Drop the model"""
        all_ids = [i[0] for i in session.query(db.Models.model_id)]
        session.commit()
        if model_id in all_ids:
            model_name = session.query(db.Models).get(model_id).model_name
            session.query(db.Models).filter(db.Models.model_id == model_id).delete(synchronize_session="fetch")
            session.commit()
            return f"Model {model_name} was successfully deleted", 200
        else:
            return f"Incorrect model_id specified, must be one of the {all_ids}", 404


fit_fields = api.model('Train data', {'train_data': fields.List(fields.List(fields.Float()), default=[[1], [2], [5]]),
                                      'target': fields.List(fields.Float(), default=[1, 3, 8])})
@api.route("/model/<int:model_id>/fit")
class Fit(Resource):
    @api.expect(fit_fields)
    def put(self, model_id):
        """Fit model with data passed in JSON format"""
        row = session.query(db.Models).get(model_id)
        model = pickle.loads(row.model_binary)
        train_data, target = np.array(api.payload["train_data"]), np.array(api.payload["target"])
        model.fit(train_data, target)
        session.query(db.Models).filter(db.Models.model_id == model_id).update(
            {"model_binary": pickle.dumps(model)}, synchronize_session="fetch")
        session.commit()
        rmse = mean_squared_error(target, model.predict(train_data)) ** 0.5
        return {"status": f"{row.model_name[:-2]} is fitted on train data", "RMSE on train": round(rmse, 4)}


predict_fields = api.model('Predict data', {'data': fields.List(fields.List(fields.Float()), default=[[1], [2], [5]])})
@api.route("/model/<int:model_id>/predict")
class Predict(Resource):
    @api.doc(responses={200: "Success inference", 400: "Incorrect data format", 424: "Model not fitted yet"})
    @api.expect(predict_fields)
    def put(self, model_id):
        """Return prediction by data passed in JSON format"""
        row = session.query(db.Models).get(model_id)
        model = pickle.loads(row.model_binary)

        data = np.array(api.payload["data"])
        try:
            y_pred = model.predict(data)
            return {"y_pred": list(y_pred)}
        except NotFittedError as e:
            return repr(e), 424
        except:
            return "Incorrect data format, must be list of list with floats or integers", 400
