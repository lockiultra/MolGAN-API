from fastapi import FastAPI
import uvicorn

from utils import get_model, get_disease_pipeline, predict_to_json, get_help_message

app = FastAPI()
model = None
dp = None

@app.on_event('startup')
def startup():
    global model, dp
    model = get_model()
    dp = get_disease_pipeline()

@app.get('/')
def root():
    return {'message': 'Hello World'}

@app.get('/help')
def help():
    return get_help_message()

@app.get('/molgan/sample_mol')
def sample_mol():
    global model
    # model = get_model()
    return {'SMILES': model.sample_prior()}

@app.get('/molgan/predict')
def predict_test(smiles):
    global dp
    # dp = get_disease_pipeline()
    try:
        predict_json = predict_to_json(dp.predict([smiles]))
        predict_json['status'] = 'OK'
    except:
        predict_json = {'status': 'NOT OK', 'error': 'Invalid SMILES'}
    return predict_json


# @app.post('/molgan/predict/{smiles}')
# def predict(smiles):
#     dp = get_disease_pipeline()
#     predict_json = predict_to_json(dp.predict([smiles]))
#     return predict_json

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8080)