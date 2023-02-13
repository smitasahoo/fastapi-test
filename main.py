from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf


app = FastAPI()

def get_predictions(data):
    model = tf.keras.models.load_model('user_model.h5')
    preds = model.predict(data)
    return preds
# Design the incoming feature data
class Features(BaseModel):
    user_id: int
    age: int
    sex: str
    income:  int
    last_connected:int
    location:str
    job:str

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Prediction endpoint
@app.post('/predict')
def get_prediction(incoming_data: Features):
    
    new_data = incoming_data.dict()
    query = {
    "user_id": tf.constant([new_data['user_id']]), # unknown user!
    "age": tf.constant([new_data['age']]),
    "sex": tf.constant([new_data['sex']]),
    "income": tf.constant([new_data['income']]),
    "last_connected": tf.constant([new_data['last_connected']]),
    "user_zip_code": tf.constant([new_data['last_connected']]),
    "location": tf.constant([new_data['location']]),
    "job": tf.ragged.constant([new_data['job']])
}
    preds = get_predictions(query)
    return {'predicted_class': preds}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
