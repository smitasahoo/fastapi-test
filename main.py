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
    user_id: tf.int64
    age: tf.int64
    sex: tf.string
    income:  tf.int64
    last_connected:tf.int64
    location:tf.string
    job:tf.string

@app.get("/")
def read_root():
    return {"Hello": "World"}

# Prediction endpoint
@app.post('/predict')
def get_prediction(incoming_data: Features):
    new_data = incoming_data.dict()
    preds = get_predictions(new_data)
    return {'predicted_class': preds}

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}
