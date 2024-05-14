from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import io

app = Flask(__name__)

def model_prediction(file):
    model = tf.keras.models.load_model('trained_model.keras')
    
    # Convert the FileStorage object to a BytesIO object
    image = tf.keras.preprocessing.image.load_img(io.BytesIO(file.read()), target_size=(128, 128))
    
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    result_index = model_prediction(file)
    print(result_index)
    if result_index == 2:
        return jsonify({'result': 'Healthy'})
    else:
        return jsonify({'result': 'Disease'})

@app.route('/')
def index():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
