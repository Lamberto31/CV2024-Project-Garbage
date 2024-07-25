from flask import Flask, request
from Inference_MNAD import *

app = Flask(__name__)

@app.route('/api/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files['image']

    # Save the image file to disk
    image_file_path = './dataset/single/images/test.jpg'
    image_file.save(image_file_path)

    # Forward the image to the deep learning model for prediction
    result = classify(image_file_path)

    # Return the prediction result as a response
    if result == 1:
        return 'Dirty'
    else:
        return 'Clean'

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')