from flask import Flask, render_template, request
from deepforest import main
from deepforest import get_data
import cv2
import os

app = Flask(__name__)

def detect_trees(image_path):
    m = main.deepforest()
    m.use_release()
    
    imgpath = get_data(image_path)
    boxes = m.predict_image(path=imgpath, return_plot=False)
    plot = m.predict_image(path=imgpath, return_plot=True)
    
    return boxes, plot

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Replace with your actual validation logic
        if username == "admin" and password == "password":
            return render_template('/tree_detection')
        else:
            return render_template('login.html')

    return render_template('login.html')

@app.route('/tree_detection', methods=['GET', 'POST'])
def tree_detection():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file:
            # Save the uploaded file temporarily
            image_path = r'C:\Users\kasus\Downloads\SIHProject\New_test\static\uploaded_image.jpg'
            file.save(image_path)
            
            # Perform tree detection
            boxes, plot = detect_trees(image_path)

            # Save the result image temporarily
            result_path = r'C:\Users\kasus\Downloads\SIHProject\New_test\static\result_image.jpg'
            cv2.imwrite(result_path, plot)

            return render_template('tree_detection.html', message='File uploaded successfully',
                                   result_path=result_path, num_trees=len(boxes))

    return render_template('tree_detection.html')

if __name__ == '__main__':
    app.run(debug=True)
