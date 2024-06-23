from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import pandas as pd
from werkzeug.utils import secure_filename
import os
from model import predict
from sklearn.preprocessing import LabelEncoder
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        app.logger.debug('No file part in the request')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        app.logger.debug('No file selected for uploading')
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        app.logger.debug(f'File saved to {filepath}')
        
        data = pd.read_csv(filepath)
        
        # Lakukan encoding yang sama seperti saat pelatihan model
        le = LabelEncoder()
        data['ProductCategory'] = le.fit_transform(data['ProductCategory'])
        data['ProductBrand'] = le.fit_transform(data['ProductBrand'])
        
        # Drop kolom target jika ada
        data = data.loc[:, 'ProductCategory':'CustomerSatisfaction']
        
        predictions = predict(data)
        
        result_df = pd.DataFrame(predictions, columns=['Predictions'])
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result.csv')
        result_df.to_csv(result_path, index=False)
        
        app.logger.debug(f'Result saved to {result_path}')
        
        return render_template('result.html', result='result.csv')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
