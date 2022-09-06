from flask import Flask,redirect,url_for,render_template,request
from tensorflow import keras
from tensorflow.keras.utils import load_img,img_to_array
# import cv2 as cv
import os

from keras.preprocessing import image
import numpy as np
import PIL
def img_saver(input_file):
    filename =  input_file.filename
    path = os.path.join(r"ml\static" , filename)
    input_file.save(path)
    return path

app=Flask(__name__)
@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='POST':
        # Handle POST Request here
        
        return render_template('template.html')
        
    return render_template('template.html')



@app.route('/result', methods=['POST'])
def result():
    picture = request.files["picture"]
    saved = img_saver(picture)
    model = keras.models.load_model(r"kaggle/working/final_model.h5")
    img_ = load_img(saved, target_size=(224, 224))
    imag = img_to_array(img_)
    imag = np.expand_dims(imag, axis=0)
    pred = model.predict(imag)
    pred = np.argmax(pred,axis=1)
    label =  {0:"Benign",1:"Malignant"}


    return render_template('result.html',pred=label[pred[0]])



if __name__ == '__main__':
    #DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=5000,debug=True)