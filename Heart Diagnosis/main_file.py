from flask import Flask, render_template, url_for, request
from keras.models import model_from_json
from sklearn.externals import joblib
import os
import numpy as np
app=Flask(__name__)#Shreyans Sonthalia
disease=""


@app.route("/")
def index():
    return render_template('home.html')

@app.route('/result', methods=['POST','GET'])
def result():
    age=int(request.form['age'])
    sex = int(request.form['sex'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = int(request.form['exang'])
    thal = int(request.form['thal'])
    cp=int(request.form['cp'])
    fbs=float(request.form['fbs'])
    x=np.array([age, sex,cp, trestbps, chol, fbs, restecg, thalach, exang, thal]).reshape(1,-1)


    clf=joblib.load(os.path.dirname(__file__)+'/models/model_1.pkl')
    y=int(clf.predict(x))
    print(y)
    global disease

    if(y==0 and chol<250):
        disease="No disease"
        return render_template('nodisease.html')
    elif(y==1 or 300<chol<350):
        disease = "Coronary Artery Disease"
        return render_template('CHD.htm')
    elif (y == 2):
        disease = "Silent Ischemia"
        return render_template('silisc.html')
    elif (y == 3 or age>65):
        disease = "Angina"
        return render_template('angina.html')
    else:
        disease = "Emphysema"
        return render_template('Emphysema.html')



@app.route('/about')
def about():
    return render_template('about.html')

if __name__=="__main__":
    app.run(debug=True)
