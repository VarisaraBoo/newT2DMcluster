from flask import Flask, render_template, request
import pickle
import numpy as np
from pickle import load

# load the model
model = load(open('model_kmean.pkl', 'rb'))

# load the scaler
scaler = load(open('scaler_kmean.pkl', 'rb'))


app = Flask(__name__)  # initializing Flask app

@app.route("/",methods=['GET'])
def hello():
    return (render_template('model_HTML.html'))

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        d1 = request.form['Age']
        d2 = request.form['Height']
        d3 = request.form['Weight']
        d4 = int(d3)/((int(d2)/100)**2)
        d4 = int(d4)
        d5 = request.form['HbA1c']
        d6 = request.form['Triglyceride']
        d7 = request.form['HDL']

        
        arr1 = np.array([[d1, d4, d6,d5, d7]]) 

 
        newdata_scaled = scaler.transform(arr1)
        cluster = (model.predict(newdata_scaled))+1
        kmeans = ""
        if cluster == 1 :
            kmeans = 'cluster 3 : MOD'
        elif cluster == 2 :
            kmeans = 'cluster 4 : MARD'
        elif cluster == 3 :
            kmeans = 'cluster 2 : MSD'
        elif cluster == 4 :
            kmeans = 'cluster 1 : SIDD'

        char = ""
        if cluster == 1 :
            char = 'High BMI, low HbA1C, young age'
        elif cluster == 2 :
            char = 'Older age and low HbA1C at diagnosis'
        elif cluster == 3 :
            char = 'High TG, low HDL-C, averaged age and BMI'
        elif cluster == 4 :
            char = 'High HbA1C, Low BMI'       
        print(arr1)
        kmeans = kmeans
        char = char
        return render_template('model_HTML2.html', kmeans=kmeans, char=char )
    else:
        return render_template('model_HTML2.html') 


if __name__ == '__main__':
   app.run()
#app.run(host="0.0.0.0")            # deploy
# app.run(debug=True)                # run on local system
