from flask import Flask, render_template,request
from sklearn.externals import joblib

app = Flask(__name__)

@app.route("/", methods=['GET'])  #landing page
def index():
    return render_template("index.html")


@app.route('/getresult', methods=['POST'])  #click getresult button
def predict_stuff():
    if request.method == 'POST':
        model = joblib.load('trained_anomaly detection.pkl')

        age = int(request.form.get('age'))
        sex = int(request.form.get('sex'))
        height = int(request.form.get('height'))
        weight = int(request.form.get('weight'))
        QRS_duration = int(request.form.get('QRS_duration'))
        PR = int(request.form.get('PR'))
        QT = int(request.form.get('QT'))
        T = int(request.form.get('T'))
        P = int(request.form.get('P'))
        aQRS = int(request.form.get('aQRS'))
        aT = int(request.form.get('aT'))
        aP = int(request.form.get('aP'))
        aQRST = int(request.form.get('aQRST'))
        J = int(request.form.get('J'))
        heart_rate = int(request.form.get('heart_rate'))
        wQ = int(request.form.get('wQ'))
        wR = int(request.form.get('wR'))
        wS = int(request.form.get('wS'))
        wR_ = int(request.form.get('wR_'))
        IntrinsicDeflections = int(request.form.get('IntrinsicDeflections'))
        QRSA = float(request.form.get('QRSA'))
        QRSTA = float(request.form.get('QRSTA'))

        webdata = [age,sex,height,weight,QRS_duration,PR,QT,T,P,aQRS, aT, aP,aQRST,J, heart_rate,wQ,wR, wS ,wR_,IntrinsicDeflections,QRSA,QRSTA]
        webdata = [webdata]



        # Run the model and make a prediction
        predicted_value = model.predict(webdata)
        predicted_value = predicted_value[0]

        if predicted_value==0:
            predicted_value="It's Normal. Enjoy!"
        else:predicted_value="It's an anomaly, Sorry !"

        return render_template("index.html", pred=predicted_value)

if __name__ == "__main__":
    app.run()
