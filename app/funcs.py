import numpy as np
import yaml
import os
import xlrd
import re
# import datetime
from datetime import datetime, timedelta
from pathlib import Path
from flask import (Flask,
                   request,
                   jsonify,
                   render_template)
import requests
from keras.models import load_model

print('import success')
flask_app = Flask(__name__)


@flask_app.route("/",  methods=['post', 'get'])
def hello():
    if request.method == 'GET':
        return render_template('main.html')
        # return render_template(Path(__file__).parent / "templates/main.html")
        # return "Hello World!"
        # return str(Path(__file__).parent / "templates/main.html")
        # return '<form action="/echo" method="POST"><input name="text"><input type="submit" value="Echo"></form>'
        # return render_template("d:\\PythonProjects\\Data_Science_PRO\\ВКР\\app\\templates\\main.html")

    if request.method == 'POST':
        params = []
        for i in range(1, 13, 1):
            param = request.form.get(f'param_{i}')
            params.append(float(param))
        # print(params)
        net_3 = load_model(Path(__file__).parent.parent / "models/net_3.keras")
        x = np.array(params)
        predict = net_3.predict(np.expand_dims(x, axis=0))
        return render_template('main.html', result=predict.ravel()[0])

def run_flask():
    # print('Работет вроде!')
    # input('Нажмите что нибудь + Enter')
    """
    # print(Path(__file__).parent.parent / "models/net_3.keras")
    net_3 = load_model(Path(__file__).parent.parent / "models/net_3.keras")
    print(type(net_3))
    # проверим работу модели
    x = np.array([2030.0, 738.736842105263, 50.0, 23.75, 284.615384615384, 210.0, 70.0, 3000.0, 220.0, 0, 4.0, 60.0])
    predict = net_3.predict(np.expand_dims(x, axis=0))
    print(predict.ravel()[0])
    """
    flask_app.run(debug=False)
