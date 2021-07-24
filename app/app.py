from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid
from joblib import load
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
app = Flask(__name__, static_folder='static',
            template_folder='templates')

@app.route("/", methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html', year=2021, county='Galway', house_type='New Dwelling house /Apartment', prediction='364555')
    else:
        year_ = request.form['sel1']
        county_ = request.form['sel2']
        house_type_ = request.form['sel3']

        data = [[int(year_), county_, house_type_]]
        test_df = pd.DataFrame(data, columns = ['SALE_DATE', 'COUNTY', 'PROPERTY_DESC'])
                
        # make it load('app/pipe.pkl')     
        pipe = load('app/pipe.pkl')
        prediction_ = pipe.predict(test_df)   
        prediction_ = str(round(prediction_[0]))   
        house_type_ = house_type_[0:3]
        return render_template('index.html', year=year_, county=county_, house_type=house_type_, prediction=prediction_)



