# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from functools import lru_cache
from datetime import timedelta

# Imputer imports
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor

app = Flask(__name__)

# 1) ISPU breakpoints
breakpoints = {
    'PM2.5': [(0.0,15.5,0,50),(15.6,55.4,51,100),(55.5,150.4,101,200),(150.5,250.4,201,300),(250.5,500.0,301,500)],
    'PM10':  [(0,50,0,50),(51,150,51,100),(151,350,101,200),(351,420,201,300),(421,600,301,500)],
    'SO2':   [(0,52,0,50),(53,180,51,100),(181,400,101,200),(401,800,201,300),(801,1200,301,500)],
    'CO':    [(0.0,4000,0,50),(4001,10000,51,100),(10001,15000,101,200),(15001,30000,201,300),(30001,60000,301,500)],
    'NO2':   [(0,80,0,50),(81,200,51,100),(201,1130,101,200),(1131,2260,201,300),(2261,5000,301,500)],
    'O3':    [(0.0,120.0,0,50),(121.0,235.0,51,100),(236.0,400.0,101,200),(401.0,800.0,201,300),(801.0,2000.0,301,500)]
}

# 2) compute sub-index helper
def compute_subindex(C, breaks):
    for C_lo, C_hi, I_lo, I_hi in breaks:
        if C_lo <= C <= C_hi:
            return ((I_hi - I_lo)/(C_hi - C_lo))*(C - C_lo) + I_lo
    return breaks[-1][3]

# 3) sandbox AQI calculator
def calculate_aqi(inputs):
    subs = {
        pol: compute_subindex(inputs[f"{pol}_conc"], breakpoints[pol])
        for pol in breakpoints
    }
    aqi = max(subs.values())
    worst = max(subs, key=subs.get)
    return aqi, worst, subs

# 4) load and prepare data
@lru_cache(maxsize=1)
def load_data():
    df = pd.read_csv('ispu_dki_all.csv')
    df['tanggal'] = pd.to_datetime(df['tanggal'], format='%Y-%m-%d')
    df = df.rename(columns={
        'pm25':'PM2.5_conc','pm10':'PM10_conc','so2':'SO2_conc',
        'co':'CO_conc','o3':'O3_conc','no2':'NO2_conc'
    })

    # impute missing pollutant data
    raw_cols = ['PM2.5_conc','PM10_conc','SO2_conc','CO_conc','O3_conc','NO2_conc']
    imputer = IterativeImputer(
        estimator=KNeighborsRegressor(n_neighbors=5),
        max_iter=30, tol=1e-2, random_state=0
    )
    df[raw_cols] = imputer.fit_transform(df[raw_cols])

    # compute sub-indices
    for pol in breakpoints:
        conc = 'PM2.5_conc' if pol=='PM2.5' else f"{pol}_conc"
        df[f"I_{pol.replace('.', '')}"] = df[conc].apply(
            lambda x: compute_subindex(x, breakpoints[pol])
        )

    # compute AQI and features
    subidx = ['I_PM25','I_PM10','I_SO2','I_CO','I_NO2','I_O3']
    df['AQI'] = df[subidx].max(axis=1)
    df = df.sort_values('tanggal').reset_index(drop=True)
    df['AQI_today']    = df['AQI']
    df['pm25_missing'] = df['PM2.5_conc'].isna().astype(int)
    df['dow']          = df['tanggal'].dt.dayofweek
    df['is_weekend']   = df['dow'].isin([5,6]).astype(int)
    df['AQI_tomorrow'] = df['AQI'].shift(-1)

    # slider ranges from actual data
    slider_ranges = {
        'PM2.5_conc': (float(df['PM2.5_conc'].min()), float(df['PM2.5_conc'].max())),
        'PM10_conc':  (float(df['PM10_conc'].min()),  float(df['PM10_conc'].max())),
        'SO2_conc':   (float(df['SO2_conc'].min()),   float(df['SO2_conc'].max())),
        'CO_conc':    (float(df['CO_conc'].min()),    float(df['CO_conc'].max())),
        'O3_conc':    (float(df['O3_conc'].min()),    float(df['O3_conc'].max())),
        'NO2_conc':   (float(df['NO2_conc'].min()),   float(df['NO2_conc'].max()))
    }
    min_date = df['tanggal'].min().date().isoformat()
    max_date = df['tanggal'].max().date().isoformat()

    return df, slider_ranges, min_date, max_date

# 5) load Random Forest model only
@lru_cache(maxsize=1)
def load_model():
    return joblib.load('rf_model.joblib')

# 6) main route
@app.route('/', methods=['GET','POST'])
def index():
    df, slider_ranges, min_date, max_date = load_data()
    rf = load_model()
    feature_cols = ['I_PM25','I_PM10','I_SO2','I_CO','I_NO2','I_O3',
                    'pm25_missing','AQI_today','dow','is_weekend']

    mode   = request.form.get('mode', 'sandbox')
    inputs, result, error = {}, None, None

    if request.method == 'POST':
        if mode == 'sandbox':
            try:
                vals = {col: float(request.form[col]) for col in slider_ranges}
                aqi_val, worst, _ = calculate_aqi(vals)
                result = {'aqi': aqi_val, 'worst': worst}
                inputs = vals
            except Exception as e:
                error = str(e)

        else:  # historical
            hist = request.form.get('hist_date','')
            if hist:
                try:
                    date = pd.to_datetime(hist)
                    row  = df[df['tanggal']==date].iloc[0]
                    Xf   = row[feature_cols].values.reshape(1,-1)
                    pred = rf.predict(Xf)[0]
                    actual = df[df['tanggal']==date+timedelta(days=1)]['AQI'].iloc[0]
                    result = {'pred': pred, 'actual': actual}
                    inputs = {'hist_date': hist}
                except:
                    error = 'No data for that date.'

    return render_template(
        'index.html',
        mode=mode,
        slider_ranges=slider_ranges,
        min_date=min_date,
        max_date=max_date,
        inputs=inputs,
        result=result,
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True)
