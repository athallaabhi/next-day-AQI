import pandas as pd
import numpy as np
import joblib
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor
import gradio as gr

# 1) ISPU breakpoints
breakpoints = {
    'PM2.5': [(0.0,15.5,0,50),(15.6,55.4,51,100),(55.5,150.4,101,200),
              (150.5,250.4,201,300),(250.5,500.0,301,500)],
    'PM10':  [(0,50,0,50),(51,150,51,100),(151,350,101,200),
              (351,420,201,300),(421,600,301,500)],
    'SO2':   [(0,52,0,50),(53,180,51,100),(181,400,101,200),
              (401,800,201,300),(801,1200,301,500)],
    'CO':    [(0.0,4000,0,50),(4001,10000,51,100),(10001,15000,101,200),
              (15001,30000,201,300),(30001,60000,301,500)],
    'NO2':   [(0,80,0,50),(81,200,51,100),(201,1130,101,200),
              (1131,2260,201,300),(2261,5000,301,500)],
    'O3':    [(0.0,120.0,0,50),(121.0,235.0,51,100),(236.0,400.0,101,200),
              (401.0,800.0,201,300),(801.0,2000.0,301,500)]
}

# compute sub-index
def compute_subindex(C, breaks):
    for C_lo, C_hi, I_lo, I_hi in breaks:
        if C_lo <= C <= C_hi:
            return ((I_hi - I_lo)/(C_hi - C_lo))*(C - C_lo) + I_lo
    return breaks[-1][3]

# sandbox AQI calculation
def predict_sandbox(pm25, pm10, so2, co, o3, no2):
    vals = {'PM2.5': pm25, 'PM10': pm10, 'SO2': so2,
            'CO': co, 'O3': o3, 'NO2': no2}
    subs = {pol: compute_subindex(val, breakpoints[pol]) 
            for pol, val in vals.items()}
    aqi = max(subs.values())
    worst = max(subs, key=subs.get)
    return round(aqi,2), worst

# load & preprocess data (once)
def load_data():
    df = pd.read_csv('ispu_dki_all.csv')
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df = df.rename(columns={
        'pm25':'PM2.5_conc','pm10':'PM10_conc','so2':'SO2_conc',
        'co':'CO_conc','o3':'O3_conc','no2':'NO2_conc'
    })
    raw = ['PM2.5_conc','PM10_conc','SO2_conc',
           'CO_conc','O3_conc','NO2_conc']
    imp = IterativeImputer(
        estimator=KNeighborsRegressor(n_neighbors=5),
        max_iter=30, tol=1e-2, random_state=0
    )
    df[raw] = imp.fit_transform(df[raw])
    # sub-indices & AQI
    for pol in breakpoints:
        conc = 'PM2.5_conc' if pol=='PM2.5' else f"{pol}_conc"
        df[f"I_{pol.replace('.','')}"] = df[conc]\
            .apply(lambda x: compute_subindex(x, breakpoints[pol]))
    subidx = ['I_PM25','I_PM10','I_SO2','I_CO','I_NO2','I_O3']
    df['AQI'] = df[subidx].max(axis=1)
    df = df.sort_values('tanggal').reset_index(drop=True)
    # features & target
    df['AQI_today']  = df['AQI']
    df['pm25_missing'] = df['PM2.5_conc'].isna().astype(int)
    df['dow']       = df['tanggal'].dt.dayofweek
    df['is_weekend']= df['dow'].isin([5,6]).astype(int)
    df['AQI_tomorrow'] = df['AQI'].shift(-1)
    # slider ranges & date bounds
    ranges = {col:(float(df[col].min()), float(df[col].max())) 
              for col in raw}
    return df, ranges, df['tanggal'].min().date(), df['tanggal'].max().date()

df, slider_ranges, min_date, max_date = load_data()
rf = joblib.load('rf_model.joblib')
feature_cols = ['I_PM25','I_PM10','I_SO2','I_CO','I_NO2','I_O3',
                'pm25_missing','AQI_today','dow','is_weekend']

def predict_historical(date):
    date = pd.to_datetime(date)
    row = df[df['tanggal']==date]
    if row.empty:
        return None, None
    X = row[feature_cols].values.reshape(1,-1)
    pred = rf.predict(X)[0]
    nxt = df[df['tanggal']==date + pd.Timedelta(days=1)]
    actual = float(nxt['AQI']) if not nxt.empty else None
    return round(pred,2), round(actual,2) if actual is not None else None

with gr.Blocks() as demo:
    gr.Markdown("## Jakarta AQI Sandbox & Forecast")
    with gr.Tab("Sandbox"):
        s1 = gr.Slider(*slider_ranges['PM2.5_conc'], step=0.1,
                       label="PM2.5 (µg/m³)")
        s2 = gr.Slider(*slider_ranges['PM10_conc'], step=0.1,
                       label="PM10 (µg/m³)")
        s3 = gr.Slider(*slider_ranges['SO2_conc'], step=0.1,
                       label="SO2 (ppm)")
        s4 = gr.Slider(*slider_ranges['CO_conc'], step=0.1,
                       label="CO (ppm)")
        s5 = gr.Slider(*slider_ranges['O3_conc'], step=0.1,
                       label="O3 (ppm)")
        s6 = gr.Slider(*slider_ranges['NO2_conc'], step=0.1,
                       label="NO2 (ppm)")
        aqi_out = gr.Textbox(label="Predicted AQI")
        worst_out = gr.Textbox(label="Worst Pollutant")
        gr.Button("Compute AQI")\
          .click(predict_sandbox, [s1,s2,s3,s4,s5,s6],
                 [aqi_out, worst_out])

    with gr.Tab("Historical"):
        date_in = gr.Date(label="Select Date",
                          minimum=min_date, maximum=max_date)
        pred_out = gr.Textbox(label="Predicted AQI")
        act_out  = gr.Textbox(label="Actual AQI")
        gr.Button("Forecast AQI")\
          .click(predict_historical, date_in,
                 [pred_out, act_out])

demo.launch()
