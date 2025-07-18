import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

# Utility function to load pre-trained model
@st.cache_resource
def load_model(model_type):
    model_map = {
        'Vanilla LSTM': 'models/model_vanilla.h5',
        'Bidirectional LSTM': 'models/model_bi.h5',
        'Stacked LSTM': 'models/model_stacked.h5'
    }
    return tf.keras.models.load_model(model_map[model_type])

# Sequence generator

def create_sequences(data, timesteps=60):
    X, y = [], []
    for i in range(timesteps, len(data)):
        X.append(data[i-timesteps:i, :])
        y.append(data[i, 0])  # kolom pertama sebagai target (Close)
    return np.array(X), np.array(y)

# Streamlit App
def main():
    st.title('Stock Prediction App with LSTM')
    st.sidebar.header('Pengaturan')

    uploaded_file = st.sidebar.file_uploader('Upload CSV file', type=['csv'])
    timesteps = st.sidebar.number_input('Timesteps', min_value=1, max_value=200, value=60)
    model_choice = st.sidebar.selectbox('Pilih Model', ['Vanilla LSTM', 'Bidirectional LSTM', 'Stacked LSTM'])

    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
        df = data[['Close', 'Volume']].values

        # Preview data
        if st.checkbox('Tampilkan Data Mentah'):
            st.dataframe(pd.DataFrame(df, index=data.index, columns=['Close', 'Volume']).head())

        # Load model (diasumsikan sudah mencakup preprocessing/normalisasi di pipeline)
        model = load_model(model_choice)

        # Prepare sequences
        X, y_true = create_sequences(df, timesteps)

        # Prediction (langsung menggunakan model yang sudah memiliki layer normalisasi)
        y_pred = model.predict(X).flatten()

        # Inverse: karena output model sudah pada skala asli, langsung gunakan
        y_true_raw = y_true

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_true_raw, y_pred))
        mae = mean_absolute_error(y_true_raw, y_pred)
        mape = np.mean(np.abs((y_true_raw - y_pred) / y_true_raw)) * 100

        st.subheader('Hasil Prediksi')
        st.write(f'RMSE: {rmse:.4f}')
        st.write(f'MAE: {mae:.4f}')
        st.write(f'MAPE: {mape:.2f}%')

        # Plot
        series = pd.DataFrame({'Actual': y_true_raw, 'Predicted': y_pred}, index=data.index[timesteps:])
        st.line_chart(series)

if __name__ == '__main__':
    main()
