import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

# Allow user to input any stock ticker
select_stock = st.text_input("Enter stock ticker for prediction", value="").upper()

n_years = st.slider("Years of prediction:", 1, 5)
period = n_years * 365

def validate_stock(ticker):
    """
    Validate the stock ticker by checking if it exists in the Yahoo Finance API.
    Return True if it exists, False otherwise.
    """
    if ticker == "":
        return False, "Please enter a stock ticker."
    
    stock = yf.Ticker(ticker)
    info = stock.info
    # A valid stock will have the 'sector' key in its info dictionary
    if 'sector' in info:
        return True, ""
    else:
        return False, "The stock ticker you entered does not exist. Please try again."

# Validate the stock ticker
is_valid, error_message = validate_stock(select_stock)

if not is_valid:
    st.warning(error_message)
else:
    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    # Load and display the data
    data_load_state = st.text("Load data...")
    data = load_data(select_stock)
    data_load_state.text("Loading data..done!")

    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw stock data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # Add Moving Averages
    def add_moving_averages(data):
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        return data

    data = add_moving_averages(data)

    st.subheader("Stock with Moving Averages")
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close'))
    fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['SMA_20'], name='SMA 20'))
    fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], name='SMA 50'))
    fig_ma.layout.update(title_text="Stock Price with Moving Averages", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig_ma)

    # Train the Prophet Model
    df_train = data[['Date', 'Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    # Prophet model with adjustable parameters
    yearly_seasonality = st.checkbox('Yearly Seasonality', value=True)
    weekly_seasonality = st.checkbox('Weekly Seasonality', value=True)
    daily_seasonality = st.checkbox('Daily Seasonality', value=False)
    changepoint_prior_scale = st.slider('Changepoint Prior Scale', 0.001, 0.5, 0.05)

    m = Prophet(yearly_seasonality=yearly_seasonality, 
                weekly_seasonality=weekly_seasonality, 
                daily_seasonality=daily_seasonality,
                changepoint_prior_scale=changepoint_prior_scale)
    m.fit(df_train)

    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Display the forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())

    st.write('Forecast Data')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write('Forecast Components')
    fig2 = m.plot_components(forecast)
    st.write(fig2)

    # Stock Information Overview
    def display_stock_info(ticker):
        stock = yf.Ticker(ticker)
        st.subheader(f"Overview of {ticker}")
        info = stock.info
        if 'sector' in info:
            st.write(f"**Sector**: {info['sector']}")
            st.write(f"**Market Cap**: {info['marketCap']}")
            st.write(f"**PE Ratio**: {info['trailingPE']}")
            st.write(f"**Dividend Yield**: {info['dividendYield']}")
            st.write(f"**52 Week High**: {info['fiftyTwoWeekHigh']}")
            st.write(f"**52 Week Low**: {info['fiftyTwoWeekLow']}")
        else:
            st.write("No overview data available for this stock.")

    display_stock_info(select_stock)

    # Highlight Financial Events
    def highlight_events(ticker):
        stock = yf.Ticker(ticker)
        events = stock.actions
        if not events.empty:
            st.subheader(f"Financial Events of {ticker}")
            st.write(events)

    highlight_events(select_stock)

    # Download Data Feature
    def download_data(data):
        csv = data.to_csv(index=False)
        st.download_button(label="Download data as CSV", data=csv, mime='text/csv')

    download_data(data)
