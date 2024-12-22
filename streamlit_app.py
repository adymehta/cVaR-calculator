import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm
import altair as alt
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="CVaR Calculator",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

# CVaR Class Definition
class CVaR:
    def __init__(self, ticker, start_date, end_date, rolling_window, confidence_level, portfolio_val):
        self.ticker = ticker
        self.start = start_date
        self.end = end_date
        self.rolling = rolling_window
        self.conf_level = confidence_level
        self.portf_val = portfolio_val
        self.historical_cvar = None
        self.parametric_cvar = None

        self.data()

    def data(self):
        df = yf.download(self.ticker, self.start, self.end)
        if "Adj Close" in df.columns:
            self.adj_close_df = df["Adj Close"]
        elif "Close" in df.columns:
            self.adj_close_df = df["Close"]
        else:
            raise KeyError("Neither 'Adj Close' nor 'Close' columns found in the data.")
        
        self.log_returns_df = np.log(self.adj_close_df / self.adj_close_df.shift(1))
        self.log_returns_df = self.log_returns_df.dropna()
        self.equal_weights = np.array([1 / len(self.ticker)] * len(self.ticker))
        historical_returns = (self.log_returns_df * self.equal_weights).sum(axis=1)
        self.rolling_returns = historical_returns.rolling(window=self.rolling).sum()
        self.rolling_returns = self.rolling_returns.dropna()
        self.historical_method()
        self.parametric_method()

    def historical_method(self):
        var_threshold = -np.percentile(self.rolling_returns, 100 - (self.conf_level * 100))
        tail_losses = self.rolling_returns[self.rolling_returns < -var_threshold]
        historical_CVaR = -tail_losses.mean() * self.portf_val
        self.historical_cvar = historical_CVaR

    def parametric_method(self):
        self.cov_matrix = self.log_returns_df.cov() * 252
        self.portfolio_std = np.sqrt(np.dot(self.equal_weights.T, np.dot(self.cov_matrix, self.equal_weights)))
        mean_return = self.log_returns_df.mean().mean()

        var_threshold = mean_return - norm.ppf(self.conf_level) * self.portfolio_std * np.sqrt(self.rolling / 252)
        z_scores = (var_threshold - self.rolling_returns.mean()) / self.portfolio_std
        cvar_factor = norm.pdf(z_scores) / (1 - self.conf_level)
        parametric_CVaR = -cvar_factor * self.portfolio_std * self.portf_val
        self.parametric_cvar = parametric_CVaR

    def plot_cvar_results(self, title, cvar_value, returns_dollar, conf_level):
        plt.figure(figsize=(12, 6))
        plt.hist(returns_dollar, bins=50, density=True)
        plt.xlabel(f'\n {title} CVaR = ${cvar_value:.2f}')
        plt.ylabel('Frequency')
        plt.title(f"Distribution of Portfolio's {self.rolling}-Day Returns ({title} CVaR)")
        plt.axvline(-cvar_value, color='r', linestyle='dashed', linewidth=2, label=f'CVaR at {conf_level:.0%} confidence level')
        plt.legend()
        plt.tight_layout()
        return plt

if 'recent_outputs' not in st.session_state:
    st.session_state['recent_outputs'] = []

# Sidebar for User Inputs
with st.sidebar:
    st.title('📉 CVaR Calculator')

    tickers = st.text_input('Enter tickers separated by space', 'AAPL MSFT GOOG').split()
    start_date = st.date_input('Start date', value=pd.to_datetime('2020-01-01'))
    end_date = st.date_input('End date', value=pd.to_datetime('today'))
    rolling_window = st.slider('Rolling window', min_value=1, max_value=252, value=20)
    confidence_level = st.slider('Confidence level', min_value=0.90, max_value=0.99, value=0.95, step=0.01)
    portfolio_val = st.number_input('Portfolio value', value=100000)
    calculate_btn = st.button('Calculate CVaR')

# Calculation and Display

def calculate_and_display_cvar(tickers, start_date, end_date, rolling_window, confidence_level, portfolio_val):
    cvar_instance = CVaR(tickers, start_date, end_date, rolling_window, confidence_level, portfolio_val)

    # Layout for charts
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.info("Historical CVaR Chart")
        historical_chart = cvar_instance.plot_cvar_results("Historical", cvar_instance.historical_cvar, cvar_instance.rolling_returns * portfolio_val, confidence_level)
        st.pyplot(historical_chart)

    with chart_col2:
        st.info("Parametric CVaR Chart")
        parametric_chart = cvar_instance.plot_cvar_results("Parametric", cvar_instance.parametric_cvar, cvar_instance.rolling_returns * portfolio_val, confidence_level)
        st.pyplot(parametric_chart)

    # Layout for input summary and recent CVaR values
    col1, col3 = st.columns([1, 1])

    with col1:
        st.info("Input Summary")
        st.write(f"Tickers: {tickers}")
        st.write(f"Start Date: {start_date}")
        st.write(f"End Date: {end_date}")
        st.write(f"Rolling Window: {rolling_window} days")
        st.write(f"Confidence Level: {confidence_level:.2%}")
        st.write(f"Portfolio Value: ${portfolio_val:,.2f}")

    with col3:
        st.info("CVaR Calculation Output")
        data = {
            "Method": ["Historical", "Parametric"],
            "CVaR Value": [f"${cvar_instance.historical_cvar:,.2f}", f"${cvar_instance.parametric_cvar:,.2f}"]
        }
        df = pd.DataFrame(data)
        st.table(df)

    st.session_state['recent_outputs'].append({
        "Historical": f"${cvar_instance.historical_cvar:,.2f}",
        "Parametric": f"${cvar_instance.parametric_cvar:,.2f}"
    })

    # Display Recent CVaR Output table
    with col3:
        st.info("Previous CVaR Calculation Outputs")
        recent_df = pd.DataFrame(st.session_state['recent_outputs'])
        st.table(recent_df)

# Default Calculation for First Run
if 'first_run' not in st.session_state or st.session_state['first_run']:
    st.session_state['first_run'] = False
    default_tickers = 'AAPL MSFT GOOG'.split()
    default_start_date = pd.to_datetime('2020-01-01')
    default_end_date = pd.to_datetime('today')
    default_rolling_window = 20
    default_confidence_level = 0.95
    default_portfolio_val = 100000

    calculate_and_display_cvar(default_tickers, default_start_date, default_end_date, default_rolling_window, default_confidence_level, default_portfolio_val)

# Display Results on Button Click
if calculate_btn:
    calculate_and_display_cvar(tickers, start_date, end_date, rolling_window, confidence_level, portfolio_val)
