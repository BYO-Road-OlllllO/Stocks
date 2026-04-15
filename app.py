import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import date
from streamlit_gsheets import GSheetsConnection # NEW: Import for Google Sheets

# Set the timeframe for historical data
START_DATE = "2024-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# App Layout & Title
st.set_page_config(page_title="Portfolio Tracker & Projector", layout="wide")
st.title('📈 Portfolio Tracker & Market Projector')

# --- NEW: Google Sheets Integration ---
# Create a connection object
conn = st.connection("gsheets", type=GSheetsConnection)

@st.cache_data(ttl=600) # Caches the data for 10 minutes to avoid API limits
def load_portfolio_from_sheets():
    try:
        # Read the specific tab. 
        # usecols: A=0, J=9, O=14. Assuming row 1 is headers.
        df = conn.read(
            worksheet="Stocks",
            usecols=[0, 9, 14] 
        )
        # Rename columns so they are easy to reference
        df.columns = ["Ticker", "Shares", "Cost_Per_Share"]
        
        # Clean the data (remove empty rows)
        df = df.dropna(subset=["Ticker"])
        df['Ticker'] = df['Ticker'].astype(str).str.upper().str.strip()
        return df
    except Exception as e:
        st.sidebar.error(f"Could not load Google Sheet. Ensure your secrets.toml is set up. Error: {e}")
        return pd.DataFrame(columns=["Ticker", "Shares", "Cost_Per_Share"])

# Load data from the sheet
sheet_portfolio_df = load_portfolio_from_sheets()
sheet_tickers = sheet_portfolio_df["Ticker"].tolist() if not sheet_portfolio_df.empty else []

# Pre-populated portfolio (Fallback / Baseline)
base_portfolio = ['AI', 'BAC', 'BCTK', 'CSX', 'DAN', 'FSTA', 'FTEC', 'GOOG', 'GOOGL', 'JEPQ', 'LUV', 'MITT', 'PSKY', 'RIVN', 'SCHD', 'SCHY', 'TSLA', 'VOO', 'WBD', 'XOVR', 'AAL', 'AFL', 'AKA', 'ARQQ', 'ASTI', 'BBAI', 'BCTX', 'BITF', 'BKYI', 'CAVA', 'CRNT', 'FDIS', 'FHLC', 'GME', 'HIMS', 'HOOD', 'HSDT', 'IHRT', 'IONQ', 'IWM', 'JNJ', 'JPM', 'K', 'KULR', 'LLY', 'MARA', 'MSTR', 'NFLX', 'NXST', 'OXY', 'PHIO', 'PLTR', 'PSEC', 'QBTS', 'QQQ', 'QUBT', 'RCAT', 'RGTI', 'RIME', 'SBET', 'SERV', 'SFTBY', 'SIDU', 'SIRI', 'SMR', 'SOFI', 'SOUN', 'SPY', 'TSM', 'UUUU', 'VRSN', 'VTI', 'VXUS', 'WKEY', 'ZONE']

# Combine base portfolio with Google Sheet tickers and remove duplicates
all_tickers = sorted(list(set(base_portfolio + sheet_tickers)))

# --- Sidebar controls ---
st.sidebar.header("Configuration")

# NEW: Allow user to manually type an additional stock ticker
custom_ticker = st.sidebar.text_input("Add a custom ticker (e.g., AAPL):").upper().strip()
if custom_ticker and custom_ticker not in all_tickers:
    all_tickers.append(custom_ticker)
    all_tickers = sorted(all_tickers)

# Dropdown uses the newly merged list
selected_stock = st.sidebar.selectbox('Select an asset to view:', all_tickers)

# NEW: Display Google Sheets position data if the selected stock is in the sheet
if not sheet_portfolio_df.empty and selected_stock in sheet_portfolio_df["Ticker"].values:
    # Get the row for the selected stock
    stock_info = sheet_portfolio_df[sheet_portfolio_df["Ticker"] == selected_stock].iloc[0]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("💼 Your Position (From Sheets)")
    st.sidebar.write(f"**Shares Owned:** {stock_info['Shares']}")
    st.sidebar.write(f"**Cost Per Share:** ${stock_info['Cost_Per_Share']}")
    
    # Optional: Calculate total cost basis
    try:
        total_cost = float(stock_info['Shares']) * float(stock_info['Cost_Per_Share'])
        st.sidebar.write(f"**Total Cost Basis:** ${total_cost:,.2f}")
    except ValueError:
        pass # Silently pass if the sheet contains non-numeric data in those columns

st.sidebar.markdown("---")
n_years = st.sidebar.slider('Years of projection:', 1, 5)
period = n_years * 365

# Function to fetch data via yfinance
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START_DATE, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading market data...')
data = load_data(selected_stock)
data_load_state.text('Market data loaded successfully!')

# Section 1: Historical Data
st.subheader(f'Historical Market Data for {selected_stock}')
st.dataframe(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open Price", line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price", line=dict(color='#ff7f0e')))
    fig.layout.update(title_text='Historical Time Series with Range Slider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)
    
plot_raw_data()

# Section 2: Projections & Forecasting
st.subheader('🔮 Market Trend Projections')
st.write('Calculating projections based on historical momentum and seasonal trends...')

# Prepare the data frame for Prophet
df_train = pd.DataFrame()
df_train['ds'] = data['Date']

# 1. Safely extract the 'Close' column
close_data = data['Close']
if isinstance(close_data, pd.DataFrame):
    close_data = close_data.iloc[:, 0]

# 2. Force the data to be numeric
df_train['y'] = pd.to_numeric(close_data, errors='coerce')

# 3. Strip timezone data from the dates
if df_train['ds'].dt.tz is not None:
    df_train['ds'] = df_train['ds'].dt.tz_localize(None)

# 4. Drop any missing values
df_train = df_train.dropna()

# Initialize and fit the Prophet model
m = Prophet(daily_seasonality=True)
m.fit(df_train)

# Create future dates and predict
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Display Projection Data
st.write(f'**Projection values for the next {n_years} year(s):**')
st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot the interactive forecast chart
st.write('**Interactive Projection Chart:**')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1, use_container_width=True)

# Plot the individual trend components
st.write("**Projection Components (Overall Trend vs. Yearly/Weekly Seasonality):**")
fig2 = m.plot_components(forecast)
st.pyplot(fig2)