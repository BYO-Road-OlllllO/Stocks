import streamlit as st
import yfinance as yf
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import date
from streamlit_gsheets import GSheetsConnection 

# Set the timeframe for historical data
START_DATE = "2024-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# App Layout & Title
st.set_page_config(page_title="Portfolio Tracker & Projector", layout="wide")
st.title('📈 Portfolio Tracker & Market Projector')

# --- Google Sheets Integration ---
# Create a connection object
conn = st.connection("gsheets", type=GSheetsConnection)

@st.cache_data(ttl=600) # Caches the data for 10 minutes to avoid API limits
def load_portfolio_from_sheets():
    try:
        # Read the specific tab. 
        # usecols: A=0, J=9, O=14. Assuming row 1 is headers.
        df = conn.read(
            worksheet="1678747277",
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

# Allow user to manually type an additional stock ticker
custom_ticker = st.sidebar.text_input("Add a custom ticker (e.g., AAPL):").upper().strip()
if custom_ticker and custom_ticker not in all_tickers:
    all_tickers.append(custom_ticker)
    all_tickers = sorted(all_tickers)

# Dropdown uses the newly merged list
selected_stock = st.sidebar.selectbox('Select an asset to view:', all_tickers)

# Display Google Sheets position data if the selected stock is in the sheet
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

# --- NEW: Dividend Fetching Function ---
@st.cache_data
def load_dividend_data(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    divs = ticker.dividends
    if not divs.empty:
        # Remove timezone information to make filtering easier
        divs.index = divs.index.tz_localize(None) 
        divs = divs[(divs.index >= pd.to_datetime(START_DATE)) & (divs.index <= pd.to_datetime(TODAY))]
    return divs

data_load_state = st.text('Loading market data...')
data = load_data(selected_stock)
div_data = load_dividend_data(selected_stock) # NEW: Call the dividend fetcher
data_load_state.text('Market data loaded successfully!')

# --- Volatility & Risk Calculation Function ---
@st.cache_data
def calculate_risk_metrics(ticker):
    # Fetch stock and market (SPY) data for the last year for beta calculation
    s_data = yf.download(ticker, START_DATE, TODAY)['Close']
    m_data = yf.download('SPY', START_DATE, TODAY)['Close']
    
    # Flatten columns if they are MultiIndex (common in newer yfinance versions)
    if isinstance(s_data, pd.DataFrame): s_data = s_data.iloc[:, 0]
    if isinstance(m_data, pd.DataFrame): m_data = m_data.iloc[:, 0]
    
    # Calculate daily returns
    s_ret = s_data.pct_change().dropna()
    m_ret = m_data.pct_change().dropna()
    
    # Align data and calculate Beta
    combined = pd.concat([s_ret, m_ret], axis=1).dropna()
    combined.columns = ['Stock', 'Market']
    beta = combined.cov().iloc[0, 1] / combined['Market'].var()
    volatility = combined['Stock'].std() * (252**0.5) 
    
    return beta, volatility

# Run the calculation right after data is loaded
try:
    beta, vol = calculate_risk_metrics(selected_stock)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚠️ Risk Analysis")
    
    # Visual cues for risk levels
    if beta > 1.5:
        st.sidebar.error(f"High Volatility: {beta:.2f} Beta")
    elif beta < 0.8:
        st.sidebar.success(f"Low Volatility: {beta:.2f} Beta")
    else:
        st.sidebar.info(f"Market Neutral: {beta:.2f} Beta")
        
    st.sidebar.write(f"**Annual Volatility:** {vol*100:.1f}%")
except Exception as e:
    st.sidebar.warning("Risk metrics unavailable for this ticker.")
    
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


# --- NEW: Section 3: Total Dividend Tracking & Projections ---
st.markdown("---")
st.subheader(f'💰 Dividend Income for {selected_stock}')

if not div_data.empty:
    # 1. Grab the number of shares from your Google Sheet (Default to 1 if not found)
    shares_owned = 1.0
    using_real_shares = False
    
    if not sheet_portfolio_df.empty and selected_stock in sheet_portfolio_df["Ticker"].values:
        try:
            stock_row = sheet_portfolio_df[sheet_portfolio_df["Ticker"] == selected_stock].iloc[0]
            
            # NEW FIX: Convert to text, strip out commas/spaces, then turn back to a number
            raw_shares = str(stock_row['Shares']).replace(',', '').replace(' ', '').replace('$', '')
            shares_owned = float(raw_shares)
            
            using_real_shares = True
        except ValueError:
            # This will now actually tell you WHY it failed if it happens again
            st.warning(f"Found {selected_stock} in your sheet, but couldn't read the shares. The sheet says: '{stock_row['Shares']}'")

    # 2. Status message for the user
    if using_real_shares:
        st.success(f"Calculations based on your actual holdings: **{shares_owned} shares** (Synced from Google Sheets)")
    else:
        st.info("Stock not found in Google Sheets. Showing default payout per 1 share.")

    # Calculate Annual Dividends
    annual_divs = div_data.resample('YE').sum() 
    last_full_year = date.today().year - 1
    annual_divs_full = annual_divs[annual_divs.index.year <= last_full_year]

    if len(annual_divs_full) >= 2:
        # Calculate Compound Annual Growth Rate (CAGR)
        start_val = annual_divs_full.iloc[0]
        end_val = annual_divs_full.iloc[-1]
        
        # Safely force standard numbers
        if isinstance(start_val, pd.Series): start_val = start_val.iloc[0]
        if isinstance(end_val, pd.Series): end_val = end_val.iloc[0]
        start_val, end_val = float(start_val), float(end_val)
        
        years_diff = len(annual_divs_full) - 1
        cagr = ((end_val / start_val) ** (1/years_diff) - 1) if start_val > 0.0 else 0.0

        # Project Future Dividends & MULTIPLY BY SHARES OWNED
        st.write(f"**Projected Annual Cash Flow ({n_years} Years)**")
        st.write(f"*Historical Dividend Growth Rate:* **{cagr:.2%}**")
        
        future_years = [date.today().year + i for i in range(1, n_years + 1)]
        # Multiply the projected per-share dividend by your total shares
        proj_divs_total = [(end_val * ((1 + cagr) ** i)) * shares_owned for i in range(1, n_years + 1)]

        fig_proj_div = go.Figure(data=[go.Bar(
            x=future_years, 
            y=proj_divs_total, 
            name="Projected Cash Income", 
            marker_color='#98df8a',
            text=[f"${val:,.2f}" for val in proj_divs_total],
            textposition='auto'
        )])
        fig_proj_div.layout.update(
            title_text=f'Projected Total Cash Income from {selected_stock}', 
            xaxis_title="Year", 
            yaxis_title="Total Estimated Cash Payout ($)"
        )
        st.plotly_chart(fig_proj_div, use_container_width=True)
    else:
        st.info("Not enough historical full-year data to accurately project future dividend growth.")
else:
    st.info(f"{selected_stock} does not currently pay a dividend, or dividend data is unavailable.")
