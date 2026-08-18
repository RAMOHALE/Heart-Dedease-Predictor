import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import os
import sys
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from allratestoday import AllRatesToday
    HAS_ALLRATES = True
except ImportError:
    HAS_ALLRATES = False

# ==========================================
# 0. LOGGING SETUP
# ==========================================
# print() only shows up in the terminal running `streamlit run`, never in the
# browser - which is why failures can make the page look silently blank.
# This logger writes to the console (for you, in the terminal) AND we mirror
# key events to the Streamlit page itself via st.error/st.exception below,
# so problems are visible wherever you're actually looking.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("heart_disease_dashboard")

st.set_page_config(page_title="Heart Disease & Economic Dashboard", layout="wide")
st.title("Heart Disease Insights: Africa")
logger.info("App started.")

# ==========================================
# 1. DATA FETCHING FUNCTIONS
# ==========================================

DATA360_BASE_URL = "https://data360api.worldbank.org"

# ==========================================
# EXCHANGE RATE FETCHING
# ==========================================

@st.cache_data(ttl=3600)
def get_live_exchange_rates():
    """
    Fetch live exchange rates using allratestoday API.
    Falls back to 2019 rates if API fails.
    """
    fallback_rates = {
        'ZAF': ('ZAR', 'R', 14.47),
        'NGA': ('NGN', '₦', 360.73),
        'BWA': ('BWP', 'P', 10.65),
        'LSO': ('LSL', 'L', 14.47),
        'MOZ': ('MZN', 'MT', 62.55),
        'RWA': ('RWF', 'Fr', 895.83),
        'SYC': ('SCR', '₨', 13.64),
        'UGA': ('UGX', 'Sh', 3695.75),
        'ZMB': ('ZMW', 'ZK', 13.23),
        'ZWE': ('ZWL', '$', 79.65)
    }
    
    if not HAS_ALLRATES:
        logger.warning("[exchange] allratestoday not installed. Using fallback rates.")
        return fallback_rates
    
    try:
        client = AllRatesToday()
        live_rates = {}
        
        currency_map = {
            'ZAF': ('ZAR', 'R'),
            'NGA': ('NGN', '₦'),
            'BWA': ('BWP', 'P'),
            'LSO': ('LSL', 'L'),
            'MOZ': ('MZN', 'MT'),
            'RWA': ('RWF', 'Fr'),
            'SYC': ('SCR', '₨'),
            'UGA': ('UGX', 'Sh'),
            'ZMB': ('ZMW', 'ZK'),
            'ZWE': ('ZWL', '$')
        }
        
        for country_code, (curr_code, symbol) in currency_map.items():
            try:
                result = client.get_rate("USD", curr_code)
                rate = result.get('rate', fallback_rates[country_code][2])
                live_rates[country_code] = (curr_code, symbol, float(rate))
                logger.info(f"[exchange] {country_code} ({curr_code}): 1 USD = {rate}")
            except Exception as e:
                logger.warning(f"[exchange] Failed to fetch {country_code}: {e}")
                live_rates[country_code] = fallback_rates[country_code]
        
        return live_rates
    
    except Exception as e:
        logger.error(f"[exchange] Failed to initialize AllRatesToday: {e}")
        return fallback_rates

@st.cache_data
def get_worldbank_data(indicator_dot_code, country_codes, year=2019):
    """
    Fetch indicator values from the World Bank Data360 API in parallel.
    Uses ThreadPoolExecutor to fetch multiple countries concurrently.
    """
    data360_indicator = "WB_WDI_" + indicator_dot_code.replace(".", "_")
    
    def fetch_country(code):
        """Fetch data for a single country with retry logic."""
        params = {
            "DATABASE_ID": "WB_WDI",
            "INDICATOR": data360_indicator,
            "REF_AREA": code,
            "TIME_PERIOD": str(year),
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{DATA360_BASE_URL}/data360/data", params=params, timeout=30)
                logger.info(f"[worldbank] {indicator_dot_code} / {code} -> HTTP {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                records = payload.get("value", [])
                if records:
                    val = records[0].get("OBS_VALUE")
                    result = float(val) if val is not None else 0
                    logger.info(f"[worldbank] {indicator_dot_code} / {code} -> value={result}")
                    return code, result
                else:
                    logger.warning(f"[worldbank] {indicator_dot_code} / {code} -> 0 records returned")
                    return code, 0
            except requests.exceptions.Timeout:
                logger.warning(f"[worldbank] TIMEOUT {indicator_dot_code} / {code} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"[worldbank] FAILED after {max_retries} retries: {indicator_dot_code} / {code}")
                    return code, 0
            except Exception as e:
                logger.error(f"[worldbank] ERROR {indicator_dot_code} / {code}: {e}")
                return code, 0
        
        return code, 0
    
    # Fetch all countries in parallel (max 5 concurrent requests)
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_country, code): code for code in country_codes}
        for future in as_completed(futures):
            code, value = future.result()
            results[code] = value
    
    return results

# ==========================================
# 2. FETCH AND MERGE DATA
# ==========================================

african_countries = ['ZAF', 'NGA', 'BWA', 'LSO', 'MOZ', 'RWA', 'SYC', 'UGA', 'ZMB', 'ZWE']

DATA_YEAR = 2019  # keep in sync with the year filtered in get_country_health_stats()

try:
    with st.spinner("Fetching Economic Data from World Bank Data360..."):
        gdp_data = get_worldbank_data('NY.GDP.MKTP.CD', african_countries, year=DATA_YEAR)
        pop_data = get_worldbank_data('SP.POP.TOTL', african_countries, year=DATA_YEAR)

    # Build the master DataFrame
    df = pd.DataFrame({
        'Country': african_countries,
        'GDP_USD': [gdp_data.get(c, 0) for c in african_countries],
        'Population': [pop_data.get(c, 0) for c in african_countries],
        'Hypertension_Rate': [35, 28, 32, 25, 40, 22, 30, 26, 38, 29]
    })

    logger.info(f"[main] Built dataframe with {len(df)} rows.")
    logger.info(f"[main] GDP zero-count: {(df['GDP_USD'] == 0).sum()}, "
                f"Population zero-count: {(df['Population'] == 0).sum()}")

except Exception as e:
    # This is the key change: instead of the page just going blank, any
    # failure in the data pipeline now renders as a visible error with the
    # full traceback right on the page, plus a full log entry.
    logger.exception("[main] Data pipeline failed.")
    st.error("The data pipeline failed - see details below and check the terminal logs.")
    st.exception(e)
    st.stop()

# ==========================================
# 4. DISPLAY
# ==========================================
# Nothing was being rendered after the dataframe was built, which is the
# most likely reason the app looked blank regardless of any data errors.

st.subheader("Master Data Table")
st.dataframe(df, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Hypertension Rate by Country")
    fig_hypertension = px.bar(df, x='Country', y='Hypertension_Rate',
                               title=f"Hypertension Rate ({DATA_YEAR})")
    st.plotly_chart(fig_hypertension, use_container_width=True)

with col2:
    st.subheader("GDP vs. Hypertension Rate")
    fig_scatter = px.scatter(df, x='GDP_USD', y='Hypertension_Rate', size='Population',
                              color='Country', hover_name='Country',
                              title="GDP vs. Hypertension Rate")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# 5. COUNTRY STATS SECTION
# ==========================================

st.divider()
st.subheader("Country Statistics")

# Fetch live exchange rates
country_currencies = get_live_exchange_rates()

# Create a mapping of country codes to full names
country_names = {
    'ZAF': 'South Africa',
    'NGA': 'Nigeria',
    'BWA': 'Botswana',
    'LSO': 'Lesotho',
    'MOZ': 'Mozambique',
    'RWA': 'Rwanda',
    'SYC': 'Seychelles',
    'UGA': 'Uganda',
    'ZMB': 'Zambia',
    'ZWE': 'Zimbabwe'
}

# Country selector
selected_country_code = st.selectbox(
    "Select a country to view detailed statistics:",
    options=african_countries,
    format_func=lambda code: country_names.get(code, code)
)

# Get data for selected country
if selected_country_code:
    country_row = df[df['Country'] == selected_country_code].iloc[0]
    country_full_name = country_names.get(selected_country_code, selected_country_code)
    currency_code, currency_symbol, exchange_rate = country_currencies.get(selected_country_code, ('USD', '$', 1.0))
    
    # Display stats in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Country", country_full_name)
    
    with col2:
        st.metric("Population", f"{country_row['Population']:,.0f}")
    
    with col3:
        st.metric("Hypertension Rate", f"{country_row['Hypertension_Rate']:.1f}%")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        gdp_billions_usd = country_row['GDP_USD'] / 1_000_000_000
        gdp_native = gdp_billions_usd * exchange_rate
        st.metric(f"GDP ({currency_code})", f"{currency_symbol}{gdp_native:.2f}B")
    
    with col5:
        if country_row['Population'] > 0:
            gdp_per_capita_usd = country_row['GDP_USD'] / country_row['Population']
            gdp_per_capita_native = gdp_per_capita_usd * exchange_rate
            st.metric(f"GDP per Capita ({currency_code})", f"{currency_symbol}{gdp_per_capita_native:,.0f}")
        else:
            st.metric(f"GDP per Capita ({currency_code})", "N/A")
    
    with col6:
        # Calculate health burden score (higher = worse)
        pop_millions = country_row['Population'] / 1_000_000
        health_burden = (country_row['Hypertension_Rate'] / 100) * pop_millions
        st.metric("Estimated Hypertension Cases (millions)", f"{health_burden:.2f}M")
    
    # Show exchange rate info
    with st.expander("Exchange Rate Information"):
        rate_source = "Live Rate" if HAS_ALLRATES else "Fallback (2019 Average)"
        st.info(f"**{currency_code}** ({currency_symbol}) - {rate_source}: **1 USD = {exchange_rate:.4f} {currency_code}**")

logger.info("[main] App rendered successfully.")