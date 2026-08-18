import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import json
import os
import sys
import logging
import time

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

@st.cache_data
def get_worldbank_data(indicator_dot_code, country_codes, year=2019):
    """
    Fetch indicator values from the World Bank Data360 API
    (https://data360.worldbank.org/en/api), which replaces the legacy
    api.worldbank.org endpoint.

    indicator_dot_code: WDI-style code, e.g. 'NY.GDP.MKTP.CD'
    year: TIME_PERIOD to request, kept aligned with the 2019 health data
          so GDP/population/CVD are all comparing the same year.
    """
    # Data360 indicator IDs are WDI codes with dots replaced by underscores,
    # prefixed with the database ID (WB_WDI).
    data360_indicator = "WB_WDI_" + indicator_dot_code.replace(".", "_")

    results = {}
    for code in country_codes:
        params = {
            "DATABASE_ID": "WB_WDI",
            "INDICATOR": data360_indicator,
            "REF_AREA": code,
            "TIME_PERIOD": str(year),
        }
        
        # Retry logic with exponential backoff
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{DATA360_BASE_URL}/data360/data", params=params, timeout=30)
                logger.info(f"[worldbank] {indicator_dot_code} / {code} -> HTTP {response.status_code} | url={response.url}")
                response.raise_for_status()
                payload = response.json()
                records = payload.get("value", [])
                if records:
                    val = records[0].get("OBS_VALUE")
                    results[code] = float(val) if val is not None else 0
                    logger.info(f"[worldbank] {indicator_dot_code} / {code} -> value={results[code]}")
                else:
                    results[code] = 0
                    logger.warning(f"[worldbank] {indicator_dot_code} / {code} -> 0 records returned for TIME_PERIOD={year}")
                break  # Success, exit retry loop
            except requests.exceptions.Timeout:
                logger.warning(f"[worldbank] TIMEOUT fetching {indicator_dot_code} for {code} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    time.sleep(wait_time)
                else:
                    logger.error(f"[worldbank] FAILED after {max_retries} retries for {indicator_dot_code} / {code}")
                    results[code] = 0
            except Exception as e:
                logger.error(f"[worldbank] FAILED fetching {indicator_dot_code} for {code}: {e}")
                results[code] = 0
                break  # Don't retry on non-timeout errors
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

logger.info("[main] App rendered successfully.")