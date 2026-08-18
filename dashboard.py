import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
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

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
# 1. MODEL TRAINING FUNCTIONS
# ==========================================

@st.cache_data
def load_heart_disease_data():
    """Load and prepare heart disease dataset for model training."""
    try:
        df = pd.read_csv("heart.csv", header=None)
        df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df = df.replace('?', 0)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.fillna(0)
        df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
        return df
    except Exception as e:
        logger.error(f"Failed to load heart disease data: {e}")
        return None

@st.cache_resource
def train_heart_disease_models(_df):
    """Train multiple models for heart disease prediction."""
    try:
        X = _df.drop("target", axis=1)
        y = _df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        }

        results = {}
        for name, m in models.items():
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[name] = {"model": m, "accuracy": acc}

        return results, X.columns.tolist()
    except Exception as e:
        logger.error(f"Failed to train models: {e}")
        return {}, []

# ==========================================
# 2. DATA FETCHING FUNCTIONS
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
# 4. LOAD DATA & MODELS
# ==========================================

# Load heart disease data and train models
heart_df = load_heart_disease_data()
if heart_df is not None:
    heart_models, feature_names = train_heart_disease_models(heart_df)
else:
    heart_models, feature_names = {}, []

# ==========================================
# 5. CREATE TABS FOR DIFFERENT SECTIONS
# ==========================================

tab_dashboard, tab_predictor = st.tabs(["📊 Economic Dashboard", "🏥 Heart Disease Predictor"])

# ==========================================
# TAB 1: ECONOMIC DASHBOARD
# ==========================================

with tab_dashboard:
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

    st.subheader("Select a Country")
    selected_country_code = st.selectbox(
        "Choose a country to view detailed statistics:",
        options=african_countries,
        format_func=lambda code: country_names.get(code, code),
        key="country_selector"
    )

    # Display selected country statistics
    if selected_country_code:
        country_row = df[df['Country'] == selected_country_code].iloc[0]
        country_full_name = country_names.get(selected_country_code, selected_country_code)
        currency_code, currency_symbol, exchange_rate = country_currencies.get(selected_country_code, ('USD', '$', 1.0))
        
        st.divider()
        st.subheader(f"{country_full_name} - Detailed Statistics")
        
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
        
        # ==========================================
        # DISPLAY - FILTERED BY SELECTED COUNTRY
        # ==========================================
        
        st.divider()
        st.subheader(f"{country_full_name} - Country Data")
        
        # Show only the selected country's data
        selected_df = df[df['Country'] == selected_country_code]
        st.dataframe(selected_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Hypertension Rate")
            fig_hypertension = px.bar(selected_df, x='Country', y='Hypertension_Rate',
                                       title=f"Hypertension Rate ({DATA_YEAR})")
            st.plotly_chart(fig_hypertension, use_container_width=True)
        
        with col2:
            st.subheader("Economic Data")
            # Create a simple comparison chart
            selected_df_display = selected_df.copy()
            selected_df_display['GDP (Billions USD)'] = selected_df_display['GDP_USD'] / 1_000_000_000
            fig_gdp = px.bar(selected_df_display, x='Country', y='GDP (Billions USD)',
                             title=f"GDP ({DATA_YEAR})")
            st.plotly_chart(fig_gdp, use_container_width=True)
        
        # Summary statistics section
        st.divider()
        st.subheader("Summary")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.write("**Key Health Metric**")
            st.write(f"Hypertension Rate: {country_row['Hypertension_Rate']:.1f}%")
        
        with summary_col2:
            st.write("**Economic Indicators**")
            gdp_billions_usd = country_row['GDP_USD'] / 1_000_000_000
            st.write(f"GDP: ${gdp_billions_usd:.2f}B USD")
        
        with summary_col3:
            st.write("**Demographic Data**")
            st.write(f"Population: {country_row['Population']/1_000_000:.2f}M")

# ==========================================
# TAB 2: HEART DISEASE PREDICTOR
# ==========================================

with tab_predictor:
    st.subheader("Heart Disease Risk Predictor")
    st.write("Enter patient information below to predict heart disease risk.")
    
    if not heart_models:
        st.error("Failed to load prediction models. Please check the logs.")
    else:
        # Model comparison in expander
        with st.expander("Model Performance"):
            comparison_data = {
                "Model": list(heart_models.keys()),
                "Accuracy": [f"{heart_models[k]['accuracy']:.1%}" for k in heart_models],
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, hide_index=True, use_container_width=True)
        
        # Model selector
        best_model_name = max(heart_models, key=lambda k: heart_models[k]["accuracy"])
        selected_model_name = st.selectbox(
            "Choose prediction model",
            options=list(heart_models.keys()),
            index=list(heart_models.keys()).index(best_model_name),
            key="model_selector"
        )
        
        model = heart_models[selected_model_name]["model"]
        accuracy = heart_models[selected_model_name]["accuracy"]
        
        st.metric("Selected Model Accuracy", f"{accuracy:.1%}")
        
        # Input form
        st.divider()
        st.subheader("Patient Information")
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=1, max_value=120, value=50)
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox(
                "Chest Pain Type",
                options=[0, 1, 2, 3],
                format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x]
            )
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=250, value=120)
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )
            restecg = st.selectbox(
                "Resting ECG Results",
                options=[0, 1, 2],
                format_func=lambda x: ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"][x]
            )

        with col2:
            thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
            exang = st.selectbox(
                "Exercise Induced Angina",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Yes"
            )
            oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            slope = st.selectbox(
                "Slope of Peak Exercise ST",
                options=[0, 1, 2],
                format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x]
            )
            ca = st.selectbox("Number of Major Vessels (0-3)", options=[0, 1, 2, 3])
            thal = st.selectbox(
                "Thalassemia",
                options=[0, 1, 2, 3],
                format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x]
            )

        # Prediction
        if st.button("Predict Risk", type="primary", use_container_width=True):
            input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                    thalach, exang, oldpeak, slope, ca, thal]])

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0]

            st.divider()
            st.subheader("Prediction Results")
            st.caption(f"Using: {selected_model_name}")

            risk_percentage = probability[1] * 100

            if prediction == 1:
                st.error(f"🚨 High Risk of Heart Disease ({risk_percentage:.1f}% probability)")
            else:
                st.success(f"✅ Low Risk of Heart Disease ({risk_percentage:.1f}% probability)")

            st.subheader("Risk Level Progress")
            st.progress(risk_percentage / 100)

            if risk_percentage < 30:
                st.write("**Risk Level: LOW** - Regular checkups recommended")
            elif risk_percentage < 60:
                st.write("**Risk Level: MODERATE** - Lifestyle modifications advised")
            else:
                st.write("**Risk Level: HIGH** - Consult a healthcare professional immediately")

            # Feature importance (only for tree-based models)
            if hasattr(model, "feature_importances_"):
                st.subheader("Top Risk Factors")
                importance = model.feature_importances_
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importance
                }).sort_values("Importance", ascending=False).head(5)

                st.bar_chart(importance_df.set_index("Feature"))
            else:
                st.subheader("Model Coefficients")
                coefficients = model.coef_[0]
                coef_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Weight": np.abs(coefficients)
                }).sort_values("Weight", ascending=False).head(5)

                st.bar_chart(coef_df.set_index("Feature"))

            st.info("⚠️ **Disclaimer**: This tool is for educational purposes only. Always consult a medical professional for diagnosis and treatment.")

logger.info("[main] App rendered successfully.")