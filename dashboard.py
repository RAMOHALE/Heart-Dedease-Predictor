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
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
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
# 1. PATIENT DATA MANAGEMENT
# ==========================================

PATIENTS_FILE = "patients_data.json"

def load_patients():
    """Load patient data from JSON file."""
    if os.path.exists(PATIENTS_FILE):
        try:
            with open(PATIENTS_FILE, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded patient data from {PATIENTS_FILE}")
                return data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from {PATIENTS_FILE}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load patient data: {e}")
            return {}
    else:
        logger.info(f"Patients file does not exist yet: {PATIENTS_FILE}")
        return {}

def save_patients(patients_data):
    """Save patient data to JSON file."""
    try:
        # Ensure the patients_data is JSON-serializable
        serialized_data = json.loads(json.dumps(patients_data, default=str))
        
        with open(PATIENTS_FILE, 'w') as f:
            json.dump(serialized_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
        
        logger.info(f"Patient data saved successfully. File size: {os.path.getsize(PATIENTS_FILE)} bytes")
        return True
    except Exception as e:
        logger.error(f"Failed to save patient data: {e}", exc_info=True)
        return False

def add_prediction_to_history(username, prediction_data):
    """Add a prediction to patient's history."""
    try:
        logger.info(f"Adding prediction for user: {username}")
        patients = load_patients()
        if username in patients:
            if 'history' not in patients[username]:
                patients[username]['history'] = []
            prediction_data['timestamp'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            patients[username]['history'].append(prediction_data)
            logger.info(f"Prediction added to memory. Data: {prediction_data}")
            success = save_patients(patients)
            logger.info(f"Prediction saved for {username}. Total predictions: {len(patients[username]['history'])}. Save success: {success}")
            return success
        else:
            logger.error(f"Username {username} not found in patients database. Available users: {list(patients.keys())}")
            return False
    except Exception as e:
        logger.error(f"Error saving prediction: {e}", exc_info=True)
        return False

def create_history_pdf(username, patient, history):
    """Create a PDF containing every saved prediction for a patient."""
    pdf_buffer = BytesIO()
    patient_name = f"{patient.get('first_name', '')} {patient.get('last_name', '')}".strip()

    with PdfPages(pdf_buffer) as pdf:
        for page_start in range(0, len(history), 18):
            page_records = history[page_start:page_start + 18]
            figure, axis = plt.subplots(figsize=(11.69, 8.27))
            axis.axis("off")
            axis.set_title(
                f"Heart Disease Checkup History\n{patient_name} ({username})",
                fontsize=16,
                fontweight="bold",
                pad=18,
            )

            table_rows = []
            for record_number, record in enumerate(page_records, page_start + 1):
                table_rows.append([
                    str(record_number),
                    str(record.get("timestamp", "N/A")),
                    f"{record.get('patient_name', '')} {record.get('patient_surname', '')}".strip(),
                    str(record.get("age", "N/A")),
                    f"{float(record.get('risk_percentage', 0)):.1f}%",
                    str(record.get("risk_level", "N/A")),
                    str(record.get("model_used", "N/A")),
                    str(record.get("prediction", "N/A")),
                ])

            axis.table(
                cellText=table_rows,
                colLabels=["#", "Timestamp", "Patient", "Age", "Risk", "Level", "Model", "Result"],
                colWidths=[0.04, 0.16, 0.18, 0.06, 0.08, 0.10, 0.18, 0.12],
                cellLoc="left",
                loc="upper center",
                bbox=[0.02, 0.08, 0.96, 0.78],
            )
            axis.text(
                0.02,
                0.03,
                f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}    |    Page {page_start // 18 + 1}",
                transform=axis.transAxes,
                fontsize=8,
            )
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)

    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# ==========================================
# 2. MODEL TRAINING FUNCTIONS
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

@st.cache_data(ttl=3600)
def get_worldbank_series(indicator_dot_code, country_codes, start_year=2019, end_year=2026):
    """Fetch annual World Bank Data360 values for a group of countries."""
    data360_indicator = "WB_WDI_" + indicator_dot_code.replace(".", "_")
    
    def fetch_country(code):
        params = {
            "DATABASE_ID": "WB_WDI",
            "INDICATOR": data360_indicator,
            "REF_AREA": code,
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{DATA360_BASE_URL}/data360/data", params=params, timeout=30)
                logger.info(f"[worldbank] {indicator_dot_code} / {code} -> HTTP {response.status_code}")
                response.raise_for_status()
                payload = response.json()
                records = payload.get("value", [])
                values = {}
                for record in records:
                    try:
                        record_year = int(str(record.get("TIME_PERIOD", ""))[:4])
                        value = float(record.get("OBS_VALUE"))
                        if start_year <= record_year <= end_year:
                            values[record_year] = value
                    except (TypeError, ValueError):
                        continue
                logger.info(f"[worldbank] {indicator_dot_code} / {code} -> {len(values)} annual values")
                return code, values
            except requests.exceptions.Timeout:
                logger.warning(f"[worldbank] TIMEOUT {indicator_dot_code} / {code} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"[worldbank] FAILED after {max_retries} retries: {indicator_dot_code} / {code}")
                    return code, {}
            except Exception as e:
                logger.error(f"[worldbank] ERROR {indicator_dot_code} / {code}: {e}")
                return code, {}
        
        return code, {}
    
    # Fetch all countries in parallel (max 5 concurrent requests)
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_country, code): code for code in country_codes}
        for future in as_completed(futures):
            code, values = future.result()
            results[code] = values
    
    return results

# ==========================================
# 2. FETCH AND MERGE DATA
# ==========================================

african_countries = ['ZAF', 'NGA', 'BWA', 'LSO', 'MOZ', 'RWA', 'SYC', 'UGA', 'ZMB', 'ZWE']

DATA_YEAR = 2019
END_YEAR = 2026
YEARS = list(range(DATA_YEAR, END_YEAR + 1))

try:
    with st.spinner("Fetching Economic Data from World Bank Data360..."):
        gdp_series = get_worldbank_series('NY.GDP.MKTP.CD', african_countries, DATA_YEAR, END_YEAR)
        pop_series = get_worldbank_series('SP.POP.TOTL', african_countries, DATA_YEAR, END_YEAR)

    economic_history = pd.DataFrame([
        {
            'Country': country,
            'Year': year,
            'GDP_USD': gdp_series.get(country, {}).get(year, np.nan),
            'Population': pop_series.get(country, {}).get(year, np.nan),
        }
        for country in african_countries
        for year in YEARS
    ])

    current_year_data = economic_history.dropna(subset=['GDP_USD', 'Population']).sort_values('Year').groupby('Country').tail(1).set_index('Country')
    missing_population = current_year_data['Population'].isna().sum()
    if missing_population:
        logger.warning(f"[main] Population missing for {missing_population} countries in current data")

    # Build the master DataFrame
    df = pd.DataFrame({
        'Country': african_countries,
        'Data Year': [current_year_data.loc[c, 'Year'] if c in current_year_data.index else np.nan for c in african_countries],
        'GDP_USD': [current_year_data.loc[c, 'GDP_USD'] if c in current_year_data.index else np.nan for c in african_countries],
        'Population': [current_year_data.loc[c, 'Population'] if c in current_year_data.index else np.nan for c in african_countries],
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
# 6. PATIENT LOGIN SYSTEM
# ==========================================

# Initialize session state for authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.patient_name = None
    st.session_state.patient_surname = None
    st.session_state.current_prediction = None

# Authentication sidebar
st.sidebar.title("Patient Portal")

if not st.session_state.logged_in:
    auth_mode = st.sidebar.radio("Choose:", ["Login", "Register"])
    
    if auth_mode == "Register":
        st.sidebar.subheader("Create New Account")
        new_username = st.sidebar.text_input("Username", key="reg_username")
        new_password = st.sidebar.text_input("Password", type="password", key="reg_password")
        first_name = st.sidebar.text_input("First Name", key="reg_first")
        last_name = st.sidebar.text_input("Last Name", key="reg_last")
        
        if st.sidebar.button("Register"):
            patients = load_patients()
            if new_username in patients:
                st.sidebar.error("Username already exists!")
            elif new_username and new_password and first_name and last_name:
                patients[new_username] = {
                    'password': new_password,
                    'first_name': first_name,
                    'last_name': last_name,
                    'history': []
                }
                if save_patients(patients):
                    st.sidebar.success("Account created! Please login.")
                    logger.info(f"New account created: {new_username}")
            else:
                st.sidebar.error("Please fill all fields!")
    
    else:  # Login
        st.sidebar.subheader("Login")
        username = st.sidebar.text_input("Username", key="login_username")
        password = st.sidebar.text_input("Password", type="password", key="login_password")
        
        if st.sidebar.button("Login"):
            patients = load_patients()
            if username in patients and patients[username]['password'] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.patient_name = patients[username]['first_name']
                st.session_state.patient_surname = patients[username]['last_name']
                logger.info(f"User logged in successfully: {username}")
                st.sidebar.success(f"Welcome, {patients[username]['first_name']}!")
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password!")
else:
    st.sidebar.write(f"**Logged in as:** {st.session_state.patient_name} {st.session_state.patient_surname}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.patient_name = None
        st.session_state.patient_surname = None
        st.sidebar.success("Logged out!")
        logger.info(f"User logged out: {st.session_state.username}")
        st.rerun()

# ==========================================
# 7. CREATE TABS FOR DIFFERENT SECTIONS
# ==========================================

tab_dashboard, tab_predictor, tab_history = st.tabs(["📊 Economic Dashboard", "🏥 Heart Disease Predictor", "📋 My Predictions"])

# The selected st.tabs value is browser-side, so use the active tab's ARIA
# state to keep authentication controls out of the economic dashboard view.
st.markdown(
    """
    <style>
    body:has(.stTabs [data-baseweb="tab-list"] button:nth-child(1)[aria-selected="true"]) [data-testid="stSidebar"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
        current_data_year = int(country_row['Data Year']) if pd.notna(country_row['Data Year']) else None
        currency_code, currency_symbol, exchange_rate = country_currencies.get(selected_country_code, ('USD', '$', 1.0))
        
        st.divider()
        st.subheader(f"{country_full_name} - Detailed Statistics")
        if current_data_year:
            st.caption(f"Current summary data: World Bank {current_data_year}. Graphs: {DATA_YEAR}-{END_YEAR}.")
        
        # Display stats in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Country", country_full_name)
        
        with col2:
            population_label = f"Population ({current_data_year})" if current_data_year else "Population"
            st.metric(population_label, f"{country_row['Population']:,.0f}")
        
        with col3:
            st.metric("Hypertension Rate", f"{country_row['Hypertension_Rate']:.1f}%")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            gdp_billions_usd = country_row['GDP_USD'] / 1_000_000_000
            gdp_native = gdp_billions_usd * exchange_rate
            gdp_label = f"GDP ({currency_code}, {current_data_year})" if current_data_year else f"GDP ({currency_code})"
            st.metric(gdp_label, f"{currency_symbol}{gdp_native:.2f}B")
        
        with col5:
            if country_row['Population'] > 0:
                gdp_per_capita_usd = country_row['GDP_USD'] / country_row['Population']
                gdp_per_capita_native = gdp_per_capita_usd * exchange_rate
                per_capita_label = f"GDP per Capita ({currency_code}, {current_data_year})" if current_data_year else f"GDP per Capita ({currency_code})"
                st.metric(per_capita_label, f"{currency_symbol}{gdp_per_capita_native:,.0f}")
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
        # RISK FACTORS PIE CHART
        # ==========================================
        
        st.divider()
        st.subheader("Heart Disease Risk Factors Profile")
        
        # Simulated risk factors based on country health profile
        # These are estimated distributions based on health indicators
        risk_factors_distribution = {
            'Hypertension': country_row['Hypertension_Rate'] * 0.35,
            'Cholesterol': (100 - country_row['Hypertension_Rate']) * 0.25,
            'Obesity': country_row['Hypertension_Rate'] * 0.20,
            'Diabetes': country_row['Hypertension_Rate'] * 0.12,
            'Smoking/Lifestyle': (100 - country_row['Hypertension_Rate']) * 0.08
        }
        
        risk_df = pd.DataFrame({
            'Risk Factor': list(risk_factors_distribution.keys()),
            'Prevalence': list(risk_factors_distribution.values())
        })
        
        fig_pie = px.pie(risk_df, values='Prevalence', names='Risk Factor',
                         title=f"Estimated Heart Disease Risk Factors in {country_full_name}",
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # ==========================================
        # DISPLAY - FILTERED BY SELECTED COUNTRY
        # ==========================================
        
        st.divider()
        st.subheader(f"{country_full_name} - Country Data")
        
        # Show the selected country's annual World Bank series
        selected_df = df[df['Country'] == selected_country_code]
        selected_history = economic_history[economic_history['Country'] == selected_country_code].copy()
        selected_history['GDP (Billions USD)'] = selected_history['GDP_USD'] / 1_000_000_000
        selected_history['Population (Millions)'] = selected_history['Population'] / 1_000_000
        st.dataframe(selected_history[['Year', 'GDP (Billions USD)', 'Population (Millions)']], use_container_width=True, hide_index=True)
        
        st.subheader(f"{country_full_name} - GDP and Population Trends")
        st.caption("Annual World Bank data. GDP is shown in billions of USD and population in millions.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Population Trend")
            fig_population = px.line(
                selected_history,
                x='Year',
                y='Population (Millions)',
                markers=True,
                title=f"Population ({DATA_YEAR}-{END_YEAR})",
            )
            fig_population.update_layout(xaxis=dict(dtick=1))
            st.plotly_chart(fig_population, use_container_width=True)
        
        with col2:
            st.subheader("GDP Trend")
            gdp_history = selected_history.dropna(subset=['GDP (Billions USD)'])
            if gdp_history.empty:
                st.error("No GDP data was returned by the World Bank for this country.")
            else:
                latest_gdp_year = int(gdp_history['Year'].max())
                st.caption(f"World Bank GDP data available through {latest_gdp_year}.")
            fig_gdp = px.line(
                gdp_history,
                x='Year',
                y='GDP (Billions USD)',
                markers=True,
                title=f"GDP ({DATA_YEAR}-{END_YEAR})",
            )
            fig_gdp.update_layout(xaxis=dict(dtick=1))
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
            st.write(f"GDP ({current_data_year}): ${gdp_billions_usd:.2f}B USD")
        
        with summary_col3:
            st.write("**Demographic Data**")
            st.write(f"Population ({current_data_year}): {country_row['Population']/1_000_000:.2f}M")

# ==========================================
# TAB 2: HEART DISEASE PREDICTOR
# ==========================================

with tab_predictor:
    st.subheader("Heart Disease Risk Predictor")
    
    if not st.session_state.logged_in:
        st.info("Please login first to use the prediction tool and save your history.")
        st.write("Go to the sidebar to login or create an account.")
    else:
        st.write(f"Welcome, {st.session_state.patient_name}! Enter patient information below to predict heart disease risk.")
        
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
            
            # Patient name and surname
            col_name1, col_name2 = st.columns(2)
            with col_name1:
                patient_first_name = st.text_input("First Name", value=st.session_state.patient_name or "")
            with col_name2:
                patient_last_name = st.text_input("Last Name", value=st.session_state.patient_surname or "")
            
            st.divider()
            
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
            if st.button("Predict Risk", type="primary", use_container_width=True, key="predict_btn"):
                input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                        thalach, exang, oldpeak, slope, ca, thal]])

                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                risk_percentage = probability[1] * 100

                # Store prediction in session state for later saving
                st.session_state.current_prediction = {
                    'patient_name': patient_first_name,
                    'patient_surname': patient_last_name,
                    'age': age,
                    'risk_percentage': float(risk_percentage),
                    'risk_level': 'LOW' if risk_percentage < 30 else ('MODERATE' if risk_percentage < 60 else 'HIGH'),
                    'model_used': selected_model_name,
                    'prediction': 'High Risk' if prediction == 1 else 'Low Risk'
                }
                save_success = add_prediction_to_history(
                    st.session_state.username,
                    st.session_state.current_prediction.copy(),
                )
                if save_success:
                    st.session_state.prediction_saved = True
                    logger.info(f"Prediction generated and saved for user: {st.session_state.username}")
                else:
                    st.session_state.prediction_saved = False
                    logger.error(f"Prediction generated but could not be saved for user: {st.session_state.username}")

            # Display prediction if available
            if st.session_state.current_prediction:
                pred = st.session_state.current_prediction
                risk_percentage = pred['risk_percentage']
                
                st.divider()
                st.subheader("Prediction Results")
                st.caption(f"Using: {pred['model_used']}")

                if risk_percentage > 50:
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
                if st.session_state.get('prediction_saved'):
                    st.success("✅ Checkup saved automatically. Open 'My Predictions' to view the full history.")
                else:
                    st.error("❌ This checkup could not be saved. Please try again.")

# ==========================================
# TAB 3: PREDICTION HISTORY
# ==========================================

with tab_history:
    st.subheader("My Prediction History")
    
    if not st.session_state.logged_in:
        st.info("Please login to view your prediction history.")
    else:
        try:
            # Force reload from file (don't use cache)
            with open(PATIENTS_FILE, 'r') as f:
                patients = json.load(f)
            
            # Debug section
            with st.expander("🔧 Debug Information"):
                st.write(f"**Logged in as:** {st.session_state.username}")
                st.write(f"**Data file location:** {os.path.abspath(PATIENTS_FILE)}")
                st.write(f"**File exists:** {os.path.exists(PATIENTS_FILE)}")
                if os.path.exists(PATIENTS_FILE):
                    st.write(f"**File size:** {os.path.getsize(PATIENTS_FILE)} bytes")
                    st.write(f"**Last modified:** {pd.Timestamp.fromtimestamp(os.path.getmtime(PATIENTS_FILE))}")
                if st.button("🔄 Refresh data from file"):
                    st.rerun()
            
            if st.session_state.username not in patients:
                st.warning(f"⚠️ Patient account not found. (Username: {st.session_state.username})")
                st.write("Please make sure you're logged in with the correct account.")
            else:
                patient = patients[st.session_state.username]
                history = patient.get('history', [])
                
                if not history:
                    st.info("📝 No predictions saved yet. Go to the 'Heart Disease Predictor' tab to make your first prediction and save it!")
                    st.write("**Steps:**")
                    st.write("1. Fill in patient information")
                    st.write("2. Click 'Predict Risk' button")
                    st.write("3. Click 'Save This Prediction to My History' button")
                    st.write("4. Return to this tab to view your saved predictions")
                else:
                    st.success(f"✅ Total predictions saved: {len(history)}")
                    st.caption("Every saved checkup is listed below with the date and time it was recorded.")
                    st.divider()
                    
                    for idx, record in enumerate(reversed(history), 1):
                        with st.expander(f"📋 Prediction #{len(history) - idx + 1} - {record.get('timestamp', 'N/A')}"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.write(f"**Patient:** {record.get('patient_name', 'N/A')} {record.get('patient_surname', 'N/A')}")
                                st.write(f"**Age:** {record.get('age', 'N/A')}")
                            
                            with col2:
                                risk_pct = record.get('risk_percentage', 0)
                                st.write(f"**Risk Level:** {record.get('risk_level', 'N/A')}")
                                st.write(f"**Risk Percentage:** {risk_pct:.1f}%")
                            
                            with col3:
                                st.write(f"**Model Used:** {record.get('model_used', 'N/A')}")
                                st.write(f"**Prediction:** {record.get('prediction', 'N/A')}")
                    
                    st.divider()
                    history_df = pd.DataFrame(history)
                    csv_data = history_df.to_csv(index=False)
                    pdf_data = create_history_pdf(st.session_state.username, patient, history)
                    download_col1, download_col2 = st.columns(2)
                    with download_col1:
                        st.download_button(
                            label="Download History as PDF",
                            data=pdf_data,
                            file_name=f"{st.session_state.username}_prediction_history.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    with download_col2:
                        st.download_button(
                            label="Download History as CSV",
                            data=csv_data,
                            file_name=f"{st.session_state.username}_prediction_history.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
        except Exception as e:
            st.error(f"❌ Error loading history: {e}")
            logger.error(f"Error in history tab: {e}")
            st.write("Please refresh the page or try again later.")

logger.info("[main] App rendered successfully.")