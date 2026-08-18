import requests
import pandas as pd

# --- FIXED: World Bank API (GDP and Population) ---
def get_worldbank_data(indicator, country_codes):
    results = {}
    # The API sometimes needs us to request one country at a time to be safe
    for code in country_codes:
        url = f"https://api.worldbank.org/v2/country/{code}/indicator/{indicator}?format=json&per_page=1"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            # Sometimes the data is nested deep, this safely grabs the latest value
            try:
                val = data[1][0]['value']
                if val is not None:
                    results[code] = float(val)
                else:
                    results[code] = 0
            except:
                results[code] = 0
        else:
            results[code] = 0
    return results

# List of African ISO-3 codes (Real codes used by the World Bank)
african_countries = ['ZAF', 'NGA', 'KEN', 'EGY', 'ETH', 'GHA', 'TZA', 'UGA', 'AGO', 'MOZ']

print("Fetching GDP and Population...")
# NY.GDP.MKTP.CD is the code for GDP in current USD
# SP.POP.TOTL is the code for Total Population
gdp_data = get_worldbank_data('NY.GDP.MKTP.CD', african_countries)
pop_data = get_worldbank_data('SP.POP.TOTL', african_countries)

# --- Combine them into one Master DataFrame ---
master_df = pd.DataFrame({
    'Country': african_countries,
    'GDP_USD': [gdp_data.get(c, 0) for c in african_countries],
    'Population': [pop_data.get(c, 0) for c in african_countries],
    # WHO data is complex, we will keep the placeholder for now until we tackle the WHO API
    'CVD_Mortality_Rate': [120, 200, 150, 85, 110, 140, 130, 90, 160, 180] 
})

print("\n--- Data Preview ---")
print(master_df)