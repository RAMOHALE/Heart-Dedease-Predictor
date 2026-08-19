"""Test script to verify prediction persistence works correctly."""
import json
import os
import pandas as pd
from datetime import datetime

PATIENTS_FILE = "patients_data.json"

# Load current patients data
with open(PATIENTS_FILE, 'r') as f:
    patients = json.load(f)

# Check the structure
print(f"Loaded patients: {list(patients.keys())}")
print()

for username, user_data in patients.items():
    print(f"\n{'='*60}")
    print(f"User: {username}")
    print(f"Name: {user_data.get('first_name')} {user_data.get('last_name')}")
    print(f"History count: {len(user_data.get('history', []))}")
    
    if user_data.get('history'):
        print("\nHistory entries:")
        for i, record in enumerate(user_data['history'], 1):
            print(f"\n  Prediction #{i}:")
            print(f"    Timestamp: {record.get('timestamp')}")
            print(f"    Patient: {record.get('patient_name')} {record.get('patient_surname')}")
            print(f"    Age: {record.get('age')}")
            print(f"    Risk: {record.get('risk_percentage')}% ({record.get('risk_level')})")
            print(f"    Model: {record.get('model_used')}")
            print(f"    Prediction: {record.get('prediction')}")
    else:
        print("  No history entries yet.")

print(f"\n{'='*60}")

# Test adding a prediction to KAT123456
print("\n\nTesting prediction save...")
test_prediction = {
    'patient_name': 'YOLA',
    'patient_surname': 'GCOLOTELA',
    'age': 45,
    'risk_percentage': 35.5,
    'risk_level': 'MODERATE',
    'model_used': 'Random Forest',
    'prediction': 'Low Risk',
    'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
}

if 'KAT123456' in patients:
    patients['KAT123456']['history'].append(test_prediction)
    print(f"Added test prediction. History now has {len(patients['KAT123456']['history'])} entries.")
    
    # Save to file
    with open(PATIENTS_FILE, 'w') as f:
        json.dump(patients, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    
    print("Saved to file successfully.")
    
    # Verify by reading back
    with open(PATIENTS_FILE, 'r') as f:
        verified_patients = json.load(f)
    
    verified_count = len(verified_patients['KAT123456']['history'])
    print(f"Verified: KAT123456 now has {verified_count} predictions in file.")
    
    if verified_count == len(patients['KAT123456']['history']):
        print("✅ Persistence test PASSED!")
    else:
        print("❌ Persistence test FAILED!")
else:
    print("❌ User KAT123456 not found!")
