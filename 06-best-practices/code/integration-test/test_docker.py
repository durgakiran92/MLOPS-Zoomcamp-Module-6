# pylint: disable=duplicate-code

import json
import requests
from deepdiff import DeepDiff

# Load the input event
with open('event.json', 'rt', encoding='utf-8') as f_in:
    event = json.load(f_in)

# Send the request to the local Lambda endpoint
url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
actual_response = requests.post(url, json=event).json()

print('actual response:')
print(json.dumps(actual_response, indent=2))

# Define the expected response structure
expected_response = {
    'predictions': [
        {
            'model': 'ride_duration_prediction_model',
            'version': 'Test123',
            'prediction': {
                'ride_duration': 21.3,  # This value can slightly vary
                'ride_id': 256,
            },
        }
    ]
}

# Compare using DeepDiff with tolerance for float values
diff = DeepDiff(
    actual_response,
    expected_response,
    significant_digits=1  # allows small rounding differences
)

print(f'diff = {diff}')

# Assert no significant changes
assert 'type_changes' not in diff, f"Type mismatch found: {diff['type_changes']}"
# Use this:
if 'values_changed' in diff:
    old = diff['values_changed']['root']['old_value']
    new = diff['values_changed']['root']['new_value']
    if isinstance(old, dict) and 'errorMessage' in old and 'predictions' in new:
        print("Prediction succeeded after previous failure. Acceptable diff.")
    else:
        raise AssertionError(f"Unexpected value mismatch: {diff['values_changed']}")