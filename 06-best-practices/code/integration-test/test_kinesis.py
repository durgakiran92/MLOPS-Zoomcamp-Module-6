# pylint: disable=duplicate-code

import os
import json
from pprint import pprint

import boto3
from deepdiff import DeepDiff

kinesis_endpoint = os.getenv('KINESIS_ENDPOINT_URL', "http://localhost:4566")
kinesis_client = boto3.client('kinesis', endpoint_url=kinesis_endpoint)

stream_name = os.getenv('PREDICTIONS_STREAM_NAME', 'ride_predictions')
shard_id = 'shardId-000000000000'

# Get shard iterator
shard_iterator_response = kinesis_client.get_shard_iterator(
    StreamName=stream_name,
    ShardId=shard_id,
    ShardIteratorType='TRIM_HORIZON',
)

shard_iterator_id = shard_iterator_response['ShardIterator']

# Read records from Kinesis
records_response = kinesis_client.get_records(
    ShardIterator=shard_iterator_id,
    Limit=1,
)

records = records_response['Records']

print("\nRaw Record from Kinesis:")
pprint(records)

assert len(records) == 1

# Decode and load JSON
raw_data = records[0]['Data']
decoded_data = raw_data.decode("utf-8")
actual_record = json.loads(decoded_data)

print("\nDecoded and Parsed Record:")
print(json.dumps(actual_record, indent=2))  # Pretty-print JSON

# Expected format
expected_record = {
    'model': 'ride_duration_prediction_model',
    'version': 'Test123',
    'prediction': {
        'ride_duration': 21.3,
        'ride_id': 256,
    },
}

# Compare actual vs expected
diff = DeepDiff(actual_record, expected_record, significant_digits=1)
print(f"\ndiff={diff}")

assert 'values_changed' not in diff
assert 'type_changes' not in diff

print("\n✅ All good!")
