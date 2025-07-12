#!/usr/bin/env python3

import json
from pydantic import BaseModel
from flujo.testing.utils import DummyRemoteBackend
from flujo.utils.serialization import safe_serialize

class ComplexNested(BaseModel):
    name: str
    metadata: dict[str, str]

class DeepContainer(BaseModel):
    level1: ComplexNested
    level2: list[ComplexNested]
    level3: dict[str, ComplexNested]

# Create the test payload
payload = DeepContainer(
    level1=ComplexNested(name="test", metadata={"key": "value"}),
    level2=[
        ComplexNested(name="item1", metadata={"id": "1"}),
        ComplexNested(name="item2", metadata={"id": "2"}),
    ],
    level3={
        "first": ComplexNested(name="first", metadata={"type": "primary"}),
        "second": ComplexNested(name="second", metadata={"type": "secondary"}),
    }
)

print("Original payload:")
print(f"Type: {type(payload)}")
print(f"Data: {payload.model_dump()}")

# Simulate the DummyRemoteBackend serialization process
request_data = {
    "input_data": payload,
    "context": None,
    "resources": None,
    "context_model_defined": False,
    "usage_limits": None,
    "stream": False,
}

print(f"\nRequest data:")
print(f"input_data type: {type(request_data['input_data'])}")
print(f"input_data: {request_data['input_data'].model_dump()}")

# Test safe_serialize directly
serialized = safe_serialize(request_data)
print(f"\nSafe serialized:")
print(f"Type: {type(serialized)}")
print(f"Data: {serialized}")

# Test JSON roundtrip
json_str = json.dumps(serialized)
print(f"\nJSON string:")
print(json_str)

# Test deserialization
data = json.loads(json_str)
print(f"\nDeserialized:")
print(f"Type: {type(data)}")
print(f"Data: {data}")

# Check the input_data field specifically
input_data = data.get('input_data', {})
print(f"\nInput data after roundtrip:")
print(f"Type: {type(input_data)}")
print(f"Data: {input_data}")

if isinstance(input_data, dict):
    level2 = input_data.get('level2')
    level3 = input_data.get('level3')
    print(f"\nLevel2 field:")
    print(f"Type: {type(level2)}")
    print(f"Data: {level2}")

    print(f"\nLevel3 field:")
    print(f"Type: {type(level3)}")
    print(f"Data: {level3}")
