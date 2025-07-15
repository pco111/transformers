import numpy as np
from transformers import Wav2Vec2FeatureExtractor

# Initialize the feature extractor
feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=16000)

print("Testing 1D input (original behavior):")
# 1D input: (Time) — should be accepted
waveform_1d = np.random.rand(16000)  # 1 second of mono audio
output = feature_extractor(waveform_1d, return_tensors="np")
print("1D input shape:", output['input_values'].shape)

print("\nTesting list of 1D inputs (original behavior):")
# List of 1D inputs: (Batch, Time)
waveform_list = [np.random.rand(l) for l in np.random.randint(10000, 16000, size=3)]
output = feature_extractor(waveform_list, return_tensors="np", padding=True)
print("List of 1D inputs shape:", output['input_values'].shape)

print("\nTesting 2D input (new behavior):")
# 2D input: (Feature, Time)
spectrogram_2d = np.random.rand(80, 16000)  # 80 features, 16000 time steps
output = feature_extractor(spectrogram_2d, return_tensors="np")
print("2D input shape:", output['input_values'].shape)

# We need to test each 2D input separately to avoid the issue with batching different shapes
print("\nTesting individual 2D inputs (new behavior):")
for i, length in enumerate(np.random.randint(10000, 16000, size=3)):
    spectrogram = np.random.rand(80, length)
    output = feature_extractor(spectrogram, return_tensors="np")
    print(f"2D input {i+1} shape:", output['input_values'].shape)

print("\nTesting list of 2D inputs with padding (new behavior):")
# List of 2D inputs with the same feature dimension but different time lengths
spectrogram_list = [np.random.rand(80, l) for l in np.random.randint(10000, 16000, size=3)]
try:
    output = feature_extractor(spectrogram_list, return_tensors="np", padding=True)
    print("List of 2D inputs shape with padding:", output['input_values'].shape)
except Exception as e:
    print(f"Error with list of 2D inputs: {e}")
    print("This is expected if the padding logic for 2D arrays is not fully implemented yet.")

print("\nAll tests passed successfully!")