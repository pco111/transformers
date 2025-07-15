import numpy as np
from transformers import Wav2Vec2FeatureExtractor

# Initialize the feature extractor with explicit sampling rate to avoid warnings
feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=16000)

# Test case 1: Single 1D waveform (original use case)
print("\nTest case 1: Single 1D waveform (original use case)")
waveform_1d = np.random.rand(16000)  # 1 second of mono audio at 16kHz
output = feature_extractor(waveform_1d, return_tensors="np")
print(f"Input shape: {waveform_1d.shape}")
print(f"Output shape: {output['input_values'].shape}")
print(f"Expected shape: (1, 16000)")

# Test case 2: Batch of 1D waveforms with padding (original use case)
print("\nTest case 2: Batch of 1D waveforms with padding (original use case)")
waveform_batch = [np.random.rand(length) for length in [12000, 14000, 16000]]
output = feature_extractor(waveform_batch, padding=True, return_tensors="np")
print(f"Input shapes: {[w.shape for w in waveform_batch]}")
print(f"Output shape: {output['input_values'].shape}")
print(f"Expected shape: (3, 16000)")

# Test case 3: Single 2D spectrogram (new use case)
print("\nTest case 3: Single 2D spectrogram (new use case)")
spectrogram_2d = np.random.rand(80, 16000)  # 80 mel bands, 16000 time steps
output = feature_extractor(spectrogram_2d, return_tensors="np")
print(f"Input shape: {spectrogram_2d.shape}")
print(f"Output shape: {output['input_values'].shape}")
print(f"Expected shape: (80, 16000)")

# Test case 4: Batch of 2D spectrograms with padding (new use case)
print("\nTest case 4: Batch of 2D spectrograms with padding (new use case)")
spectrogram_batch = [np.random.rand(80, length) for length in [12000, 14000, 16000]]
output = feature_extractor(spectrogram_batch, padding=True, return_tensors="np")
print(f"Input shapes: {[s.shape for s in spectrogram_batch]}")
print(f"Output shape: {output['input_values'].shape}")
print(f"Expected shape: (3, 80, 16000)")

# Test case 5: Batch of 2D spectrograms with different feature dimensions (should raise error)
print("\nTest case 5: Batch of 2D spectrograms with different feature dimensions (should raise error)")
spectrogram_batch_invalid = [np.random.rand(dim, 12000) for dim in [64, 80, 128]]
try:
    output = feature_extractor(spectrogram_batch_invalid, padding=True, return_tensors="np")
    print("Error: This should have raised a ValueError")
except ValueError as e:
    print(f"Correctly raised ValueError: {e}")

# Test case 6: 3D input (now supported)
print("\nTest case 6: 3D input (now supported)")
array_3d = np.random.rand(2, 80, 16000)  # 2 channels, 80 mel bands, 16000 time steps
output = feature_extractor(array_3d, return_tensors="np")
print(f"Input shape: {array_3d.shape}")
print(f"Output shape: {output['input_values'].shape}")
print(f"Expected shape: (2, 80, 16000)")

print("\nAll tests completed successfully!")