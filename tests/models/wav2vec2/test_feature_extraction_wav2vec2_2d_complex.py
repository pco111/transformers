# Copyright 2023 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import unittest

import numpy as np

from transformers import Wav2Vec2FeatureExtractor
from transformers.testing_utils import require_torch


class Wav2Vec2FeatureExtraction2DComplexTest(unittest.TestCase):
    """More complex tests for 2D input support in Wav2Vec2FeatureExtractor"""

    def setUp(self):
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            return_attention_mask=True,
            do_normalize=True,
        )

    def test_mixed_1d_2d_inputs(self):
        """Test that a mix of 1D and 2D inputs raises an error"""
        # Create a mix of 1D and 2D inputs
        inputs = [
            np.random.rand(1000),  # 1D input
            np.random.rand(80, 1000),  # 2D input
        ]

        # Should raise ValueError due to mixed input dimensions
        with self.assertRaises(ValueError):
            self.feature_extractor(inputs, padding="longest", return_tensors="np")

    def test_2d_input_with_padding_truncation(self):
        """Test padding and truncation with 2D inputs"""
        # Create 2D inputs of different lengths
        inputs = [
            np.random.rand(80, 1000),
            np.random.rand(80, 1200),
            np.random.rand(80, 800),
        ]

        # Test padding to longest
        processed = self.feature_extractor(inputs, padding="longest", return_tensors="np")
        self.assertEqual(processed.input_values.shape, (3, 80, 1200))  # Should pad to longest (1200)

        # Test padding to max_length
        processed = self.feature_extractor(
            inputs, padding="max_length", max_length=1500, return_tensors="np"
        )
        self.assertEqual(processed.input_values.shape, (3, 80, 1500))  # Should pad to max_length (1500)

        # Test truncation
        processed = self.feature_extractor(
            inputs, padding="max_length", max_length=900, truncation=True, return_tensors="np"
        )
        self.assertEqual(processed.input_values.shape, (3, 80, 900))  # Should truncate to max_length (900)

    def test_2d_input_with_attention_mask(self):
        """Test attention mask generation with 2D inputs"""
        # Create 2D inputs of different lengths
        inputs = [
            np.random.rand(80, 1000),
            np.random.rand(80, 1200),
            np.random.rand(80, 800),
        ]

        # Process with attention mask
        processed = self.feature_extractor(
            inputs, padding="longest", return_tensors="np", return_attention_mask=True
        )

        # Check attention mask shape and values
        self.assertEqual(processed.attention_mask.shape, (3, 1200))  # (batch_size, max_length)

        # First sample: 1000 ones followed by 200 zeros
        self.assertTrue(np.all(processed.attention_mask[0, :1000] == 1))
        self.assertTrue(np.all(processed.attention_mask[0, 1000:] == 0))

        # Second sample: all ones (length 1200)
        self.assertTrue(np.all(processed.attention_mask[1, :] == 1))

        # Third sample: 800 ones followed by 400 zeros
        self.assertTrue(np.all(processed.attention_mask[2, :800] == 1))
        self.assertTrue(np.all(processed.attention_mask[2, 800:] == 0))

    def test_2d_input_normalization(self):
        """Test normalization of 2D inputs"""
        # Create a 2D input with known statistics
        # Each feature dimension will have a different mean and variance
        feature_dim = 3
        seq_len = 1000
        input_2d = np.zeros((feature_dim, seq_len))

        # First feature: mean=5, std=2
        input_2d[0, :] = np.random.normal(5, 2, seq_len)
        # Second feature: mean=-3, std=1.5
        input_2d[1, :] = np.random.normal(-3, 1.5, seq_len)
        # Third feature: mean=0, std=3
        input_2d[2, :] = np.random.normal(0, 3, seq_len)

        # Process with normalization
        processed = self.feature_extractor(input_2d, return_tensors="np", do_normalize=True)

        # Check that each feature dimension is normalized to zero mean and unit variance
        for i in range(feature_dim):
            normalized_feature = processed.input_values[0, i, :]
            self.assertTrue(abs(np.mean(normalized_feature)) < 0.1)  # Close to zero mean
            self.assertTrue(abs(np.std(normalized_feature) - 1.0) < 0.1)  # Close to unit variance

    @require_torch
    def test_2d_input_with_torch(self):
        """Test 2D inputs with PyTorch tensors"""
        import torch

        # Create 2D inputs
        inputs_np = [
            np.random.rand(80, 1000),
            np.random.rand(80, 1200),
        ]

        # Convert to torch tensors
        inputs_torch = [torch.tensor(x, dtype=torch.float32) for x in inputs_np]

        # Process with numpy arrays
        processed_np = self.feature_extractor(inputs_np, padding="longest", return_tensors="np")

        # Process with torch tensors
        processed_torch = self.feature_extractor(inputs_torch, padding="longest", return_tensors="pt")

        # Check shapes match
        self.assertEqual(
            processed_np.input_values.shape,
            tuple(processed_torch.input_values.shape),
        )

        # Convert torch output to numpy and compare values (should be close)
        torch_as_np = processed_torch.input_values.numpy()
        self.assertTrue(np.allclose(processed_np.input_values, torch_as_np, atol=1e-5))

    def test_pad_to_multiple_of_2d(self):
        """Test pad_to_multiple_of with 2D inputs"""
        # Create 2D inputs
        inputs = [
            np.random.rand(80, 1000),  # Not a multiple of 64
            np.random.rand(80, 1025),  # Not a multiple of 64
        ]

        # Process with pad_to_multiple_of
        processed = self.feature_extractor(
            inputs, padding="longest", pad_to_multiple_of=64, return_tensors="np"
        )

        # The sequence length should be padded to the next multiple of 64 after 1025
        expected_length = 1088  # 1025 padded to next multiple of 64
        self.assertEqual(processed.input_values.shape, (2, 80, expected_length))

    def test_different_feature_dims_error(self):
        """Test that different feature dimensions raise an error"""
        # Create 2D inputs with different feature dimensions
        inputs = [
            np.random.rand(80, 1000),  # 80 features
            np.random.rand(100, 1000),  # 100 features
        ]

        # Should raise ValueError due to different feature dimensions
        with self.assertRaises(ValueError):
            self.feature_extractor(inputs, padding="longest", return_tensors="np")

    def test_3d_input_error(self):
        """Test that 3D inputs raise an error"""
        # Create a 3D input
        input_3d = np.random.rand(2, 80, 1000)  # (channels, features, time)

        # Should raise ValueError due to 3D input
        with self.assertRaises(ValueError):
            self.feature_extractor(input_3d, return_tensors="np")

        # Create a batch of 3D inputs
        inputs_3d = [
            np.random.rand(2, 80, 1000),  # (channels, features, time)
            np.random.rand(2, 80, 1200),  # (channels, features, time)
        ]

        # Should raise ValueError due to 3D input
        with self.assertRaises(ValueError):
            self.feature_extractor(inputs_3d, padding="longest", return_tensors="np")

    def test_non_array_2d_input(self):
        """Test 2D inputs provided as lists of lists"""
        # Create a 2D input as a list of lists
        input_2d = [[float(i + j) for j in range(100)] for i in range(80)]

        # Process the list of lists
        processed = self.feature_extractor(input_2d, return_tensors="np")

        # Check output shape
        self.assertEqual(processed.input_values.shape, (1, 80, 100))

    def test_2d_input_without_padding(self):
        """Test 2D inputs without padding"""
        # Create 2D inputs of different lengths
        inputs = [
            np.random.rand(80, 1000),
            np.random.rand(80, 1200),
        ]

        # Process without padding
        processed = self.feature_extractor(inputs, padding=False)

        # Should return a list of arrays with original shapes
        self.assertEqual(len(processed.input_values), 2)
        self.assertEqual(processed.input_values[0].shape, (80, 1000))
        self.assertEqual(processed.input_values[1].shape, (80, 1200))