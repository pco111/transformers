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


import itertools
import random
import unittest

import numpy as np

from transformers import Wav2Vec2FeatureExtractor
from transformers.testing_utils import require_torch


global_rng = random.Random()


def floats_list(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    values = []
    for batch_idx in range(shape[0]):
        values.append([])
        for _ in range(shape[1]):
            values[-1].append(rng.random() * scale)

    return values


class Wav2Vec2FeatureExtraction2DTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        min_seq_length=400,
        max_seq_length=2000,
        feature_size=80,  # Using a larger feature size for 2D inputs
        padding_value=0.0,
        sampling_rate=16000,
        return_attention_mask=True,
        do_normalize=True,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.seq_length_diff = (self.max_seq_length - self.min_seq_length) // (self.batch_size - 1)
        self.feature_size = feature_size
        self.padding_value = padding_value
        self.sampling_rate = sampling_rate
        self.return_attention_mask = return_attention_mask
        self.do_normalize = do_normalize

    def prepare_feat_extract_dict(self):
        return {
            "feature_size": self.feature_size,
            "padding_value": self.padding_value,
            "sampling_rate": self.sampling_rate,
            "return_attention_mask": self.return_attention_mask,
            "do_normalize": self.do_normalize,
        }

    def prepare_1d_inputs(self, equal_length=False, numpify=False):
        """Prepare 1D inputs for testing"""
        def _flatten(list_of_lists):
            return list(itertools.chain(*list_of_lists))

        if equal_length:
            speech_inputs = floats_list((self.batch_size, self.max_seq_length))
        else:
            # make sure that inputs increase in size
            speech_inputs = [
                _flatten(floats_list((x, 1)))
                for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            ]

        if numpify:
            speech_inputs = [np.asarray(x) for x in speech_inputs]

        return speech_inputs

    def prepare_2d_inputs(self, equal_length=False, numpify=True):
        """Prepare 2D inputs for testing"""
        if equal_length:
            # Create batch_size 2D arrays of shape (feature_size, max_seq_length)
            speech_inputs = [np.random.rand(self.feature_size, self.max_seq_length) for _ in range(self.batch_size)]
        else:
            # Create batch_size 2D arrays with increasing sequence length
            speech_inputs = [
                np.random.rand(self.feature_size, x)
                for x in range(self.min_seq_length, self.max_seq_length, self.seq_length_diff)
            ]

        if not numpify:
            speech_inputs = [x.tolist() for x in speech_inputs]

        return speech_inputs


class Wav2Vec2FeatureExtraction2DTest(unittest.TestCase):
    feature_extraction_class = Wav2Vec2FeatureExtractor

    def setUp(self):
        self.feat_extract_tester = Wav2Vec2FeatureExtraction2DTester(self)

    def _check_zero_mean_unit_variance(self, input_vector):
        self.assertTrue(np.all(np.mean(input_vector, axis=0) < 1e-3))
        self.assertTrue(np.all(np.abs(np.var(input_vector, axis=0) - 1) < 1e-3))

    def test_1d_input(self):
        """Test that 1D inputs still work as expected"""
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = self.feat_extract_tester.prepare_1d_inputs(numpify=True)

        # Test single 1D input
        processed = feat_extract(speech_inputs[0], return_tensors="np")
        self.assertEqual(processed.input_values.ndim, 2)  # (batch_size, seq_len)
        self.assertEqual(processed.input_values.shape[0], 1)  # batch_size=1

        # Test batch of 1D inputs with padding
        processed = feat_extract(speech_inputs, padding="longest", return_tensors="np")
        self.assertEqual(processed.input_values.ndim, 2)  # (batch_size, seq_len)
        self.assertEqual(processed.input_values.shape[0], len(speech_inputs))  # batch_size

    def test_2d_input_single(self):
        """Test that a single 2D input works"""
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_input = self.feat_extract_tester.prepare_2d_inputs(equal_length=True, numpify=True)[0]

        # Test single 2D input
        processed = feat_extract(speech_input, return_tensors="np")
        self.assertEqual(processed.input_values.ndim, 3)  # (batch_size, feature_dim, seq_len)
        self.assertEqual(processed.input_values.shape[0], 1)  # batch_size=1
        self.assertEqual(processed.input_values.shape[1], speech_input.shape[0])  # feature_dim

    def test_2d_input_batch_equal_length(self):
        """Test that a batch of 2D inputs with equal length works"""
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = self.feat_extract_tester.prepare_2d_inputs(equal_length=True, numpify=True)

        # Test batch of 2D inputs with equal length
        processed = feat_extract(speech_inputs, return_tensors="np")
        self.assertEqual(processed.input_values.ndim, 3)  # (batch_size, feature_dim, seq_len)
        self.assertEqual(processed.input_values.shape[0], len(speech_inputs))  # batch_size
        self.assertEqual(processed.input_values.shape[1], speech_inputs[0].shape[0])  # feature_dim

    def test_2d_input_batch_variable_length(self):
        """Test that a batch of 2D inputs with variable length works with padding"""
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = self.feat_extract_tester.prepare_2d_inputs(equal_length=False, numpify=True)

        # Test batch of 2D inputs with padding
        processed = feat_extract(speech_inputs, padding="longest", return_tensors="np")
        self.assertEqual(processed.input_values.ndim, 3)  # (batch_size, feature_dim, seq_len)
        self.assertEqual(processed.input_values.shape[0], len(speech_inputs))  # batch_size
        self.assertEqual(processed.input_values.shape[1], speech_inputs[0].shape[0])  # feature_dim
        # Should pad to the longest sequence
        self.assertEqual(processed.input_values.shape[2], speech_inputs[-1].shape[1])  # max_seq_len

    def test_2d_input_normalization(self):
        """Test that normalization works correctly for 2D inputs"""
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_input = self.feat_extract_tester.prepare_2d_inputs(equal_length=True, numpify=True)[0]

        # Test normalization on 2D input
        processed = feat_extract(speech_input, return_tensors="np")
        # Check normalization along time dimension (axis=2)
        # For 2D input, the output shape is (batch_size, feature_dim, seq_len)
        self.assertEqual(processed.input_values.ndim, 3)
        for i in range(processed.input_values.shape[1]):
            # Check mean close to 0
            self.assertTrue(np.abs(np.mean(processed.input_values[0, i, :])) < 1e-3)
            # Check variance close to 1
            self.assertTrue(np.abs(np.var(processed.input_values[0, i, :]) - 1) < 1e-3)

    def test_attention_mask_2d(self):
        """Test that attention mask works correctly for 2D inputs"""
        feat_dict = self.feat_extract_tester.prepare_feat_extract_dict()
        feat_dict["return_attention_mask"] = True
        feat_extract = self.feature_extraction_class(**feat_dict)
        speech_inputs = self.feat_extract_tester.prepare_2d_inputs(equal_length=False, numpify=True)

        # Test attention mask with 2D inputs
        processed = feat_extract(speech_inputs, padding="longest", return_tensors="np")
        self.assertIn("attention_mask", processed)
        # Attention mask should be (batch_size, seq_len)
        self.assertEqual(processed.attention_mask.shape[0], len(speech_inputs))
        self.assertEqual(processed.attention_mask.shape[1], processed.input_values.shape[2])

        # Verify mask values - should be 1 for actual values, 0 for padding
        for i, input_len in enumerate([x.shape[1] for x in speech_inputs]):
            # All positions up to input_len should be 1
            self.assertTrue(np.all(processed.attention_mask[i, :input_len] == 1))
            # All positions after input_len should be 0
            if input_len < processed.attention_mask.shape[1]:
                self.assertTrue(np.all(processed.attention_mask[i, input_len:] == 0))
                
        # Test attention mask with 3D inputs
        speech_inputs_3d = [
            np.random.rand(2, 500),  # (channels=2, seq_len=500)
            np.random.rand(2, 700),  # (channels=2, seq_len=700)
        ]
        processed_3d = feat_extract(speech_inputs_3d, padding="longest", return_tensors="np")
        self.assertIn("attention_mask", processed_3d)
        # Attention mask should be (batch_size, seq_len)
        self.assertEqual(processed_3d.attention_mask.shape[0], len(speech_inputs_3d))
        self.assertEqual(processed_3d.attention_mask.shape[1], processed_3d.input_values.shape[2])
        
        # Verify mask values for 3D inputs
        for i, input_len in enumerate([x.shape[1] for x in speech_inputs_3d]):
            # All positions up to input_len should be 1
            self.assertTrue(np.all(processed_3d.attention_mask[i, :input_len] == 1))
            # All positions after input_len should be 0
            if input_len < processed_3d.attention_mask.shape[1]:
                self.assertTrue(np.all(processed_3d.attention_mask[i, input_len:] == 0))

    def test_different_feature_dim_error(self):
        """Test that an error is raised when 2D inputs have different feature dimensions"""
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        
        # Create inputs with different feature dimensions
        speech_inputs = [
            np.random.rand(80, 1000),  # feature_dim = 80
            np.random.rand(100, 1000)  # feature_dim = 100
        ]
        
        # Should raise ValueError due to different feature dimensions
        with self.assertRaises(ValueError):
            feat_extract(speech_inputs, padding="longest", return_tensors="np")

    def test_3d_input_support(self):
        """Test that 3D inputs are supported"""
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        
        # Create a 3D input
        speech_input = np.random.rand(2, 80, 1000)  # (channels, feature_dim, seq_len)
        
        # Process 3D input
        processed = feat_extract(speech_input, return_tensors="np")
        
        # Check output shape
        self.assertEqual(processed.input_values.ndim, 3)  # (batch_size, channels*feature_dim, seq_len)
        self.assertEqual(processed.input_values.shape[0], 1)  # batch_size=1
        self.assertEqual(processed.input_values.shape[1], 2)  # channels=2
        self.assertEqual(processed.input_values.shape[2], 1000)  # seq_len=1000

    @require_torch
    def test_torch_tensors(self):
        """Test that torch tensors work for 2D inputs"""
        import torch
        
        feat_extract = self.feature_extraction_class(**self.feat_extract_tester.prepare_feat_extract_dict())
        speech_inputs = self.feat_extract_tester.prepare_2d_inputs(equal_length=True, numpify=True)
        
        # Convert to torch tensors
        torch_inputs = [torch.tensor(x) for x in speech_inputs]
        
        # Process with torch tensors
        processed = feat_extract(torch_inputs, return_tensors="pt")
        
        # Check output is a torch tensor with correct shape
        self.assertTrue(torch.is_tensor(processed.input_values))
        self.assertEqual(processed.input_values.dim(), 3)  # (batch_size, feature_dim, seq_len)
        self.assertEqual(processed.input_values.shape[0], len(speech_inputs))  # batch_size
        self.assertEqual(processed.input_values.shape[1], speech_inputs[0].shape[0])  # feature_dim
        
        # Test 3D torch tensor input
        speech_input_3d = np.random.rand(2, 80, 1000)  # (channels, feature_dim, seq_len)
        torch_input_3d = torch.tensor(speech_input_3d)
        
        # Process 3D torch tensor
        processed_3d = feat_extract(torch_input_3d, return_tensors="pt")
        
        # Check output shape for 3D input
        self.assertTrue(torch.is_tensor(processed_3d.input_values))
        self.assertEqual(processed_3d.input_values.dim(), 3)  # (batch_size, channels, seq_len)
        self.assertEqual(processed_3d.input_values.shape[0], 1)  # batch_size=1
        self.assertEqual(processed_3d.input_values.shape[1], 2)  # channels=2
        self.assertEqual(processed_3d.input_values.shape[2], 1000)  # seq_len