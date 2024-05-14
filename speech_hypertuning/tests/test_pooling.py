from unittest import TestCase

import torch

from speech_hypertuning.tasks.models.pooling import TemporalMeanPooling


class PoolingTestCase(TestCase):
    def setUp(self):
        # Set random seeds
        torch.manual_seed(44)
        torch.use_deterministic_algorithms(True)

    def test_single_embedding_1_hidden_dim_TMP_pooling(self):
        pooling_layer = TemporalMeanPooling()
        embedding = torch.FloatTensor([[[[1], [0]]]])

        expected_out = torch.FloatTensor([[[0.5]]])
        actual_out = pooling_layer(embedding)

        self.assertEqual(actual_out, expected_out)

    def test_single_embedding_2_hidden_dim_TMP_pooling(self):
        pooling_layer = TemporalMeanPooling()
        embedding = torch.FloatTensor([[[[1, 0], [0, 2]]]])

        expected_out = torch.FloatTensor([[[0.5, 1]]])
        actual_out = pooling_layer(embedding)

        torch.testing.assert_close(actual_out, expected_out)

    def test_single_embedding_2_hidden_dim_multiple_frames_TMP_pooling(self):
        pooling_layer = TemporalMeanPooling()
        embedding = torch.FloatTensor([[[[1, 0], [0, 2], [2, 4]]]])

        expected_out = torch.FloatTensor([[[1, 2]]])
        actual_out = pooling_layer(embedding)

        torch.testing.assert_close(actual_out, expected_out)

    def test_batch_embedding_2_hidden_dim_TMP_pooling(self):
        pooling_layer = TemporalMeanPooling()
        embedding = torch.FloatTensor([[[[1, 0], [0, 2]]], [[[1, 1], [0, 2]]]])

        expected_out = torch.FloatTensor([[[0.5, 1]], [[0.5, 1.5]]])
        actual_out = pooling_layer(embedding)

        torch.testing.assert_close(actual_out, expected_out)
