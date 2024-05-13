import os
from unittest import TestCase

import torch

from speech_hypertuning.tasks.models import S3PRLUpstreamMLPDownstreamForCls


class S3PRLUpstreamMLPDownstreamForClsTestCase(TestCase):
    def setUp(self):
        self.state = {
            "speaker_id_mapping": {
                "id1": 1,
                "id2": 2,
                "id3": 3,
                "id4": 4,
                "id5": 5,
            }
        }
        self.model = S3PRLUpstreamMLPDownstreamForCls(self.state)
        self.embedding_example = torch.load(
            os.path.dirname(os.path.realpath(__file__)) + "/data/embedding_example.pt"
        )
        self.mocked_out = torch.tensor(
            [
                [
                    0.087228991091251,
                    -0.042148575186729,
                    0.079113274812698,
                    -0.051021523773670,
                    -0.021729048341513,
                ]
            ]
        )

        # Set random seeds
        torch.manual_seed(44)
        torch.use_deterministic_algorithms(True)

        self.waveform = torch.rand(1, 16000)

    def test_forward_from_audio(self):
        valid_length = torch.tensor([self.waveform.size(1)])
        with torch.no_grad():
            out = self.model({'wav': self.waveform, 'wav_lens': valid_length})

        self.assertEqual(out.shape, torch.Size([1, 5]))

        torch.testing.assert_close(out, self.mocked_out)

    def test_forward_from_precalculated_upstream_embedding(self):

        with torch.no_grad():
            out = self.model(
                {
                    'upstream_embedding_precalculated': torch.Tensor([True]),
                    'upstream_embedding': self.embedding_example,
                }
            )

        self.assertEqual(out.shape, torch.Size([1, 5]))

        torch.testing.assert_close(
            out,
            self.mocked_out,
        )

    # FIXME def test_forward_from_batch_audio(self):
    #     waveform_batch = torch.cat([self.waveform, self.waveform])
    #     valid_length = torch.tensor([self.waveform.size(1), self.waveform.size(1)])

    #     with torch.no_grad():
    #         out = self.model({'wav': waveform_batch, 'wav_lens': valid_length})

    #     self.assertEqual(out.shape, torch.Size([2, 5]))

    #     expected_output = torch.cat([self.mocked_out, self.mocked_out])
    #     torch.testing.assert_close(out, expected_output)

    def test_forward_from_batch_precalculated(self):
        batch_embeddings = torch.cat([self.embedding_example, self.embedding_example])

        with torch.no_grad():
            out = self.model(
                {
                    'upstream_embedding_precalculated': torch.Tensor([True]),
                    'upstream_embedding': batch_embeddings,
                }
            )

        self.assertEqual(out.shape, torch.Size([2, 5]))

        expected_output = torch.cat([self.mocked_out, self.mocked_out])
        torch.testing.assert_close(out, expected_output)
