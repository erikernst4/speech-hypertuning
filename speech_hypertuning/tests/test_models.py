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
                    0.088501535356045,
                    -0.043360225856304,
                    0.076865889132023,
                    -0.049365781247616,
                    -0.018453067168593,
                ]
            ]
        )

    def test_forward_from_audio(self):
        waveform = torch.rand(1, 16000)
        valid_length = torch.tensor([waveform.size(1)])
        out = self.model({'wav': waveform, 'wav_lens': valid_length})

        self.assertEqual(out.shape, torch.Size([1, 5]))

        torch.testing.assert_close(out, self.mocked_out)

    def test_forward_from_precalculated_upstream_embedding(self):

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
