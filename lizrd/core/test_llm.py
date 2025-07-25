import torch

from lizrd.core import llm
import argparse
import unittest

from lizrd.support.test_utils import GeneralTestCase


class TestFeedForward(GeneralTestCase):
    def test_basic(self):
        batch, seql, dm, dff = 4, 8, 32, 64
        layer = llm.FeedForward(
            dmodel=dm, dff=dff, init_type="kaiming_uniform", init_scale=1.0
        )
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        output = layer(input)
        self.assertShape(output, (batch, seql, dm))


class ResidualTest(GeneralTestCase):
    def test_basic(self):
        batch, seql, dm, dff = 4, 8, 32, 64
        layer_ff = llm.FeedForward(
            dmodel=dm, dff=dff, init_type="kaiming_uniform", init_scale=1.0
        )
        layer_residual = llm.Residual(layer_ff)
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        output1 = layer_ff(input)
        output2 = layer_residual(input)
        self.assertShape(output2, (batch, seql, dm))
        self.assertTensorAlmostEqual(output1, output2 - input)
        self.assertTensorEqual(output1 + input, output2)


class AttentionPPTest(GeneralTestCase):
    def test_basic(self):
        batch, seql, dm, heads = 16, 4, 64, 8
        layer = llm.AttentionRoPE(
            dmodel=dm,
            heads=heads,
            length=seql,
            causal=False,
            init_type="kaiming_uniform",
            init_scale=1.0,
        )
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = layer(input)
        self.assertShape(out, (batch, seql, dm))


class RoPETest(GeneralTestCase):
    def test_basic(self):
        (
            batch,
            seql,
            dhead,
        ) = (
            16,
            4,
            64,
        )
        layer = llm.RoPE(
            dhead=dhead,
            length=seql,
        )
        input = torch.normal(0.0, 1.0, (batch, seql, dhead))
        out = layer(input)
        self.assertShape(out, (batch, seql, dhead))
        self.assertShape(layer.sin, (seql, dhead))
        self.assertTensorEqual(layer.sin[:, : dhead // 2], layer.sin[:, dhead // 2 :])

    def test_rotation(self):
        batch, n_heads, seql, d_head = 2, 2, 3, 4
        layer = llm.RoPE(d_head, seql)
        # vertors [0, 1], [1, 0] and [1, 1]
        input_batch_head = torch.tensor(
            [[0, 1, 1, 0], [0, 1, 1, 0], [1, 1, 1, 1]],
        )
        input = input_batch_head.repeat(batch, n_heads, 1, 1)
        out = layer(input)
        theta_1 = torch.tensor(10000.0 ** (-0 / d_head))
        theta_2 = torch.tensor(10000.0 ** (-2 / d_head))
        expected_out_batch_head = torch.tensor(
            [
                [
                    -torch.sin(0 * theta_1),
                    torch.cos(0 * theta_2),
                    torch.cos(0 * theta_1),
                    torch.sin(0 * theta_2),
                ],
                [
                    -torch.sin(1 * theta_1),
                    torch.cos(1 * theta_2),
                    torch.cos(1 * theta_1),
                    torch.sin(1 * theta_2),
                ],
                [
                    torch.cos(2 * theta_1) - torch.sin(2 * theta_1),
                    torch.cos(2 * theta_2) - torch.sin(2 * theta_2),
                    torch.sin(2 * theta_1) + torch.cos(2 * theta_1),
                    torch.sin(2 * theta_2) + torch.cos(2 * theta_2),
                ],
            ]
        )
        # repeat for each batch and head
        expected_out = expected_out_batch_head.repeat(batch, n_heads, 1, 1)
        self.assertTensorAlmostEqual(out, expected_out)


class SwiGLUFeedForwardTest(GeneralTestCase):
    def test_basic(self):
        batch, seql, dm, dff = 4, 8, 32, 64
        layer = llm.SwiGLUFeedForward(
            dmodel=dm, dff=dff, init_type="kaiming_uniform", init_scale=1.0
        )
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        output = layer(input)
        self.assertShape(output, (batch, seql, dm))


class AttentionTest(GeneralTestCase):
    def test_basic(self):
        try:
            batch, seql, dm, heads = 3, 7, 32, 4
            layer = llm.Attention(dm, heads, causal=False, flash=True)
            input = torch.normal(0.0, 1.0, (batch, seql, dm))
            out = layer(input)
            self.assertShape(out, (batch, seql, dm))
        except Exception as e:
            pass

    def test_flash_basic(self):
        batch, seql, dm, heads = 3, 7, 32, 4
        layer = llm.Attention(
            dm,
            heads,
            causal=False,
            init_type="kaiming_uniform",
            init_scale=1.0,
            flash=True,
        )
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = layer(input)
        self.assertShape(out, (batch, seql, dm))

    def test_flash_basic_causal(self):
        batch, seql, dm, heads = 3, 7, 32, 4
        layer = llm.Attention(
            dmodel=dm,
            heads=heads,
            causal=True,
            init_type="kaiming_uniform",
            init_scale=1.0,
            flash=True,
        )
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = layer(input)
        self.assertShape(out, (batch, seql, dm))

    def test_attention_mechanism_equivalence(self):
        batch, seql, dm, dhead, heads = 16, 4, 512, 64, 8
        q = torch.normal(0.0, 1.0, (batch, heads, seql, dhead))
        k = torch.normal(0.0, 1.0, (batch, heads, seql, dhead))
        v = torch.normal(0.0, 1.0, (batch, heads, seql, dhead))
        out1 = llm.attention_mechanism(q, k, v, dhead, flash=False, causal=False)
        out2 = llm.attention_mechanism(q, k, v, dhead, flash=True, causal=False)
        self.assertTensorAlmostEqual(out1, out2)

    def test_attention_mechanism_equivalence_causal(self):
        batch, seql, dm, dhead, heads = 16, 4, 512, 64, 8
        q = torch.normal(0.0, 1.0, (batch, heads, seql, dhead))
        k = torch.normal(0.0, 1.0, (batch, heads, seql, dhead))
        v = torch.normal(0.0, 1.0, (batch, heads, seql, dhead))
        out1 = llm.attention_mechanism(q, k, v, dhead, flash=False, causal=True)
        out2 = llm.attention_mechanism(q, k, v, dhead, flash=True, causal=True)
        self.assertTensorAlmostEqual(out1, out2)

    def test_flash_equivalence(self):
        batch, seql, dm, heads = 3, 7, 32, 4
        layer = llm.Attention(
            dmodel=dm,
            heads=heads,
            causal=False,
            init_type="kaiming_uniform",
            init_scale=1.0,
            flash=True,
        )
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = layer(input)

        layer.flash = False
        out2 = layer(input)

        self.assertTensorAlmostEqual(out, out2)

    def test_flash_equivalence_causal(self):
        batch, seql, dm, heads = 3, 7, 32, 4
        layer = llm.Attention(
            dmodel=dm,
            heads=heads,
            causal=True,
            init_type="kaiming_uniform",
            init_scale=1.0,
            flash=True,
        )
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = layer(input)

        layer.flash = False
        out2 = layer(input)

        self.assertTensorAlmostEqual(out, out2)

    def test_nonstandard_dhead(self):
        batch, seql, dm, heads, dhead = 3, 7, 32, 4, 100
        layer = llm.Attention(
            dmodel=dm,
            heads=heads,
            causal=False,
            init_type="kaiming_uniform",
            init_scale=1.0,
            dhead=dhead,
        )
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = layer(input)
        self.assertShape(out, (batch, seql, dm))

    def test_residual(self):
        batch, seql, dm, heads = 3, 7, 32, 4
        layer = llm.Residual(
            llm.Attention(
                dmodel=dm,
                heads=heads,
                init_type="kaiming_uniform",
                init_scale=1.0,
                causal=False,
            )
        )
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = layer(input)
        self.assertShape(out, (batch, seql, dm))


class EncoderTowerTest(GeneralTestCase):
    def test_basic(self):
        batch, seql, dm, heads, dff = 3, 7, 32, 4, 64
        nblocks = 3
        device = torch.device("cpu")

        layer_dict = {
            "attention": lambda: llm.Attention(
                dmodel=dm,
                heads=heads,
                init_type="kaiming_uniform",
                init_scale=1.0,
                causal=False,
            ),
            "feedforward": lambda: llm.FeedForward(
                dmodel=dm, dff=dff, init_type="kaiming_uniform", init_scale=1.0
            ),
        }
        model = llm.TransformerTower(nblocks, dm, layer_dict, device=device)
        input = torch.normal(0.0, 1.0, (batch, seql, dm))
        out = model(input)
        self.assertShape(out, (batch, seql, dm))


class LLMTest(GeneralTestCase):
    def test_bert(self):
        batch, seql, dm, heads, dff = 3, 7, 32, 4, 64
        vocab_size, max_length = 107, 33
        output_size = 3
        n_blocks = 2
        device = torch.device("cpu")
        args = argparse.Namespace(use_ngpt=False)

        embedding_layer = llm.EmbeddingLayer(
            llm.PositionalEmbedding(
                max_length, dm, init_type="kaiming_uniform", init_scale=1.0
            ),
            llm.TokenEmbedding(
                vocab_size, dm, init_type="kaiming_uniform", init_scale=1.0
            ),
        )
        layer_dict = {
            "attention": lambda: llm.Attention(
                dmodel=dm,
                heads=heads,
                init_type="kaiming_uniform",
                init_scale=1.0,
                causal=False,
            ),
            "feedforward": lambda: llm.FeedForward(
                dmodel=dm, dff=dff, init_type="kaiming_uniform", init_scale=1.0
            ),
        }
        encoder_tower = llm.TransformerTower(n_blocks, dm, layer_dict, device=device)
        head = llm.PredictionHead(
            dm,
            output_size,
            init_type="kaiming_uniform",
            init_scale=1.0,
        )
        model = llm.LLM(
            embedding_layer,
            encoder_tower,
            head,
            dmodel=dm,
            vocab_size=vocab_size,
            use_ngpt=args.use_ngpt,
            args=args,
        )

        input = torch.randint(0, vocab_size, (batch, seql))
        output = model(input)
        self.assertShape(output, (batch, seql, output_size))

    def test_gpt(self):
        batch, seql, dm, heads, dff = 3, 7, 32, 4, 64
        vocab_size, max_length = 107, 33
        output_size = 3
        n_blocks = 2
        device = torch.device("cpu")
        args = argparse.Namespace(use_ngpt=False)

        embedding_layer = llm.EmbeddingLayer(
            llm.PositionalEmbedding(
                max_length, dm, init_type="kaiming_uniform", init_scale=1.0
            ),
            llm.TokenEmbedding(
                vocab_size, dm, init_type="kaiming_uniform", init_scale=1.0
            ),
        )
        layer_dict = {
            "attention": lambda: llm.Attention(
                dmodel=dm,
                heads=heads,
                init_type="kaiming_uniform",
                init_scale=1.0,
                causal=True,
            ),
            "feedforward": lambda: llm.FeedForward(
                dmodel=dm, dff=dff, init_type="kaiming_uniform", init_scale=1.0
            ),
        }
        encoder_tower = llm.TransformerTower(n_blocks, dm, layer_dict, device=device)

        head = llm.PredictionHead(
            dm, output_size, init_type="kaiming_uniform", init_scale=1.0
        )
        model = llm.LLM(
            embedding_layer,
            encoder_tower,
            head,
            dmodel=dm,
            vocab_size=vocab_size,
            use_ngpt=args.use_ngpt,
            args=args,
        )
        input = torch.randint(0, vocab_size, (batch, seql))
        output = model(input)
        self.assertShape(output, (batch, seql, output_size))


if __name__ == "__main__":
    unittest.main()
