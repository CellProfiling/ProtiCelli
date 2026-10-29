"""Compare inherited normalization behavior with its upstream implementation."""

import unittest
import tempfile
import json
from pathlib import Path

import torch
from diffusers.models.attention import BasicTransformerBlock as UpstreamBlock

from proticelli.models.basic_transformer_block import BasicTransformerBlock
from proticelli.models.dit import DiTTransformer2DModel
from proticelli import Model


class TransformerCompatibilityTests(unittest.TestCase):
    def test_legacy_discrete_loading_preserves_weights_and_reserved_null_row(self):
        with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory() as directory:
            torch.random.default_generator.manual_seed(84)
            legacy = DiTTransformer2DModel(
                num_attention_heads=2, attention_head_dim=4, in_channels=4,
                out_channels=2, num_layers=1, sample_size=8, patch_size=2,
                num_embeds_ada_norm=16, num_protein_labels=20, num_cell_labels=3,
                norm_type="ada_norm_zero", legacy_discrete_capacity=True,
            ).eval()
            path = Path(directory) / "unet"
            legacy.save_pretrained(path)
            config_path = path / "config.json"
            config = json.loads(config_path.read_text())
            # An old checkpoint predates this option; its actual weights and
            # capacity remain unchanged, including the reserved dropout row.
            config.pop("legacy_discrete_capacity")
            config_path.write_text(json.dumps(config), encoding="utf-8")
            explicit = DiTTransformer2DModel.from_pretrained(
                path, local_files_only=True, legacy_discrete_capacity=True).eval()
            automatic = Model(checkpoint_dir=directory, device="cpu", dtype="float32",
                              protein_map={"KNOWN": 1}, cellline_map={"CELL": 1}).model
            for name, value in legacy.state_dict().items():
                torch.testing.assert_close(explicit.state_dict()[name], value, rtol=0, atol=0)
                torch.testing.assert_close(automatic.state_dict()[name], value, rtol=0, atol=0)
            image = torch.zeros(2, 4, 8, 8)
            conditions = {"timestep": torch.tensor([2, 5]),
                          "protein_labels": torch.tensor([0, 15]),
                          "cell_line_labels": torch.tensor([0, 2])}
            with torch.no_grad():
                expected = legacy(image, **conditions).sample
                torch.testing.assert_close(automatic(image, **conditions).sample, expected, rtol=0, atol=0)
                conditions["protein_labels"] = torch.tensor([0, 19])
                with self.assertRaisesRegex(ValueError, "no trained class embeddings"):
                    automatic(image, **conditions)

    def test_discrete_dit_accepts_the_full_declared_protein_vocabulary(self):
        with torch.random.fork_rng(devices=[]), tempfile.TemporaryDirectory() as directory:
            torch.random.default_generator.manual_seed(83)
            model = DiTTransformer2DModel(
                num_attention_heads=2, attention_head_dim=4, in_channels=4,
                out_channels=2, num_layers=1, sample_size=8, patch_size=2,
                num_embeds_ada_norm=16, num_protein_labels=20, num_cell_labels=3,
                norm_type="ada_norm_zero",
            ).eval()
            image = torch.zeros(2, 4, 8, 8)
            conditions = {"timestep": torch.tensor([2, 5]),
                          "protein_labels": torch.tensor([0, model.config.num_protein_labels - 1]),
                          "cell_line_labels": torch.tensor([0, model.config.num_cell_labels - 1])}
            with torch.no_grad():
                expected = model(image, **conditions).sample
                self.assertEqual(expected.shape, (2, 2, 8, 8))
                self.assertTrue(torch.isfinite(expected).all().item())
                model.save_pretrained(directory)
                restored = DiTTransformer2DModel.from_pretrained(directory, local_files_only=True).eval()
                torch.testing.assert_close(restored(image, **conditions).sample, expected, rtol=0, atol=0)

    def test_discrete_and_single_norms_match_upstream_for_the_same_weights(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(81)
            hidden = torch.linspace(-1, 1, 2 * 4 * 8).reshape(2, 4, 8)
            for mode in ("ada_norm_zero", "ada_norm_single"):
                with self.subTest(mode=mode):
                    options = dict(dim=8, num_attention_heads=2, attention_head_dim=4,
                                   num_embeds_ada_norm=16, norm_type=mode)
                    reference = UpstreamBlock(**options).eval()
                    actual = BasicTransformerBlock(**options).eval()
                    actual.load_state_dict(reference.state_dict())
                    conditions = {"timestep": torch.tensor([2, 5]),
                                  "class_labels": torch.tensor([1, 2])}
                    if mode == "ada_norm_single":
                        conditions["timestep"] = torch.linspace(-0.3, 0.7, 2 * 6 * 8).reshape(2, 48)
                    with torch.no_grad():
                        expected = reference(hidden, **conditions)
                        observed = actual(hidden, **conditions)
                    torch.testing.assert_close(observed, expected, rtol=0, atol=0)

    def test_dit_forwards_discrete_timestep_and_protein_labels_to_blocks(self):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(82)
            model = DiTTransformer2DModel(
                num_attention_heads=2, attention_head_dim=4, in_channels=4,
                out_channels=2, num_layers=1, sample_size=8, patch_size=2,
                num_embeds_ada_norm=16, num_protein_labels=5, num_cell_labels=3,
                norm_type="ada_norm_zero",
            ).eval()
            image = torch.linspace(-1, 1, 2 * 4 * 8 * 8).reshape(2, 4, 8, 8)
            conditions = {"timestep": torch.tensor([2, 5]),
                          "protein_labels": torch.tensor([1, 2]),
                          "cell_line_labels": torch.tensor([1, 2])}
            # The actual upstream block supplies a separate executable reference
            # for conditioning; model embedding/output geometry stays real.
            options = dict(dim=8, num_attention_heads=2, attention_head_dim=4,
                           num_embeds_ada_norm=16, norm_type="ada_norm_zero",
                           norm_elementwise_affine=model.config.norm_elementwise_affine,
                           norm_eps=model.config.norm_eps,
                           activation_fn=model.config.activation_fn,
                           attention_bias=model.config.attention_bias)
            reference = UpstreamBlock(**options).eval()
            reference.load_state_dict(model.transformer_blocks[0].state_dict())
            with torch.no_grad():
                embedding = torch.cat([
                    model.emb_protein_labels(conditions["timestep"], conditions["protein_labels"]),
                    model.emb_cell_line_labels(conditions["timestep"], conditions["cell_line_labels"]),
                ], dim=1)
                patches = model.pos_embed(image)
                expected = reference(patches, timestep=conditions["timestep"],
                                     class_labels=conditions["protein_labels"])
                observed = model.transformer_blocks[0](patches, embedding=embedding,
                    timestep=conditions["timestep"], class_labels=conditions["protein_labels"])
                torch.testing.assert_close(observed, expected, rtol=0, atol=0)
                output = model(image, **conditions).sample
            self.assertEqual(output.shape, (2, 2, 8, 8))
            self.assertTrue(torch.isfinite(output).all().item())
            original_input = image.clone().requires_grad_(True)
            original = model(original_input, **conditions).sample
            original_gradient, = torch.autograd.grad(original.square().mean(), original_input)
            model.gradient_checkpointing = True
            checkpointed_input = image.clone().requires_grad_(True)
            checkpointed = model(checkpointed_input, **conditions).sample
            checkpointed_gradient, = torch.autograd.grad(
                checkpointed.square().mean(), checkpointed_input)
            torch.testing.assert_close(checkpointed, original, rtol=0, atol=0)
            torch.testing.assert_close(checkpointed_gradient, original_gradient, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
