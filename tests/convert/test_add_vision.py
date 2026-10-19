import torch

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorSpec
from tools.artifact.writer import ArtifactWriter
from tools.convert.add_vision import add_vision
from tools.convert.qwen3_5 import text_config
from tests.convert.test_qwen3_5 import _checkpoint, _config


def test_add_vision_preserves_original_weights_resources_and_bindings(tmp_path):
    config = _config()
    tensors = {
        "model.visual.patch_embed.proj.weight": torch.zeros(8, 3, 1, 2, 2),
        "model.visual.patch_embed.proj.bias": torch.zeros(8),
        "model.visual.pos_embed.weight": torch.zeros(4, 8),
        "model.visual.blocks.0.attn.qkv.weight": torch.zeros(24, 8),
        "model.visual.blocks.0.attn.qkv.bias": torch.zeros(24),
        "model.visual.blocks.0.attn.proj.weight": torch.zeros(8, 8),
        "model.visual.blocks.0.attn.proj.bias": torch.zeros(8),
        "model.visual.blocks.0.mlp.linear_fc1.weight": torch.zeros(12, 8),
        "model.visual.blocks.0.mlp.linear_fc1.bias": torch.zeros(12),
        "model.visual.blocks.0.mlp.linear_fc2.weight": torch.zeros(8, 12),
        "model.visual.blocks.0.mlp.linear_fc2.bias": torch.zeros(8),
        "model.visual.merger.linear_fc1.weight": torch.zeros(32, 32),
        "model.visual.merger.linear_fc1.bias": torch.zeros(32),
        "model.visual.merger.linear_fc2.weight": torch.zeros(16, 32),
        "model.visual.merger.linear_fc2.bias": torch.zeros(16),
    }
    for norm in ("blocks.0.norm1", "blocks.0.norm2", "merger.norm"):
        for role in ("weight", "bias"):
            tensors[f"model.visual.{norm}.{role}"] = torch.zeros(8)
    source_path = tmp_path / "source"
    with _checkpoint(source_path, config, tensors):
        pass
    refs = {role: "resource/text/" + role for role in (
        "tokenizer.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja")}
    payloads = {key: (source_path / role).read_bytes() for role, key in refs.items()}
    payloads["weight/000000"] = bytes.fromhex("803f00400040803f")
    specs = [ResourceSpec(key, len(data)) for key, data in payloads.items() if key.startswith("resource/")]
    specs.append(TensorSpec("weight/000000", (2, 2), "bf16", "contiguous_le_v1"))
    bindings = {"text/output_head": {"object": "weight/000000"}}
    components = {"text": {"config": text_config(config, mtp=False), "resources": refs}}
    original = tmp_path / "text.ninfer"
    with ArtifactWriter(original, specs, components=components, bindings=bindings) as writer:
        for key, data in payloads.items():
            writer.write_object(key, data)
    output = tmp_path / "vision.ninfer"
    add_vision(original, source_path, output)
    with Artifact(output) as result:
        assert result.directory.components["text"] == components["text"]
        assert result.directory.bindings["text/output_head"] == bindings["text/output_head"]
        assert "vision/patch_embedding" in result.directory.bindings
        for key, data in payloads.items():
            assert result.read_object(key) == data
