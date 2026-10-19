from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from tools.artifact.reader import Artifact
from tools.artifact.schema import ResourceSpec, TensorObject, TensorSpec
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from .model import Model
from .official_recipes import _optional
from .qwen3_5 import _Builder, text_config, vision_config
from .recipe import Recipe
from .resources import load_resources
from .sources.safetensors import SafetensorsSource


def add_vision(artifact_path: Path, source_path: Path, output: Path, device: str = "cpu") -> dict:
    with Artifact(artifact_path) as original, SafetensorsSource(source_path) as source:
        components = deepcopy(original.directory.components)
        if "vision" in components:
            raise ValueError("artifact already contains Vision")
        text = components["text"]["config"]
        candidate = text_config(source.config, mtp="mtp" in components)
        if candidate != text:
            raise ValueError("Vision source and artifact text configurations differ")
        config = vision_config(source.config, text)
        refs, resources, count, special = load_resources(
            source.root, vocab_size=text["vocab_size"], vision_config=config
        )
        for role in ("tokenizer.json", "tokenizer_config.json"):
            old = original.directory.components["text"]["resources"][role]
            original_tokens = json.loads(original.read_object(old))
            source_tokens = json.loads(resources[refs["text"][role]])
            if role == "tokenizer.json":
                original_tokens.pop("truncation", None)
                source_tokens.pop("truncation", None)
            else:
                original_tokens.setdefault("add_bos_token", False)
                source_tokens.setdefault("add_bos_token", False)
            if original_tokens != source_tokens:
                raise ValueError(f"Vision source tokenizer differs: {role}")
        components["vision"] = {"config": config, "target": "text", "resources": refs["vision"]}
        resources = {key: resources[key] for key in refs["vision"].values()}
        model = Model(components, resources=resources, token_count=count, special_token_ids=special)
        _Builder(model).vision(source, config, text["hidden_size"])
        recipe = Recipe(model)
        _optional(model, recipe)
        prepared = recipe.prepare(device=device, rows_per_chunk=512)
        mapping = {job.spec.id: "vision/" + job.spec.id for job in prepared.weights}
        mapping.update({spec.id: "vision/" + spec.id for spec, _ in prepared.auxiliaries})
        def remap(value):
            if isinstance(value, str):
                return mapping.get(value, value)
            if isinstance(value, dict):
                return {key: remap(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [remap(item) for item in value]
            return value
        prepared = replace(
            prepared,
            weights=tuple(replace(job, spec=replace(job.spec, id=mapping[job.spec.id])) for job in prepared.weights),
            auxiliaries=tuple((replace(spec, id=mapping[spec.id]), data) for spec, data in prepared.auxiliaries),
            bindings=remap(prepared.bindings), uses=tuple(remap(prepared.uses)),
        )
        specs = [
            TensorSpec(obj.id, obj.shape, obj.format, obj.layout, obj.divisors)
            if isinstance(obj, TensorObject) else ResourceSpec(obj.id, obj.bytes, obj.encoding)
            for obj in original.objects
        ]
        specs += [ResourceSpec(key, len(value)) for key, value in resources.items()]
        specs += [job.spec for job in prepared.weights]
        specs += [spec for spec, _ in prepared.auxiliaries]
        bindings = {**original.directory.bindings, **prepared.bindings}
        uses = [*original.directory.uses, *prepared.uses]
        digests = {}
        with ArtifactWriter(
            output, specs, components=components, bindings=bindings, uses=uses,
            metadata=original.directory.metadata,
            provenance={"text_artifact": str(artifact_path), "vision_source": str(source_path), "vision_recipe": "official"},
        ) as writer:
            for obj in original.objects:
                digest = hashlib.sha256()
                def chunks():
                    for chunk in original.iter_object(obj.id):
                        digest.update(chunk)
                        yield chunk
                writer.write_object(obj.id, chunks())
                digests[obj.id] = digest.hexdigest()
            for key, value in resources.items():
                writer.write_object(key, value)
            for index, job in enumerate(prepared.weights):
                job.prepared.produce(TensorOutput(writer, job.spec.id))
                print(f"Vision {index + 1}/{len(prepared.weights)}: {job.parameters[0]}", flush=True)
            for spec, data in prepared.auxiliaries:
                writer.write_object(spec.id, data)
        with Artifact(output) as result:
            for key, expected in digests.items():
                digest = hashlib.sha256()
                for chunk in result.iter_object(key):
                    digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError(f"original object changed: {key}")
        report = {"artifact": str(output), "source": str(source_path), "preserved_objects_sha256": digests,
                  "vision_objects": len(prepared.weights), "vision_recipe": "official", "components": components}
        Path(str(output) + ".vision.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    add_vision(args.artifact, args.source, args.out, args.device)


if __name__ == "__main__":
    main()
