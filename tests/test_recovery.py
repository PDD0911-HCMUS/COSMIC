"""Regression checks for damaged OSG copies and validation checkpoint recovery."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import gzip
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from config import load_config
from data.osg_storage import read_json
from model.criteria import build_criteria
from scripts.decompress_osg import main as decompress
from train_engine import TrainEngine


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 8)
        self.dropout = nn.Dropout(0.15)

    def forward(self, observation):
        descriptor = F.normalize(self.dropout(self.projection(observation["image"])), dim=-1)
        return {"grounding": {
            "fused_descriptor": descriptor,
            "graph_mask": torch.ones(len(descriptor), dtype=torch.bool),
        }}


class RecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.workspace = Path(__file__).resolve().parents[1]
        scratch = (self.workspace / "outputs").resolve()
        self.assertTrue(scratch.is_relative_to(self.workspace))
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="test_recovery_", dir=scratch)
        self.root = Path(self.temporary.name).resolve()
        self.assertTrue(self.root.is_relative_to(scratch))

    def tearDown(self):
        # The absolute scratch target was checked before recursive cleanup.
        self.temporary.cleanup()

    @staticmethod
    def graph(generation="test-generation"):
        return {
            "schema_version": 1, "bbox_format": "xyxy_pixels",
            "image_size": {"width": 20, "height": 10},
            "image_id": "image.jpg", "image_path": "image.jpg",
            "objects": [
                {"id": 0, "label": "building", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
                {"id": 1, "label": "tree", "confidence": 0.7, "bbox": [10, 0, 20, 10]},
            ],
            "relations": [{"subject": 0, "object": 1, "predicate": "beside",
                           "confidence": 0.8, "triplet_score": 0.9 * 0.8 * 0.7}],
            "provenance": {"generation_id": generation},
        }

    def osg_root(self):
        root = self.root / "osg"
        root.mkdir()
        (root / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "graph_format": "json", "generation_id": "test-generation",
            "complete_dataset": True, "completed_images": 5,
        }), encoding="utf-8")
        return root

    def run_repair(self, root):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            code = decompress([str(root), "--workers", "2", "--repair"])
        return code, output.getvalue()

    def test_read_error_names_the_file(self):
        path = self.root / "empty.json"
        path.write_bytes(b"")
        with self.assertRaises(ValueError) as caught:
            read_json(path)
        self.assertIn(str(path), str(caught.exception))
        self.assertIn("Expecting value", str(caught.exception))

    def test_repair_restores_bad_copies_and_keeps_valid_data(self):
        root = self.osg_root()
        content = (json.dumps(self.graph()) + "\n").encode()
        sources = []
        for name in ("empty", "schema", "generation", "missing", "valid"):
            source = root / (name + ".json.gz")
            source.write_bytes(gzip.compress(content, mtime=0))
            sources.append(source)
        (root / "empty.json").write_bytes(b"")
        (root / "schema.json").write_text('{"objects": []}', encoding="utf-8")
        (root / "generation.json").write_text(json.dumps(self.graph("wrong")), encoding="utf-8")
        valid = root / "valid.json"
        valid.write_bytes(content)
        modified = valid.stat().st_mtime_ns
        compressed = [source.read_bytes() for source in sources]
        original_manifest = (root / "manifest.json").read_bytes()
        code, output = self.run_repair(root)
        self.assertEqual(code, 0, output)
        self.assertIn("repaired=3", output)
        for source, before in zip(sources, compressed):
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(read_json(source.with_suffix("")), self.graph())
        self.assertEqual(valid.stat().st_mtime_ns, modified)
        self.assertEqual((root / "manifest.json").read_bytes(), original_manifest)

    def test_bad_source_does_not_replace_copy_or_activate_metadata(self):
        root = self.osg_root()
        target = root / "bad.json"
        target.write_bytes(b"")
        source = root / "bad.json.gz"
        source.write_bytes(gzip.compress(json.dumps(self.graph("wrong")).encode()))
        before = (root / "manifest.json").read_bytes()
        code, output = self.run_repair(root)
        self.assertEqual(code, 1, output)
        self.assertIn(str(source), output)
        self.assertEqual(target.read_bytes(), b"")
        self.assertEqual((root / "manifest.json").read_bytes(), before)
        self.assertFalse(list(root.glob("*.tmp")))

    def test_plain_only_invalid_graph_is_not_silently_ignored(self):
        root = self.osg_root()
        (root / "orphan.json").write_bytes(b"")
        code, output = self.run_repair(root)
        self.assertEqual(code, 1, output)
        self.assertIn("orphan.json.gz", output)

    def test_corrupt_gzip_names_source_and_keeps_existing_copy(self):
        root = self.osg_root()
        target = root / "truncated.json"
        target.write_bytes(b"")
        source = root / "truncated.json.gz"
        source.write_bytes(gzip.compress(json.dumps(self.graph()).encode())[:-8])
        code, output = self.run_repair(root)
        self.assertEqual(code, 1, output)
        self.assertIn(str(source), output)
        self.assertEqual(target.read_bytes(), b"")
        self.assertFalse(list(root.glob("*.tmp")))

    def test_default_extraction_still_compares_exact_bytes(self):
        root = self.osg_root()
        source = root / "copy.json.gz"
        content = (json.dumps(self.graph()) + "\n").encode()
        source.write_bytes(gzip.compress(content))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(decompress([str(root)]), 0)
        target = source.with_suffix("")
        self.assertEqual(target.read_bytes(), content)
        target.write_bytes(content + b" ")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(decompress([str(root)]), 1)
        self.assertEqual(target.read_bytes(), content + b" ")

    def config(self, directory, resume=None, epochs=2):
        config = load_config(self.workspace / "config.yaml")
        return replace(
            config, device="cpu", mode="train",
            data=replace(config.data, batch_size=4, num_workers=0),
            model=replace(config.model, criteria=replace(config.model.criteria, verification_weight=0)),
            training=replace(config.training, epochs=epochs, precision="fp32", accumulation_steps=1),
            output=replace(config.output, root=directory),
            save=replace(config.save, enabled=True, resume=resume),
        )

    def engine(self, directory, *, resume=None, epochs=2):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        config = self.config(directory, resume, epochs)
        model = TinyModel()
        criteria = build_criteria(config.model.criteria, "cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.optimizer.lr,
                                      weight_decay=config.training.optimizer.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9**epoch)
        context = SimpleNamespace(device="cpu", directory=directory,
                                  checkpoint_dir=directory / "checkpoints")
        engine = TrainEngine(model, config, context, criteria=criteria, optimizer=optimizer,
                             scheduler=scheduler, scaler=torch.amp.GradScaler("cuda", enabled=False))
        samples = []
        for index in range(8):
            a, b = index % 2, (index // 2) % 2
            samples.append({
                "a": {"image": torch.tensor([index / 8, 1., .3]), "place_label": a},
                "b": {"image": torch.tensor([.2, index / 8, 1.]), "place_label": b},
                "label": float(a == b),
            })
        loader = DataLoader(samples, batch_size=4, generator=torch.Generator().manual_seed(42))
        return engine, loader

    def compare_recovery(self, fail_epoch):
        baseline, loader = self.engine(self.root / "baseline")
        with redirect_stdout(io.StringIO()):
            baseline.fit(loader, validate=lambda model, epoch: {"recall@1": epoch / 10})
        broken, loader = self.engine(self.root / "interrupted")

        def fail_validation(model, epoch):
            if epoch == fail_epoch:
                raise ValueError("simulated invalid validation OSG")
            return {"recall@1": epoch / 10}

        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "invalid validation OSG"):
            broken.fit(loader, validate=fail_validation)
        pending_path = broken.context.checkpoint_dir / "pending_validation.pt"
        self.assertTrue(pending_path.is_file())
        payload = torch.load(pending_path, map_location="cpu", weights_only=True)
        self.assertEqual(payload["pending_validation"]["epoch"], fail_epoch)
        self.assertEqual(len(payload["history"]), fail_epoch - 1)
        if fail_epoch == 1:
            self.assertFalse((broken.context.checkpoint_dir / "last.pt").exists())
        else:
            previous = torch.load(broken.context.checkpoint_dir / "last.pt",
                                  map_location="cpu", weights_only=True)
            self.assertEqual(previous["epoch"], fail_epoch - 1)
        fresh, loader = self.engine(broken.context.directory)
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(FileExistsError, "pending_validation"):
            fresh.fit(loader, validate=lambda model, epoch: {"recall@1": .1})

        recovered, loader = self.engine(broken.context.directory, resume=pending_path)
        validated = []

        def validate(model, epoch):
            validated.append(epoch)
            return {"recall@1": epoch / 10}

        with redirect_stdout(io.StringIO()), patch.object(
                recovered, "train_one_epoch", wraps=recovered.train_one_epoch) as train:
            recovered.fit(loader, validate=validate)
        self.assertEqual([call.args[1] for call in train.call_args_list], list(range(fail_epoch + 1, 3)))
        self.assertEqual(validated, list(range(fail_epoch, 3)))
        self.assertFalse(pending_path.exists())
        self.assertEqual(recovered.global_step, baseline.global_step)
        self.assertEqual(recovered.scheduler.state_dict(), baseline.scheduler.state_dict())
        self.assertEqual(recovered.optimizer.param_groups[0]["lr"], baseline.optimizer.param_groups[0]["lr"])
        self.assertEqual(recovered.best_metric, baseline.best_metric)
        self.assertEqual([row["epoch"] for row in recovered.history], [1, 2])
        for key, expected in baseline.model.state_dict().items():
            self.assertTrue(torch.equal(recovered.model.state_dict()[key], expected), key)

        if fail_epoch == 2:
            # Format-1 normal checkpoints from older code have no pending field.
            old_style = self.root / "normal_epoch_1.pt"
            torch.save(previous, old_style)
            normal, loader = self.engine(self.root / "normal_resume", resume=old_style)
            with redirect_stdout(io.StringIO()):
                normal.fit(loader, validate=lambda model, epoch: {"recall@1": epoch / 10})
            for key, expected in baseline.model.state_dict().items():
                self.assertTrue(torch.equal(normal.model.state_dict()[key], expected), key)
            self.assertEqual(normal.scheduler.state_dict(), baseline.scheduler.state_dict())

        # A normal completed checkpoint still skips already-completed epochs.
        last = recovered.context.checkpoint_dir / "last.pt"
        finished, loader = self.engine(recovered.context.directory, resume=last)
        with redirect_stdout(io.StringIO()), patch.object(
                finished, "train_one_epoch", wraps=finished.train_one_epoch) as train:
            finished.fit(loader, validate=validate)
        self.assertEqual(train.call_count, 0)
        self.assertEqual(finished.global_step, baseline.global_step)

    def test_resume_retries_validation_without_retraining_epoch(self):
        self.compare_recovery(fail_epoch=1)

    def test_last_epoch_pending_validation_is_not_skipped(self):
        self.compare_recovery(fail_epoch=2)


if __name__ == "__main__":
    unittest.main()
