"""Fast CPU unit tests for FrameFlow models, losses, export integrity, and datasets."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from frameflow.config import PRESETS, get_preset
from frameflow.dataset import (
    SequenceFrameDataset,
    SyntheticSmokeDataset,
    ValidationTripletDataset,
    WeightedMixDataset,
    overlay_synthetic_hud,
)
from frameflow.export import export_model
from frameflow import FrameFlowLoss, FrameFlowNet, GameFILoss, GameFINet
from frameflow.sr_config import SR_PRESETS, get_sr_preset
from frameflow.sr_dataset import GameSRDataset, ValidationSRDataset
from frameflow.sr_export import export_sr_model
from frameflow.sr_losses import GameSRLoss
from frameflow.sr_model import GameSRNet


class TestFrameFlow(unittest.TestCase):
    def test_presets_and_param_counts(self) -> None:
        self.assertIs(GameFINet, FrameFlowNet)
        self.assertIs(GameFILoss, FrameFlowLoss)
        std = GameFINet("std")
        lite = FrameFlowNet("lite")
        self.assertEqual(std.cfg.name, "GameFI-Std")
        self.assertEqual(lite.cfg.name, "GameFI-Lite")
        self.assertEqual(std.param_count(), 775_811)
        self.assertEqual(lite.param_count(), 222_179)
        self.assertEqual(get_preset("standard"), PRESETS["std"])

    def test_forward_shapes_and_arbitrary_t(self) -> None:
        for preset in ("std", "lite"):
            model = FrameFlowNet(preset).eval()
            a = torch.rand(2, 3, 64, 96)
            b = torch.rand(2, 3, 64, 96)
            t = torch.tensor([0.5, 1.0 / 3.0])
            with torch.no_grad():
                out = model(a, b, t, k=1, still_test=False)
                out_still = model(a, b, t, k=2, still_test=True)
            self.assertEqual(tuple(out["render"].shape), (2, 3, 64, 96))
            self.assertEqual(tuple(out_still["render"].shape), (2, 3, 64, 96))
            self.assertEqual(tuple(out["flow_tq"].shape), (2, 4, 16, 24))
            self.assertEqual(tuple(out["vis_tq"].shape), (2, 3, 16, 24))
            self.assertEqual(tuple(out["R_t"].shape), (2, 3, 16, 24))
            self.assertTrue(torch.isfinite(out["render"]).all())
            self.assertGreaterEqual(float(out["render"].min()), 0.0)
            self.assertLessEqual(float(out["render"].max()), 1.0)

    def test_backward_and_kd_gradients(self) -> None:
        teacher = FrameFlowNet("std").eval()
        student = FrameFlowNet("lite").train()
        crit = FrameFlowLoss()

        a = torch.rand(2, 3, 64, 64)
        b = torch.rand(2, 3, 64, 64)
        gt = 0.5 * (a + b)
        t = torch.tensor([0.5, 0.25])
        hud_mask = torch.zeros(2, 1, 64, 64)
        hud_mask[:, :, 48:60, 10:50] = 1.0

        with torch.no_grad():
            t_out = teacher(a, b, t)
        s_out = student(a, b, t)
        loss, metrics = crit(s_out, a, b, gt, t, hud_mask=hud_mask, teacher_out=t_out)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("hud", metrics)
        self.assertIn("kd_flow", metrics)

        loss.backward()
        grads_present = sum(1 for p in student.parameters() if p.grad is not None and torch.isfinite(p.grad).all())
        self.assertEqual(grads_present, sum(1 for _ in student.parameters()))

    def test_export_graph_and_manifest_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for preset in ("std", "lite"):
                model = FrameFlowNet(preset).eval()
                out_dir = Path(tmp) / preset
                info = export_model(model, out_dir, train_step=123)
                manifest = json.loads((out_dir / "model.json").read_text())
                graph = json.loads((out_dir / "graph.json").read_text())
                blob_bytes = (out_dir / "model.bin").read_bytes()

                self.assertEqual(len(blob_bytes), info["n_params"] * 2)
                self.assertEqual(manifest["meta"]["model"], f"GameFI-{preset.capitalize()}")
                self.assertEqual(graph["meta"]["model"], f"GameFI-{preset.capitalize()}")
                self.assertEqual(manifest["meta"]["train_step"], 123)
                tensors = manifest["tensors"]

                # Verify every weight referenced in graph.json (including repeat bodies) exists in manifest
                def check_ops(ops: list[dict]) -> None:
                    for op in ops:
                        for wname in op.get("w", []):
                            self.assertIn(wname, tensors, f"Missing weight {wname} in {preset} manifest")
                            tmeta = tensors[wname]
                            nelem = 1
                            for dim in tmeta["shape"]:
                                nelem *= dim
                            self.assertLessEqual((tmeta["offset"] + nelem) * 2, len(blob_bytes))
                        if op["op"] == "repeat":
                            check_ops(op["body"])

                check_ops(graph["ops"])

    def test_hud_overlay_and_datasets(self) -> None:
        import random

        img = np.full((128, 128, 3), 90, dtype=np.uint8)
        a, gt, b, mask = overlay_synthetic_hud(img, img, img, random.Random(7))
        self.assertEqual(a.shape, (128, 128, 3))
        self.assertGreater(mask.sum(), 0.0)
        # Static HUD must be identical across A, GT, B
        np.testing.assert_array_equal(a, gt)
        np.testing.assert_array_equal(gt, b)

        smoke = SyntheticSmokeDataset(length=4, crop_size=64, hud_prob=1.0)
        item = smoke[0]
        self.assertEqual(tuple(item["A"].shape), (3, 64, 64))
        self.assertGreater(float(item["hud_mask"].sum()), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip0"
            clip.mkdir()
            for i in range(6):
                arr = np.full((96, 128, 3), i * 35, dtype=np.uint8)
                Image.fromarray(arr).save(clip / f"{i:04d}.jpg")

            seq_ds = SequenceFrameDataset([tmp], crop_size=64, hud_prob=0.5, samples_per_epoch=8)
            val_ds = ValidationTripletDataset([tmp], crop_size=64, max_samples=4, hud_prob=0.5)
            mix_ds = WeightedMixDataset([seq_ds, smoke], [0.7, 0.3], samples_per_epoch=6)

            for ds in (seq_ds, val_ds, mix_ds):
                sample = ds[0]
                self.assertEqual(tuple(sample["A"].shape), (3, 64, 64))
                self.assertEqual(tuple(sample["GT"].shape), (3, 64, 64))
                self.assertGreater(float(sample["t"]), 0.0)
                self.assertLess(float(sample["t"]), 1.0)

    def test_gamesr_models_losses_export_and_dataset(self) -> None:
        sr_std = GameSRNet("sr_std")
        sr_lite = GameSRNet("sr_lite")
        self.assertEqual(sr_std.param_count(), 11_548)
        self.assertEqual(sr_lite.param_count(), 6_936)
        self.assertEqual(get_sr_preset("standard"), SR_PRESETS["sr_std"])
        self.assertEqual(get_sr_preset("lite"), SR_PRESETS["sr_lite"])

        lr = torch.rand(2, 3, 32, 48, dtype=torch.float32)
        hr = torch.rand(2, 3, 64, 96, dtype=torch.float32)
        criterion = GameSRLoss(w_kd=0.25)
        with torch.no_grad():
            t_out = sr_std(lr)
        out = sr_lite(lr)
        self.assertEqual(tuple(out["render"].shape), (2, 3, 64, 96))
        self.assertEqual(tuple(out["delta_hr"].shape), (2, 3, 64, 96))
        self.assertEqual(tuple(out["base_hr"].shape), (2, 3, 64, 96))
        self.assertEqual(tuple(out["delta_lr"].shape), (2, 12, 32, 48))
        self.assertGreaterEqual(float(out["render"].detach().min()), 0.0)
        self.assertLessEqual(float(out["render"].detach().max()), 1.0)

        loss, metrics = criterion(out, hr, teacher_out=t_out)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(metrics["total"], 0.0)
        self.assertIn("kd", metrics)
        loss.backward()
        grad_sum = sum(p.grad.abs().sum().item() for p in sr_lite.parameters() if p.grad is not None)
        self.assertGreater(grad_sum, 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            for preset in ("sr_std", "sr_lite"):
                model = GameSRNet(preset).eval()
                out_dir = Path(tmp) / preset
                info = export_sr_model(model, out_dir, train_step=77)
                manifest = json.loads((out_dir / "model.json").read_text())
                graph = json.loads((out_dir / "graph.json").read_text())
                blob_bytes = (out_dir / "model.bin").read_bytes()

                self.assertEqual(len(blob_bytes), info["n_params"] * 2)
                self.assertEqual(manifest["meta"]["task"], "sr")
                self.assertEqual(manifest["meta"]["scale"], 2)
                self.assertEqual(manifest["meta"]["train_step"], 77)
                tensors = manifest["tensors"]
                for op in graph["ops"]:
                    for wname in op.get("w", []):
                        self.assertIn(wname, tensors)
                        nelem = 1
                        for dim in tensors[wname]["shape"]:
                            nelem *= dim
                        self.assertLessEqual((tensors[wname]["offset"] + nelem) * 2, len(blob_bytes))

            ds_dir = Path(tmp) / "sr_imgs"
            ds_dir.mkdir()
            Image.fromarray(np.full((160, 160, 3), 120, dtype=np.uint8)).save(ds_dir / "sample.png")
            ds = GameSRDataset([(ds_dir, 1.0)], hr_crop=128, scale=2, ui_prob=0.1, samples_per_epoch=20)
            vds = ValidationSRDataset([ds_dir], hr_crop=128, scale=2, max_samples=4)
            for idx in range(len(ds)):
                s = ds[idx]
                self.assertEqual(tuple(s["LR"].shape), (3, 64, 64))
                self.assertEqual(tuple(s["HR"].shape), (3, 128, 128))
            vs = vds[0]
            self.assertEqual(tuple(vs["LR"].shape), (3, 64, 64))
            self.assertEqual(tuple(vs["HR"].shape), (3, 128, 128))


if __name__ == "__main__":
    unittest.main()
