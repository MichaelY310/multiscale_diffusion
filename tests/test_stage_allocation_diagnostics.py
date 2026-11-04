import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class StageAllocationDiagnosticTests(unittest.TestCase):
    def test_laplacian_pyramid_is_an_orthogonal_energy_split(self):
        module = _load("analyze_image_frequency", "2d/diagnostics/analyze_image_frequency.py")
        rng = np.random.default_rng(123)
        images = rng.standard_normal((2, 256, 256, 3), dtype=np.float32)
        p128 = module._block_project(images, 128)
        p64 = module._block_project(images, 64)
        channel_mean = images.mean(axis=(1, 2), keepdims=True)
        low = p64 - channel_mean
        middle = p128 - p64
        high = images - p128
        total = images - channel_mean

        reconstructed = low + middle + high
        np.testing.assert_allclose(reconstructed, total, rtol=0, atol=2e-6)
        component_energy = np.mean(low * low + middle * middle + high * high)
        total_energy = np.mean(total * total)
        np.testing.assert_allclose(component_energy, total_energy, rtol=2e-6, atol=2e-6)
        self.assertLess(abs(float(np.mean(low * middle))), 2e-6)
        self.assertLess(abs(float(np.mean(low * high))), 2e-6)
        self.assertLess(abs(float(np.mean(middle * high))), 2e-6)

    def test_report_parser_requires_all_five_metrics(self):
        module = _load("build_stage_allocation_report", "2d/diagnostics/build_stage_allocation_report.py")
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            complete = tmp_path / "complete.txt"
            complete.write_text(
                "Inception Score: 12.5\nFID: 4.25\nsFID: 5.5\nPrecision: 0.7\nRecall: 0.6\n",
                encoding="utf-8",
            )
            self.assertEqual(
                module.parse_metrics(complete),
                {"IS": 12.5, "FID": 4.25, "sFID": 5.5, "Precision": 0.7, "Recall": 0.6},
            )

            partial = tmp_path / "partial.txt"
            partial.write_text("Inception Score: 12.5\nFID: 4.25\nsFID: 5.5\n", encoding="utf-8")
            self.assertIsNone(module.parse_metrics(partial))

    def test_independent_frequency_effects_do_not_assume_sample_alignment(self):
        module = _load("build_stage_allocation_report_independent", "2d/diagnostics/build_stage_allocation_report.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.csv"
            second = root / "second.csv"
            first.write_text("source_index,high_fraction\n2,0.1\n7,0.2\n", encoding="utf-8")
            second.write_text("source_index,high_fraction\n2,0.3\n7,0.5\n", encoding="utf-8")
            effects = module.independent_csv_effects(first, second, "test")
            self.assertEqual(len(effects), 1)
            self.assertAlmostEqual(effects[0]["mean_continued_minus_original"], 0.25)
            self.assertEqual(effects[0]["n_original"], 2)
            self.assertEqual(effects[0]["n_continued"], 2)

            second.write_text("source_index,high_fraction\n7,0.5\n2,0.3\n", encoding="utf-8")
            effects = module.independent_csv_effects(first, second, "misaligned")
            self.assertEqual(len(effects), 1)
            self.assertAlmostEqual(effects[0]["mean_continued_minus_original"], 0.25)

    def test_report_accepts_every_producer_schema(self):
        module = _load("build_stage_allocation_report_integration", "2d/diagnostics/build_stage_allocation_report.py")

        def write_json(path: Path, payload) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")

        def write_metrics(path: Path, fid: float) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                f"Inception Score: 100\nFID: {fid}\nsFID: 5\nPrecision: 0.7\nRecall: 0.6\n",
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            diagnostics = root / "eval_results" / "diagnostics"
            bundle = root / "bundle"
            write_metrics(
                diagnostics / "full_metrics" / "ms100300600_400k_808090_ddpm_cfg1_noisefilling.txt",
                17.3,
            )
            write_metrics(
                diagnostics / "full_metrics" / "ms100300600_400k_808090_ddpm_cfg15_noisefilling.txt",
                5.04,
            )
            write_metrics(
                root / "eval_results" / "ms100300600_8x8continued600k_16x32orig400k_808090_ddpm_cfg1_noisefilling.txt",
                16.8,
            )
            write_metrics(
                root / "eval_results" / "ms100300600_8x8continued600k_16x32orig400k_808090_ddpm_cfg15_noisefilling.txt",
                4.97,
            )

            training_root = diagnostics / "training_losses"
            write_json(
                training_root / "training_loss_summary.json",
                {
                    "summaries": [
                        {
                            "run": "RDD_100_300_600",
                            "resolution": 8,
                            "first_step": 100,
                            "final_step": 400_000,
                            "previous_window_mean": 0.055,
                            "last_window_mean": 0.054,
                            "relative_last_minus_previous": -0.01,
                            "linear_slope_over_last_2windows_per_100k_steps": -0.001,
                        },
                        {
                            "run": "RDD_100_300_600_8x8_continuation",
                            "resolution": 8,
                            "first_step": 400_100,
                            "final_step": 600_000,
                            "previous_window_mean": 0.054,
                            "last_window_mean": 0.054,
                            "relative_last_minus_previous": 0.0,
                            "linear_slope_over_last_2windows_per_100k_steps": 0.0,
                        },
                    ]
                },
            )
            (training_root / "training_loss_records.csv").write_text(
                "run,resolution,step,loss\ntest,8,1,0.1\n", encoding="utf-8"
            )
            (training_root / "training_loss_convergence.png").write_bytes(b"fake-png")

            frequency_labels = (
                "ImageNet_original",
                "RDD_100_300_600_cfg1",
                "RDD_100_300_600_cfg1p5",
                "RDD_100_300_600_8x8continued600K_cfg1",
                "RDD_100_300_600_8x8continued600K_cfg1p5",
            )
            frequency_metrics = {
                key: {"n": 10, "mean": value, "std": 0.01, "sem": 0.003}
                for key, value in {
                    "low64_fraction": 0.5,
                    "mid_fraction": 0.3,
                    "high_fraction": 0.2,
                    "gradient_energy": 0.1,
                    "laplacian_filter_energy": 0.05,
                }.items()
            }
            write_json(
                diagnostics / "stage_allocation" / "summary.json",
                {
                    "datasets": [
                        {
                            "label": label,
                            "analyzed_images": 10,
                            "metrics": frequency_metrics,
                        }
                        for label in frequency_labels
                    ]
                },
            )
            write_json(
                diagnostics / "vae_frequency_control" / "analysis" / "summary.json",
                {
                    "datasets": [
                        {
                            "label": label,
                            "analyzed_images": 10,
                            "metrics": frequency_metrics,
                        }
                        for label in ("ImageNet_original", "ImageNet_SD_VAE_mean_reconstruction")
                    ]
                },
            )

            lowpass = diagnostics / "stage_allocation" / "lowpass_metrics"
            for scale in (64, 128):
                for label in frequency_labels[1:]:
                    write_metrics(lowpass / f"{label}_lowpass{scale}_10k.txt", 10.0 + scale / 100)

            write_json(
                diagnostics / "stage_pred_x0" / "300_vs_100" / "metadata.json",
                {"probe_times": [100, 300, 400, 600], "captures": []},
            )
            teacher_rows = []
            for index, label in enumerate(
                ("RDD_300_300_400", "RDD_100_300_600", "RDD_100_300_600_8x8continued600K")
            ):
                teacher_rows.append(
                    {
                        "experiment": label,
                        "global_t": 400,
                        "resolution": 16 if index == 0 else 8,
                        "stage_interval": "test",
                        "model_excess_mse": 0.1 + index * 0.05,
                        "unavoidable_representation_mse": 0.2 + index * 0.05,
                        "common_32_mse": 0.3 + index * 0.1,
                        "stage_model_nmse": 0.4 + index * 0.1,
                    }
                )
            write_json(
                diagnostics / "teacher_forced_pred_x0" / "300_vs_100" / "teacher_forced_summary.json",
                {"summary": teacher_rows},
            )

            for allocation in (
                "RDD_300_300_400",
                "RDD_100_300_600",
                "RDD_100_300_600_8x8continued600K",
            ):
                for solver in ("ddpm", "dpm2"):
                    write_json(
                        diagnostics / "local_solver_error" / f"{allocation}_{solver}.json",
                        {
                            "difficulty_sampling_order": [0.1, 0.2, 0.3],
                            "difficulty_ratio_sampling_order": [1.0, 2.0, 3.0],
                        },
                    )

            class_datasets = []
            for label in frequency_labels[1:]:
                class_datasets.append(
                    {
                        "label": label,
                        "top1_accuracy": 0.4,
                        "top5_accuracy": 0.7,
                        "mean_target_probability": 0.3,
                        "mean_target_nll": 2.0,
                        "predicted_class_coverage": 900,
                        "predicted_class_entropy_fraction_of_log1000": 0.95,
                    }
                )
            write_json(
                diagnostics / "class_consistency" / "100_300_600" / "class_consistency_summary.json",
                {"datasets": class_datasets},
            )

            original_argv = sys.argv
            try:
                sys.argv = [
                    "build_stage_allocation_report.py",
                    "--repo-root",
                    str(root),
                    "--bundle-dir",
                    str(bundle),
                ]
                module.main()
            finally:
                sys.argv = original_argv

            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(all(manifest["completion"].values()), manifest["completion"])
            report = (bundle / manifest["report"]).read_text(encoding="utf-8")
            self.assertIn("teacher-forced t=400", report)
            self.assertIn("像素频率", report)


if __name__ == "__main__":
    unittest.main()
