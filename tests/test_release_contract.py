import importlib, pkgutil, unittest
import traser
from traser.io import ROOT, read
from traser.experiments.replay import requirements


class ReleaseContract(unittest.TestCase):
    def test_every_public_module_imports_without_server_or_model_weights(self):
        for module in pkgutil.walk_packages(traser.__path__, traser.__name__ + "."):
            importlib.import_module(module.name)

    def test_current_cases_cannot_mix_old_weibo_configuration(self):
        a = read(ROOT / "configs/experiments/blm.json")
        b = read(ROOT / "configs/experiments/weibo.json")
        self.assertEqual(a["datasets"], ["blm_x"])
        self.assertEqual(b["datasets"], ["hk_weibo"])
        self.assertEqual(a["window_hours"], [6, 12, 24])
        self.assertEqual(b["window_hours"], [3, 6, 12])
        self.assertEqual(
            b["boundaries"]["hk_weibo"],
            [
                "2025-11-25T00:00:00Z",
                "2025-11-28T00:00:00Z",
                "2025-11-30T00:00:00Z",
                "2026-01-01T00:00:00Z",
            ],
        )

    def test_full_calendar_test_origin_counts(self):
        for d, hs, expected in [
            ("blm_x", [6, 12, 24], [11, 5, 2]),
            ("hk_weibo", [3, 6, 12], [235, 123, 63]),
        ]:
            for h, n in zip(hs, expected):
                ws = read(ROOT / f"assets/{d}/w{h}/calendar.json")["windows"]
                actual = sum(
                    a["phase"] == b["phase"] == "test"
                    and a["full_duration"]
                    and b["full_duration"]
                    and b["eligible_occurrences"] > 0
                    for a, b in zip(ws, ws[1:])
                )
                self.assertEqual(actual, n)

    def test_weibo_frozen_assets_are_complete_and_blm_missing_assets_explicit(self):
        _, paths = requirements("hk_weibo", ROOT / "assets")
        self.assertTrue(all(p.exists() for p in paths))
        _, paths = requirements("blm_x", ROOT / "assets")
        missing = [p.relative_to(ROOT / "assets").as_posix() for p in paths if not p.exists()]
        self.assertEqual(len(missing), 4)
        self.assertIn("blm_x/representation/coordinates.parquet", missing)

    def test_blm_public_dataset_has_only_native_string_id_csv_parts(self):
        m = read(ROOT / "data/blm/manifest.json")
        self.assertEqual(m["rows"], 5416377)
        self.assertEqual(sum(p["rows"] for p in m["parts"]), 5416377)
        self.assertEqual(len(m["parts"]), 6)
        self.assertEqual(m["dtype"], "string")


if __name__ == "__main__":
    unittest.main()
