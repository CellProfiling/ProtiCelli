from __future__ import annotations

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HTML = (ROOT / "proticelli_web" / "static" / "index.html").read_text(encoding="utf-8")
JAVASCRIPT = (ROOT / "proticelli_web" / "static" / "app.js").read_text(encoding="utf-8")
API = (ROOT / "proticelli_web" / "app.py").read_text(encoding="utf-8")
STYLES = (ROOT / "proticelli_web" / "static" / "styles.css").read_text(encoding="utf-8")


class WebFrontendContractTests(unittest.TestCase):
    def test_biological_default_colors_and_picker(self):
        expected = {
            "microtubules": "#ff4057",
            "er": "#ffd23f",
            "nucleus": "#3d7eff",
            "prediction": "#46f28d",
        }
        for channel, color in expected.items():
            self.assertRegex(JAVASCRIPT, rf'{channel}: "{re.escape(color)}"')
        self.assertIn('id="channelColor" type="color"', HTML)

    def test_ensemble_is_numeric_and_capped_at_fifty(self):
        self.assertIn('id="ensembleInput" type="number" min="1" max="50" step="1" value="1"', HTML)
        self.assertRegex(API, r"num_samples: int = Field\(default=1, ge=1, le=50\)")

    def test_simulation_heading_uses_staining_language(self):
        self.assertIn('<p class="eyebrow">Simulation</p>', HTML)
        self.assertIn('<h1>Simulating protein staining in cells</h1>', HTML)

    def test_selected_prediction_title_is_only_the_protein_name(self):
        self.assertIn("elements.viewerTitle.textContent = job.request.protein;", JAVASCRIPT)
        self.assertNotIn('in ${job.request.cell_line || "unconditioned cell"}', JAVASCRIPT)

    def test_compute_precision_defaults_to_float32_and_is_selectable(self):
        self.assertIn('id="dtypeSelect"', HTML)
        self.assertIn('<option value="float32" selected>', HTML)
        self.assertIn('<option value="float16">', HTML)
        self.assertIn('dtype: elements.dtypeSelect.value', JAVASCRIPT)
        self.assertRegex(API, r'dtype: Literal\["float32", "float16"\] = "float32"')

    def test_no_inaccurate_scale_or_percentage_progress(self):
        self.assertNotIn("scale-bar", HTML)
        self.assertNotIn("<progress", HTML)
        self.assertNotIn("progressValue", JAVASCRIPT)
        self.assertNotIn("Loading model", (ROOT / "proticelli_web" / "inference.py").read_text())

    def test_windows_one_click_launcher_is_present(self):
        launcher = ROOT / "proticelli-local.bat"
        self.assertTrue(launcher.is_file())
        text = launcher.read_text(encoding="utf-8")
        self.assertIn("-m venv .venv", text)
        self.assertIn('-m pip install -e ".[web]"', text)
        self.assertIn("-m proticelli.utils.download --check", text)
        self.assertIn("-m proticelli.utils.download", text)
        self.assertIn("PROTICELLI_SKIP_ASSET_DOWNLOAD", text)
        gpu_setup = ROOT / "proticelli-enable-nvidia.bat"
        self.assertTrue(gpu_setup.is_file())
        self.assertIn("download.pytorch.org/whl/cu126", gpu_setup.read_text(encoding="utf-8"))

    def test_macos_and_linux_launchers_and_accelerator_ui_are_present(self):
        unix_launcher = ROOT / "proticelli-local.sh"
        macos_launcher = ROOT / "proticelli-macos.command"
        linux_gpu_setup = ROOT / "proticelli-enable-nvidia.sh"
        self.assertTrue(unix_launcher.is_file())
        self.assertTrue(macos_launcher.is_file())
        self.assertTrue(linux_gpu_setup.is_file())
        self.assertIn("PROTICELLI_WEB_DEVICE:=auto", unix_launcher.read_text(encoding="utf-8"))
        launcher_text = unix_launcher.read_text(encoding="utf-8")
        self.assertIn("PYTORCH_ENABLE_MPS_FALLBACK:=1", launcher_text)
        self.assertIn("-m proticelli.utils.download --check", launcher_text)
        self.assertIn("PROTICELLI_SKIP_ASSET_DOWNLOAD", launcher_text)
        self.assertIn("python3.13 python3.12 python3.11 python3.10 python3.9", launcher_text)
        self.assertIn("/opt/homebrew/bin/python3", launcher_text)
        self.assertIn("/Library/Frameworks/Python.framework/Versions/Current/bin/python3", launcher_text)
        self.assertIn(
            '"$PROTICELLI_CONDA" create --yes --prefix "$PROTICELLI_ENV" python=3.12 pip',
            launcher_text,
        )
        self.assertIn('"$PROTICELLI_UV" python install 3.12', launcher_text)
        self.assertIn("download.pytorch.org/whl/cu126", linux_gpu_setup.read_text(encoding="utf-8"))
        self.assertIn("mps_ready", JAVASCRIPT)
        self.assertIn("rocm_ready", JAVASCRIPT)
        self.assertIn("Accelerator diagnostics", HTML)

    def test_gallery_brand_and_original_logo_are_used(self):
        self.assertIn("ProtiCelli Interactive Gallery", HTML)
        self.assertIn('src="/proticelli-logo.png"', HTML)
        self.assertTrue((ROOT / "proticelli_web" / "static" / "proticelli-logo.png").is_file())

    def test_phase_two_mapper_and_phase_three_collection_are_exposed(self):
        self.assertIn('id="fileInput" type="file" multiple', HTML)
        for extension in (".tif", ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"):
            self.assertIn(extension, HTML)
        self.assertIn("supported_input_extensions", API)
        self.assertIn('id="mappingGrid"', HTML)
        self.assertIn('id="uploadResample"', HTML)
        self.assertIn('id="uploadNormalize"', HTML)
        self.assertIn('id="normalizationBitDepth"', HTML)
        self.assertIn('normalize: elements.uploadNormalize.checked', JAVASCRIPT)
        self.assertIn('normalization_bit_depth:', JAVASCRIPT)
        self.assertRegex(API, r"normalize: bool = False")
        self.assertRegex(API, r"normalization_bit_depth: Literal\[8, 16\] \| None = None")
        self.assertIn('id="cropCanvas"', HTML)
        self.assertIn("centerCropWindow", JAVASCRIPT)
        self.assertNotIn('id="cropX"', HTML)
        self.assertNotIn('id="cropY"', HTML)
        self.assertIn('id="predictionGallery"', HTML)
        self.assertIn('id="galleryComparison"', HTML)
        self.assertIn("drawComparison", JAVASCRIPT)

    def test_workspace_navigation_and_unconditional_run_default(self):
        self.assertIn('data-tab="gallery"', HTML)
        self.assertIn('data-tab="analysis"', HTML)
        self.assertNotIn('data-tab="compare"', HTML)
        self.assertIn('id="activeRunSelect"', HTML)
        self.assertIn('id="newRunButton"', HTML)
        self.assertIn('unconditional.textContent = "Unconditional"', JAVASCRIPT)
        self.assertIn('id="predictionGallery"', HTML)
        self.assertIn('id="galleryComparison"', HTML)

    def test_run_rename_and_per_prediction_cell_line(self):
        self.assertIn('id="renameRunButton"', HTML)
        self.assertIn('id="renameRunDialog"', HTML)
        self.assertNotIn('id="newRunCellLine"', HTML)
        self.assertIn('cell_line: elements.cellLineSelect.value || null', JAVASCRIPT)
        self.assertIn('run.conditions?.join(", ")', JAVASCRIPT)

    def test_runs_can_be_deleted_with_confirmation_and_cleanup(self):
        self.assertIn('id="deleteRunDialog"', HTML)
        self.assertIn('id="confirmDeleteRun"', HTML)
        self.assertIn("The shared reference input is kept", HTML)
        self.assertIn('remove.textContent = "Delete"', JAVASCRIPT)
        self.assertIn('method: "DELETE"', JAVASCRIPT)
        self.assertIn('@app.delete("/api/runs/{run_id}")', API)
        self.assertIn("deleted_predictions", API)

    def test_prediction_blend_is_replaced_by_prediction_lut(self):
        self.assertNotIn('id="predictionBlend"', HTML)
        self.assertNotIn("predictionBlend", JAVASCRIPT)
        self.assertIn('key: "prediction"', JAVASCRIPT)

    def test_runtime_diagnostics_are_exposed(self):
        self.assertIn('id="runtimeDialog"', HTML)
        self.assertIn("proticelli-web --diagnose", JAVASCRIPT)
        self.assertIn("pytorch_cpu_build", (ROOT / "proticelli_web" / "inference.py").read_text())

    def test_gallery_chrome_uses_italic_brand_and_warm_accent(self):
        warm_theme = STYLES[STYLES.index("/* ProtiCelli Interactive Gallery") :]
        self.assertIn("font-style: italic", warm_theme)
        self.assertIn("--accent: #96662f", warm_theme)
        self.assertNotIn("rgba(47, 99, 87", warm_theme)


if __name__ == "__main__":
    unittest.main()
