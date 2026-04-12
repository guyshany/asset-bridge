"""Streamlit UI — Upload, Jobs, Run, Results, Export."""

from __future__ import annotations

import asyncio
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Asset Bridge", page_icon="💎", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
PUBLISH_DIR = PROJECT_ROOT / "publish"
CONFIG_PATH = PROJECT_ROOT / "config" / "pipeline.yaml"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Lazy imports (avoid loading heavy deps at module scope)
# ---------------------------------------------------------------------------

def _load_config():
    from asset_bridge.pipeline import PipelineConfig
    if CONFIG_PATH.exists():
        return PipelineConfig.load(CONFIG_PATH)
    return PipelineConfig()


def _sku_status(sku_id: str) -> str:
    sku_output = OUTPUT_DIR / sku_id
    if not sku_output.exists():
        return "pending"
    sub = [d.name for d in sku_output.iterdir() if d.is_dir()]
    if "settings" in sub or "models" in sub:
        return "done"
    if "cleaned" in sub:
        return "partial"
    return "pending"


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

tab_upload, tab_jobs, tab_run, tab_experiment, tab_results, tab_export = st.tabs(
    ["Upload", "Jobs", "Run", "Experiment", "Results", "Export"]
)

# ========================== UPLOAD =========================================
with tab_upload:
    st.header("Upload Product Photos")

    col_sku, col_auto = st.columns([3, 1])
    with col_sku:
        sku_id = st.text_input("SKU ID", placeholder="e.g. AYNP70C")
    with col_auto:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Auto-generate ID"):
            sku_id = f"SKU-{datetime.now().strftime('%Y%m%d-%H%M')}"
            st.session_state["auto_sku"] = sku_id
    if "auto_sku" in st.session_state and not sku_id:
        sku_id = st.session_state["auto_sku"]

    uploaded_files = st.file_uploader(
        "Upload product photos (1-3 recommended)",
        type=["png", "jpg", "jpeg", "webp", "tiff"],
        accept_multiple_files=True,
    )

    disk_path = st.text_input(
        "Or enter a folder path on disk (optional)",
        placeholder="/Users/you/photos/product_001",
    )

    if uploaded_files and sku_id:
        dest = INPUT_DIR / sku_id
        dest.mkdir(parents=True, exist_ok=True)
        for f in uploaded_files:
            (dest / f.name).write_bytes(f.getvalue())
        st.success(f"Saved {len(uploaded_files)} file(s) to input/{sku_id}/")

    if disk_path and sku_id:
        src = Path(disk_path)
        if src.is_dir():
            dest = INPUT_DIR / sku_id
            if src.resolve() != dest.resolve():
                dest.mkdir(parents=True, exist_ok=True)
                for f in src.iterdir():
                    if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tiff"}:
                        shutil.copy2(f, dest / f.name)
                st.success(f"Copied files from {disk_path} to input/{sku_id}/")
        else:
            st.warning("Path does not exist or is not a directory.")

    if sku_id and (INPUT_DIR / sku_id).exists():
        st.subheader(f"Files in input/{sku_id}/")
        from asset_bridge.utils.image import collect_images
        imgs = collect_images(INPUT_DIR / sku_id)
        if imgs:
            cols = st.columns(min(len(imgs), 4))
            for i, img_path in enumerate(imgs):
                with cols[i % len(cols)]:
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)
        else:
            st.info("No images yet.")


# ========================== JOBS ===========================================
with tab_jobs:
    st.header("SKU Jobs")

    sku_dirs = sorted([d for d in INPUT_DIR.iterdir() if d.is_dir()]) if INPUT_DIR.exists() else []
    if not sku_dirs:
        st.info("No SKUs found in input/. Upload photos first.")
    else:
        for d in sku_dirs:
            status = _sku_status(d.name)
            icon = {"pending": "⏳", "partial": "🔄", "done": "✅"}.get(status, "❓")
            from asset_bridge.utils.image import collect_images
            n_inputs = len(collect_images(d))
            n_outputs = 0
            out = OUTPUT_DIR / d.name
            if out.exists():
                for sub in out.rglob("*"):
                    if sub.is_file() and sub.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                        n_outputs += 1
            st.markdown(f"{icon} **{d.name}** — {n_inputs} input(s), {n_outputs} output(s) — _{status}_")


# ========================== RUN ============================================
with tab_run:
    st.header("Run Pipeline")

    config = _load_config()

    from asset_bridge.providers.budget_guard import BudgetGuard
    guard = BudgetGuard(PROJECT_ROOT, daily_cap=config.free_tier.get("daily_request_cap", 50))

    col_profile, col_warn, col_budget = st.columns([1, 2, 1])
    with col_profile:
        profile = st.selectbox(
            "Billing Profile",
            ["experiment", "production"],
            index=0 if config.billing_profile == "experiment" else 1,
        )
    with col_warn:
        if profile == "production":
            st.warning("Production uses paid OpenAI for model & settings stages.")
        else:
            st.info("Experiment: $0 — Gemini free tier only, no OpenAI.")
    with col_budget:
        remaining = guard.remaining
        st.metric("Gemini calls left today", remaining, delta=None)
        if remaining <= 5:
            st.error(f"Only {remaining} API calls left today!")
        elif remaining <= 15:
            st.warning(f"{remaining} calls remaining.")

    sku_dirs = sorted([d.name for d in INPUT_DIR.iterdir() if d.is_dir()]) if INPUT_DIR.exists() else []
    selected_skus = st.multiselect("Select SKUs", sku_dirs, default=sku_dirs)

    st.subheader("Stages")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        do_cleanup = st.checkbox("Cleanup", value=True)
    with col2:
        do_color = st.checkbox("Color Variants", value=True)
    with col3:
        do_model = st.checkbox("Model Shots", value=True)
    with col4:
        do_settings = st.checkbox("Settings Shots", value=True)

    stages_list = []
    if do_cleanup:
        stages_list.append("cleanup")
    if do_color:
        stages_list.append("color")
    if do_model:
        stages_list.append("model")
    if do_settings:
        stages_list.append("settings")

    colors = st.multiselect(
        "Metal Colors",
        ["yellow_gold", "white_gold", "rose_gold"],
        default=config.metal_colors,
    )

    if st.button("Run Pipeline", type="primary", disabled=not selected_skus or not stages_list):
        config.billing_profile = profile
        config.metal_colors = colors

        progress_container = st.empty()
        log_expander = st.expander("Logs", expanded=True)
        log_lines: list[str] = []

        def _progress_cb(p):
            log_lines.append(f"[{p.stages_done}/{p.stages_total}] {p.message}")

        def _run():
            asyncio.run(
                __import__("asset_bridge.pipeline", fromlist=["run_pipeline"]).run_pipeline(
                    selected_skus,
                    stages_list,
                    project_root=PROJECT_ROOT,
                    config=config,
                    progress_callback=_progress_cb,
                )
            )

        t = threading.Thread(target=_run, daemon=True)
        t.start()

        while t.is_alive():
            time.sleep(0.5)
            with log_expander:
                st.text("\n".join(log_lines[-30:]))
            progress_container.progress(
                min(1.0, (len(log_lines) / max(1, len(selected_skus) * len(stages_list) * 2))),
                text=log_lines[-1] if log_lines else "Starting…",
            )

        t.join()
        progress_container.progress(1.0, text="Complete!")
        with log_expander:
            st.text("\n".join(log_lines))
        st.success(f"Pipeline finished for {len(selected_skus)} SKU(s).")
        st.rerun()


# ========================== EXPERIMENT =====================================
with tab_experiment:
    st.header("Multi-Method Experiment")
    st.markdown(
        "Run every available method for each pipeline stage, then compare outputs "
        "side-by-side and pick the winner."
    )

    exp_subtab_run, exp_subtab_compare, exp_subtab_lora = st.tabs(
        ["Run Experiment", "Compare & Pick", "Train LoRA"]
    )

    # ----- Run Experiment sub-tab -----
    with exp_subtab_run:
        sku_dirs_exp = sorted([d.name for d in INPUT_DIR.iterdir() if d.is_dir()]) if INPUT_DIR.exists() else []
        exp_skus = st.multiselect("Select SKUs", sku_dirs_exp, default=sku_dirs_exp, key="exp_skus")

        st.subheader("Stages to experiment")
        exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
        with exp_col1:
            exp_do_cleanup = st.checkbox("Cleanup", value=True, key="exp_cleanup")
        with exp_col2:
            exp_do_color = st.checkbox("Color Variants", value=True, key="exp_color")
        with exp_col3:
            exp_do_model = st.checkbox("Model Shots", value=True, key="exp_model")
        with exp_col4:
            exp_do_settings = st.checkbox("Settings Shots", value=True, key="exp_settings")

        exp_stages = []
        if exp_do_cleanup:
            exp_stages.append("cleanup")
        if exp_do_color:
            exp_stages.append("color")
        if exp_do_model:
            exp_stages.append("model")
        if exp_do_settings:
            exp_stages.append("settings")

        from asset_bridge.experiment import EXPERIMENT_METHODS
        st.markdown("**Methods that will be tested:**")
        for stg in exp_stages:
            methods = EXPERIMENT_METHODS.get(stg, [])
            st.markdown(f"- **{stg}**: {', '.join(methods)}")

        if st.button("Run Experiment", type="primary", disabled=not exp_skus or not exp_stages, key="run_exp"):
            config = _load_config()
            exp_progress_container = st.empty()
            exp_log_expander = st.expander("Experiment Logs", expanded=True)
            exp_log_lines: list[str] = []

            def _exp_progress_cb(p):
                exp_log_lines.append(f"[{p.steps_done}/{p.total_steps}] {p.message}")

            def _run_exp():
                from asset_bridge.experiment import run_experiment
                asyncio.run(run_experiment(
                    exp_skus,
                    exp_stages,
                    project_root=PROJECT_ROOT,
                    config=config,
                    progress_callback=_exp_progress_cb,
                ))

            t = threading.Thread(target=_run_exp, daemon=True)
            t.start()

            while t.is_alive():
                time.sleep(0.5)
                with exp_log_expander:
                    st.text("\n".join(exp_log_lines[-40:]))
                exp_progress_container.progress(
                    min(1.0, len(exp_log_lines) / max(1, len(exp_skus) * sum(
                        len(EXPERIMENT_METHODS.get(s, [])) for s in exp_stages
                    ))),
                    text=exp_log_lines[-1] if exp_log_lines else "Starting…",
                )

            t.join()
            exp_progress_container.progress(1.0, text="Experiment complete!")
            with exp_log_expander:
                st.text("\n".join(exp_log_lines))
            st.success(f"Experiment finished for {len(exp_skus)} SKU(s). Go to 'Compare & Pick' tab.")

    # ----- Compare & Pick sub-tab -----
    with exp_subtab_compare:
        sku_dirs_out_exp = []
        if OUTPUT_DIR.exists():
            for d in sorted(OUTPUT_DIR.iterdir()):
                if d.is_dir() and (d / "experiments").exists():
                    sku_dirs_out_exp.append(d.name)

        if not sku_dirs_out_exp:
            st.info("No experiment results yet. Run an experiment first.")
        else:
            compare_sku = st.selectbox("Select SKU", sku_dirs_out_exp, key="compare_sku")
            exp_root = OUTPUT_DIR / compare_sku / "experiments"

            from asset_bridge.experiment import load_picks, save_picks, apply_winners
            current_picks = load_picks(OUTPUT_DIR / compare_sku)

            stage_dirs = sorted([d.name for d in exp_root.iterdir() if d.is_dir() and not d.name.startswith("_")])

            for stage_dir_name in stage_dirs:
                stage_path = exp_root / stage_dir_name
                method_dirs = sorted([d for d in stage_path.iterdir() if d.is_dir()])

                if not method_dirs:
                    continue

                st.subheader(f"Stage: {stage_dir_name}")

                # Show input image for reference
                input_sku_dir = INPUT_DIR / compare_sku
                if input_sku_dir.exists():
                    from asset_bridge.utils.image import collect_images as _ci
                    input_imgs = _ci(input_sku_dir)
                    if input_imgs:
                        with st.expander("Original input", expanded=False):
                            ref_cols = st.columns(min(len(input_imgs), 3))
                            for i, ip in enumerate(input_imgs):
                                with ref_cols[i % len(ref_cols)]:
                                    st.image(str(ip), caption="Input", use_container_width=True)

                # Show each method side by side
                n_methods = len(method_dirs)
                cols = st.columns(n_methods)
                method_names = []

                for idx, method_dir in enumerate(method_dirs):
                    method_name = method_dir.name
                    method_names.append(method_name)
                    with cols[idx]:
                        st.markdown(f"**{method_name}**")
                        from asset_bridge.utils.image import collect_images as _ci2
                        method_images = _ci2(method_dir)
                        if method_images:
                            for mimg in method_images[:4]:
                                st.image(str(mimg), caption=mimg.name, use_container_width=True)
                        else:
                            st.warning("No outputs (skipped or failed)")

                # Winner picker
                current_winner = current_picks.get(stage_dir_name, "")
                winner = st.radio(
                    f"Winner for {stage_dir_name}",
                    method_names,
                    index=method_names.index(current_winner) if current_winner in method_names else 0,
                    key=f"pick_{compare_sku}_{stage_dir_name}",
                    horizontal=True,
                )
                current_picks[stage_dir_name] = winner
                st.divider()

            col_save, col_apply = st.columns(2)
            with col_save:
                if st.button("Save Picks", key="save_picks"):
                    save_picks(OUTPUT_DIR / compare_sku, current_picks)
                    st.success(f"Picks saved for {compare_sku}")

            with col_apply:
                if st.button("Apply Winners to Main Output", type="primary", key="apply_winners"):
                    save_picks(OUTPUT_DIR / compare_sku, current_picks)
                    applied = apply_winners(OUTPUT_DIR / compare_sku, current_picks)
                    st.success(f"Applied {len(applied)} files from winning methods to main output folders.")

    # ----- Train LoRA sub-tab -----
    with exp_subtab_lora:
        st.subheader("LoRA Training")

        refs_dir = PROJECT_ROOT / "references"
        dataset_dir = PROJECT_ROOT / "experiments" / "lora" / "datasets" / "default"
        weights_dir = PROJECT_ROOT / "experiments" / "lora" / "weights"
        lora_config_path = PROJECT_ROOT / "experiments" / "lora" / "train_config.yaml"

        # Stats
        col_refs, col_data, col_weights = st.columns(3)
        with col_refs:
            n_refs = 0
            for search_dir in [refs_dir / "train", refs_dir / "skus"]:
                if search_dir.exists():
                    n_refs = sum(1 for f in search_dir.rglob("*") if f.suffix.lower() in {".png", ".jpg", ".jpeg"})
                    break
            st.metric("Reference images", n_refs)
        with col_data:
            n_dataset = len(list(dataset_dir.glob("*.png"))) if dataset_dir.exists() else 0
            st.metric("Dataset images", n_dataset)
        with col_weights:
            has_weights = any(weights_dir.glob("*.safetensors")) if weights_dir.exists() else False
            st.metric("LoRA trained", "Yes" if has_weights else "No")

        # Config editor
        with st.expander("Training Config", expanded=False):
            if lora_config_path.exists():
                config_text = lora_config_path.read_text()
            else:
                config_text = (
                    "base_model: stabilityai/stable-diffusion-xl-base-1.0\n"
                    "resolution: 1024\ntrain_batch_size: 1\nmax_train_steps: 1000\n"
                    "learning_rate: 1e-4\nnetwork_rank: 32\nnetwork_alpha: 16\n"
                    "trigger_word: jewlstyle\noutput_name: jewelry_style\n"
                )
            edited_config = st.text_area("train_config.yaml", config_text, height=200, key="lora_config_edit")
            if st.button("Save Config", key="save_lora_config"):
                lora_config_path.parent.mkdir(parents=True, exist_ok=True)
                lora_config_path.write_text(edited_config)
                st.success("Config saved.")

        # Actions
        lora_col1, lora_col2 = st.columns(2)
        with lora_col1:
            if st.button("Prepare Dataset", key="prepare_dataset"):
                with st.spinner("Preparing dataset..."):
                    from asset_bridge.lora.dataset import prepare_dataset
                    import yaml as _yaml
                    lora_cfg = _yaml.safe_load(lora_config_path.read_text()) if lora_config_path.exists() else {}
                    count = prepare_dataset(
                        refs_dir, dataset_dir,
                        trigger_word=lora_cfg.get("trigger_word", "jewlstyle"),
                        resolution=lora_cfg.get("resolution", 1024),
                    )
                    st.success(f"Prepared {count} image+caption pairs.")
                    st.rerun()

        with lora_col2:
            if st.button("Start Training", key="start_training", disabled=n_dataset == 0):
                st.info(
                    "LoRA training typically requires kohya-ss/sd-scripts or a compatible trainer. "
                    "Run `asset-bridge train-lora` from the terminal for full output. "
                    "See experiments/lora/README.md for setup."
                )
                lora_log_area = st.empty()
                lora_log_lines: list[str] = []

                def _train():
                    import yaml as _yaml
                    from asset_bridge.lora.train import load_config as _lc, train_lora as _tl
                    cfg = _lc(lora_config_path)
                    _tl(dataset_dir, weights_dir, cfg)

                t = threading.Thread(target=_train, daemon=True)
                t.start()

                while t.is_alive():
                    time.sleep(2)
                    lora_log_area.text("Training in progress...")

                t.join()
                st.rerun()


# ========================== RESULTS ========================================
with tab_results:
    st.header("Results Gallery")

    from asset_bridge.ui.components.image_grid import image_grid, zip_download_button

    sku_dirs_out = sorted([d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()]) if OUTPUT_DIR.exists() else []
    if not sku_dirs_out:
        st.info("No outputs yet. Run the pipeline first.")
    else:
        selected_sku = st.selectbox("Select SKU", sku_dirs_out, key="results_sku")
        sku_out = OUTPUT_DIR / selected_sku

        zip_download_button(sku_out, label=f"Download all {selected_sku} outputs as ZIP", key="zip_all")

        stage_tabs = st.tabs(["Cleaned", "Yellow Gold", "White Gold", "Rose Gold", "Models", "Settings"])

        stage_folders = ["cleaned", "yellow_gold", "white_gold", "rose_gold", "models", "settings"]
        for tab, folder_name in zip(stage_tabs, stage_folders):
            with tab:
                folder = sku_out / folder_name
                if folder.exists():
                    image_grid(folder, columns=3, caption_prefix=f"{folder_name}: ")
                    zip_download_button(folder, label=f"Download {folder_name} ZIP", key=f"zip_{folder_name}")
                else:
                    st.info(f"No {folder_name} outputs yet.")

        # Before / After comparison for cleaned
        st.subheader("Before → After")
        cleaned_dir = sku_out / "cleaned"
        input_sku = INPUT_DIR / selected_sku
        if cleaned_dir.exists() and input_sku.exists():
            from asset_bridge.utils.image import collect_images
            befores = collect_images(input_sku)
            afters = [p for p in collect_images(cleaned_dir) if "_mask" not in p.stem]
            from asset_bridge.ui.components.image_grid import before_after
            for b, a in zip(befores, afters):
                before_after(b, a)


# ========================== EXPORT =========================================
with tab_export:
    st.header("Export / Publish")

    sku_dirs_out = sorted([d.name for d in OUTPUT_DIR.iterdir() if d.is_dir()]) if OUTPUT_DIR.exists() else []
    if not sku_dirs_out:
        st.info("No outputs to export.")
    else:
        export_skus = st.multiselect("Select SKUs to publish", sku_dirs_out, key="export_skus")

        if st.button("Publish Selected", type="primary", disabled=not export_skus):
            PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
            for sku in export_skus:
                src = OUTPUT_DIR / sku
                dst = PUBLISH_DIR / sku
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            st.success(f"Published {len(export_skus)} SKU(s) to {PUBLISH_DIR}/")
