import gradio as gr
import os
import sys
import json
import subprocess

# Add current path allowing module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_languages():
    lang_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "languages.json")
    if os.path.exists(lang_file):
        try:
            with open(lang_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"auto": "Auto Detect", "kor": "Korean", "eng": "English"}

def load_environments():
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "environments.json")
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "main": sys.executable, "inference": sys.executable, "separation": sys.executable, 
        "diarization": sys.executable, "asr": sys.executable, "pipeline": sys.executable
    }

# Pre-load core models dictionary (offline mode prevents unwanted downloads during init)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
try:
    import runner
    MODELS_INFO = runner.load_models_info()
except Exception as e:
    print(f"Failed to load runner models: {e}")
    MODELS_INFO = {}
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

def get_installed_models():
    installed = []
    for name, data in MODELS_INFO.items():
        path = data.get('target_position')
        if path and os.path.exists(path):
            installed.append(name)
    return installed

def get_all_models():
    return list(MODELS_INFO.keys())

def generate_models_config(selected_names):
    models_config = []
    for name in selected_names:
        data = MODELS_INFO.get(name)
        if data:
            models_config.append({
                "model_name": name,
                "model_path": data.get('target_position'),
                "model_type": data.get('model_type'),
                "model_class": data.get('model_class', '')
            })
    return models_config

def run_process_generator(task_type, config_dict):
    """Executes runner.py exactly like main_gui.py and yields realtime logs to Gradio"""
    envs = load_environments()
    env_key = task_type
    
    if env_key == "inference": env_key = "separation"
    if env_key in ["pipeline", "download"]: env_key = "main"
    if env_key == "clean": env_key = "asr"
    
    python_exe = envs.get(env_key, sys.executable)
    if not os.path.exists(python_exe) and python_exe != "python":
        python_exe = sys.executable

    runner_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")
    config_str = json.dumps(config_dict)
    
    cmd = [python_exe, "-u", runner_script, "json_api", "--task", task_type, "--config", config_str]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    logs = f"--- Executing {task_type.upper()} Task ---\n"
    yield logs
    
    for line in iter(process.stdout.readline, ''):
        if not line: break
        line = line.strip()
        if line.startswith("PROGRESS_UPDATE:"):
            # main_gui.py catches this to draw a progress bar. We skip it in logs to keep it clean.
            pass
        else:
            logs += line + "\n"
            yield logs

    process.wait()
    if process.returncode != 0:
        logs += f"\n❌ Process failed with exit code {process.returncode}\n"
    else:
        logs += "\n✅ Task Completed Successfully!\n"
    yield logs

# === Event Handlers ===

def start_download(selected_model):
    if not selected_model:
        yield "Error: No model selected."
        return
    config = {"model": selected_model}
    for logs in run_process_generator("download", config):
        yield logs

def start_inference(input_files, output_dir, selected_models, output_format, device, normalize, use_tta, ensemble_mode):
    inputs = [i.strip() for i in input_files.split(";") if i.strip()]
    if not inputs:
        yield "Error: Input path is empty."
        return
    if not selected_models:
        yield "Error: No models selected."
        return

    models_config = generate_models_config(selected_models)
    
    # Pre-flight check
    for m in models_config:
        if not m["model_path"] or not os.path.exists(m["model_path"]):
            yield f"Error: Model {m['model_name']} is not installed. Download it in the Model Manager tab."
            return

    config = {
        "input": input_files,
        "models": models_config,
        "ensemble": ensemble_mode,
        "output_dir": output_dir,
        "device": device,
        "output_format": output_format,
        "normalize": normalize,
        "use_tta": use_tta
    }
    for logs in run_process_generator("inference", config):
        yield logs

def start_diarization(input_files, output_dir, phase1_batch, device, sr_steps, process_overlap):
    if not input_files.strip():
        yield "Error: Input path is empty."
        return
    config = {
        "input": input_files,
        "output_dir": output_dir,
        "device": device,
        "batch_size": int(phase1_batch),
        "audiosr_steps": int(sr_steps),
        "process_overlap": process_overlap
    }
    for logs in run_process_generator("diarization", config):
        yield logs

def start_asr(input_files, output_dir, device, lang, model_type):
    if not input_files.strip():
        yield "Error: Input path is empty."
        return
    config = {
        "input": input_files,
        "output_dir": output_dir,
        "device": device,
        "language": lang,
        "model_type": model_type
    }
    for logs in run_process_generator("asr", config):
        yield logs

def start_auto_pipeline(input_files, output_dir, selected_models, ensemble_mode, asr_lang, auto_asr_model):
    if not input_files.strip():
        yield "Error: Input path is empty."
        return
    if not selected_models:
        yield "Error: No separation models selected."
        return
        
    models_config = generate_models_config(selected_models)
    for m in models_config:
        if not m["model_path"] or not os.path.exists(m["model_path"]):
            yield f"Error: Model {m['model_name']} is not installed. Download it in the Model Manager tab."
            return

    config = {
        "input_file": input_files,
        "output_dir": output_dir,
        "sep_config": {
            "models": models_config,
            "ensemble": ensemble_mode
        },
        "asr_config": {
            "language": asr_lang,
            "model_type": auto_asr_model
        }
    }
    for logs in run_process_generator("pipeline", config):
        yield logs

def start_clean(target_dir, min_dur):
    if not target_dir.strip():
        yield "Error: Target directory is empty."
        return
    config = {
        "target_dir": target_dir.strip(),
        "min_duration": float(min_dur)
    }
    for logs in run_process_generator("clean", config):
        yield logs


# === Gradio Layout Mirroring main_gui.py ===

with gr.Blocks(title="VoxUnravel Web UI") as demo:
    gr.Markdown("# 🎧 VoxUnravel - Advanced Web Engine")
    gr.Markdown("Fully mirrors `main_gui.py` functionalities and environments inside the browser.")
    
    lang_dict = load_languages()
    lang_choices = [(f"{name} ({code})", code) for code, name in lang_dict.items()]
    
    with gr.Row():
        # Left Panel (Settings)
        with gr.Column(scale=3):
            with gr.Tabs():
                # 1. Separation
                with gr.TabItem("Separation"):
                    sep_input = gr.Textbox(label="Input (File path or Folder)", placeholder="/content/audio.mp4 (Use ';' for multiple files)")
                    sep_output = gr.Textbox(label="Output Directory", value="results")
                    sep_models = gr.Dropdown(choices=get_all_models(), multiselect=True, label="Select Model(s)")
                    
                    with gr.Row():
                        sep_format = gr.Dropdown(choices=["wav", "flac", "mp3"], value="wav", label="Format")
                        sep_device = gr.Dropdown(choices=["auto", "cuda", "cpu"], value="auto", label="Device")
                    
                    with gr.Row():
                        sep_norm = gr.Checkbox(label="Normalize Output", value=False)
                        sep_tta = gr.Checkbox(label="Test Time Augmentation (TTA)", value=False)
                        sep_ensemble = gr.Checkbox(label="Ensemble Mode (Average Selected Models)", value=False)
                    
                    sep_btn = gr.Button("START SEPARATION", variant="primary")
                    
                # 2. Speaker Diarization
                with gr.TabItem("Speaker Diarization"):
                    dia_input = gr.Textbox(label="Input (File path or Folder)", placeholder="/content/audio.wav")
                    dia_output = gr.Textbox(label="Output Directory", value="output_diarization")
                    
                    with gr.Row():
                        dia_batch = gr.Dropdown(choices=["4", "8", "16", "32"], value="8", label="Phase 1 Batch Size")
                        dia_device = gr.Dropdown(choices=["auto", "cuda", "cpu"], value="auto", label="Device")
                    
                    dia_sr_steps = gr.Textbox(label="AudioSR Steps (Diarization Stage 4)", value="50")
                    dia_overlap = gr.Checkbox(label="Process & Keep Overlapping Segments", value=True)
                    
                    dia_btn = gr.Button("START DIARIZATION", variant="primary")
                    
                # 3. ASR
                with gr.TabItem("ASR (Speech-to-Text)"):
                    asr_input = gr.Textbox(label="Input (File path or Folder)", placeholder="/content/audio.wav")
                    asr_output = gr.Textbox(label="Output Directory", value="output_asr")
                    
                    with gr.Row():
                        asr_device = gr.Dropdown(choices=["auto", "cuda", "cpu"], value="auto", label="Device")
                        asr_lang = gr.Dropdown(choices=lang_choices, value="auto", label="Language")
                        asr_model = gr.Dropdown(choices=[("Whisper Large v3", "whisper"), ("OWSM-CTC v4", "owsmv4")], value="whisper", label="Model")
                        
                    asr_btn = gr.Button("START ASR", variant="primary")
                
                # 4. Auto TTS Builder
                with gr.TabItem("Auto TTS Builder"):
                    gr.Markdown("🛠️ **Integrated Pipeline:** 1. Separation ➔ 2. Diarization ➔ 3. ASR")
                    auto_input = gr.Textbox(label="Input Audio (File path or Folder)", placeholder="/content/audio.mp4")
                    auto_output = gr.Textbox(label="Output Root Directory", value="tts_dataset_output")
                    auto_models = gr.Dropdown(choices=get_all_models(), multiselect=True, label="Select Separation Model(s)")
                    
                    with gr.Row():
                        auto_lang = gr.Dropdown(choices=lang_choices, value="auto", label="ASR Language")
                        auto_asr_model = gr.Dropdown(choices=[("Whisper Large v3", "whisper"), ("OWSM-CTC v4", "owsmv4")], value="whisper", label="ASR Model")
                    
                    auto_ensemble = gr.Checkbox(label="Enable Ensemble Mode for Separation Step", value=True)
                    
                    auto_btn = gr.Button("BUILD TTS DATASET", variant="primary")
                    
                # 5. Clean Dataset
                with gr.TabItem("Clean Dataset"):
                    clean_input = gr.Textbox(label="Target Directory", placeholder="/content/tts_dataset_output/Speaker_00")
                    clean_dur = gr.Number(label="Minimum Duration (sec)", value=1.0)
                    
                    clean_btn = gr.Button("CLEAN DATASET", variant="primary")
                    
                # 6. Model Manager (Replaces left-pane of GUI)
                with gr.TabItem("Model Manager"):
                    gr.Markdown("Download separation models to be used in the engine.")
                    dl_model = gr.Dropdown(choices=get_all_models(), label="Select Model to Download")
                    dl_btn = gr.Button("DOWNLOAD MODEL", variant="secondary")

        # Right Panel (Logs)
        with gr.Column(scale=2):
            logs_view = gr.Textbox(label="Console / Real-time Logs", lines=25, max_lines=40, interactive=False)

    # Attach Handlers
    dl_btn.click(fn=start_download, inputs=[dl_model], outputs=logs_view)
    sep_btn.click(fn=start_inference, inputs=[sep_input, sep_output, sep_models, sep_format, sep_device, sep_norm, sep_tta, sep_ensemble], outputs=logs_view)
    dia_btn.click(fn=start_diarization, inputs=[dia_input, dia_output, dia_batch, dia_device, dia_sr_steps, dia_overlap], outputs=logs_view)
    asr_btn.click(fn=start_asr, inputs=[asr_input, asr_output, asr_device, asr_lang, asr_model], outputs=logs_view)
    auto_btn.click(fn=start_auto_pipeline, inputs=[auto_input, auto_output, auto_models, auto_ensemble, auto_lang, auto_asr_model], outputs=logs_view)
    clean_btn.click(fn=start_clean, inputs=[clean_input, clean_dur], outputs=logs_view)

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
