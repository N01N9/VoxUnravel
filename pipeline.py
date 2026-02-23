import os
import json
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def _resolve_path(p):
    if not p or os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(SCRIPT_DIR, p))

class VoxUnravelPipeline:
    def __init__(self, device="auto", logger=None):
        import logging
        self.device = device
        self.logger = logger or logging.getLogger("vox_unravel_pipeline")
        
        env_file = os.path.join(SCRIPT_DIR, "data", "environments.json")
        self.env_dict = {}
        if os.path.exists(env_file):
            try:
                with open(env_file, "r", encoding="utf-8") as f:
                    self.env_dict = json.load(f)
            except Exception:
                pass

    def _get_python(self, task):
        import sys
        if task == "inference": task = "separation"
        return self.env_dict.get(task, sys.executable)
        
    def _run_subtask(self, task_type, config_dict, progress_callback=None):
        python_exe = self._get_python(task_type)
        if not os.path.exists(python_exe) and python_exe != "python":
            import sys
            python_exe = sys.executable
            
        runner_script = os.path.join(SCRIPT_DIR, "runner.py")
        cmd = [python_exe, "-u", runner_script, "json_api", "--task", task_type, "--config", json.dumps(config_dict)]
        
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        self.logger.info(f"Launching {task_type} via {python_exe}...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            startupinfo=startupinfo
        )

        for line in iter(process.stdout.readline, ''):
            if not line: break
            line = line.strip()
            if line.startswith("PROGRESS_UPDATE:"):
                try:
                    prog = int(line.split(":")[1])
                    if progress_callback: progress_callback(prog)
                except Exception:
                    pass
            else:
                self.logger.info(line)

        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Task {task_type} failed with exit code {process.returncode}")

    def run_full_process(self, input_audio, output_dir, sep_config=None, dia_config=None, asr_config=None, progress_callback=None):
        input_files = []
        if isinstance(input_audio, list):
            input_files = input_audio
        elif os.path.isdir(input_audio):
            input_files = [os.path.join(input_audio, f) for f in os.listdir(input_audio) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
        elif ";" in str(input_audio) or "," in str(input_audio):
            delim = ";" if ";" in str(input_audio) else ","
            input_files = [p.strip() for p in str(input_audio).split(delim) if p.strip()]
        else:
            input_files = [input_audio]

        if not input_files:
            self.logger.error("No valid input audio files found.")
            return

        self.logger.info(f"=== Starting Automated TTS Data Generation Pipeline ({len(input_files)} files) ===")

        for idx, current_audio in enumerate(input_files):
            file_name_clean = os.path.splitext(os.path.basename(current_audio))[0].replace(" ", "_")
            current_output = output_dir if len(input_files) == 1 else os.path.join(output_dir, file_name_clean)
            os.makedirs(current_output, exist_ok=True)
            
            self.logger.info(f"[{idx+1}/{len(input_files)}] Processing: {current_audio} -> {current_output}")
            
            def sub_progress(val):
                if progress_callback:
                    base = (idx / len(input_files)) * 100
                    scale = (1 / len(input_files))
                    progress_callback(int(base + val * scale))

            self._run_single_file(current_audio, current_output, file_name_clean, sep_config, dia_config, asr_config, progress_callback=sub_progress)

    def _run_single_file(self, input_audio, output_dir, file_name_clean, sep_config, dia_config, asr_config, progress_callback=None):
        
        # 1. Separation
        vocals_path = None
        if sep_config and sep_config.get("models"):
            self.logger.info("[Step 1] Running Separation environment...")
            is_ensemble = sep_config.get("ensemble", True)
            sep_cfg = {
                "input": input_audio,
                "models": sep_config["models"],
                "ensemble": is_ensemble,
                "output_dir": output_dir,
                "device": self.device,
                "output_format": "wav"
            }
            
            self._run_subtask("inference", sep_cfg, lambda p: progress_callback(int(p*0.2)) if progress_callback else None)
            
            # Find the output vocal file
            instrs = ["vocals", "Vocals"]
            for target in instrs:
                if is_ensemble:
                    vp = os.path.join(output_dir, f"{file_name_clean}_{target}_ensemble.wav")
                else:
                    m_name = sep_cfg["models"][0]["model_name"].replace(" ", "_")
                    vp = os.path.join(output_dir, m_name, f"{file_name_clean}_{target}.wav") if len(sep_cfg["models"]) > 1 else os.path.join(output_dir, f"{file_name_clean}_{target}.wav")
                if os.path.exists(vp):
                    vocals_path = vp
                    break
                    
            if not vocals_path:
                self.logger.error("Could not find extracted vocals output file. Falling back to original audio.")
                vocals_path = input_audio
        else:
            self.logger.warning("No separation config provided. Using input audio directly for diarization.")
            vocals_path = input_audio
            
        if progress_callback: progress_callback(20)

        # 2. Diarization
        self.logger.info("[Step 2] Performing Speaker Diarization...")
        diarizer_out = os.path.join(output_dir, "diarization")
        
        batch_size = 8
        audiosr_steps = 50
        process_overlap = True
        
        if dia_config:
            batch_size = dia_config.get("batch_size", batch_size)
            audiosr_steps = dia_config.get("audiosr_steps", audiosr_steps)
            process_overlap = dia_config.get("process_overlap", process_overlap)

        dia_cfg = {
            "input": vocals_path,
            "output_dir": diarizer_out,
            "device": self.device,
            "batch_size": batch_size,
            "audiosr_steps": audiosr_steps,
            "process_overlap": process_overlap
        }
        
        self._run_subtask("diarization", dia_cfg, lambda p: progress_callback(20 + int(p*0.3)) if progress_callback else None)
        if progress_callback: progress_callback(50) 
        
        # 3. ASR
        self.logger.info("[Step 3] Performing ASR on all speaker segments...")
        asr_out = os.path.join(output_dir, "asr_results")
        os.makedirs(asr_out, exist_ok=True)
        
        asr_lang = asr_config.get("language", "auto") if asr_config else "auto"
        asr_model_type = asr_config.get("model_type", "whisper") if asr_config else "whisper"
        
        asr_cfg = {
            "input": diarizer_out,
            "output_dir": asr_out,
            "device": self.device,
            "language": asr_lang,
            "model_type": asr_model_type
        }
        
        self._run_subtask("asr", asr_cfg, lambda p: progress_callback(50 + int(p*0.4)) if progress_callback else None)
        
        # 4. Metadata Mapping (Because ASR environment writes to pure text, 
        # we need to collect the transcriptions to form the metadata CSV)
        self.logger.info("[Step 4] Generating TTS Metadata...")
        
        data_rows = []
        for root, dirs, files in os.walk(diarizer_out):
            for file in files:
                if file.endswith(".wav"):
                    spk_id = os.path.basename(root)
                    audio_path = os.path.join(root, file)
                    
                    # Target ASR text file: asr_out / spk_id / final_transcription.txt or similar.
                    # Wait, ASRInfer generates something like result.txt...
                    # Oh, run_asr generates specific output structure. We can just read the generated text files.
                    rel_dir = os.path.relpath(root, diarizer_out)
                    file_name = os.path.splitext(file)[0].replace(" ", "_")
                    txt_path = os.path.join(asr_out, rel_dir, f"{file_name}.txt")
                    
                    text = ""
                    if os.path.exists(txt_path):
                        with open(txt_path, "r", encoding="utf-8") as f:
                            text = f.read().strip()
                    
                    data_rows.append({
                        "file_path": audio_path,
                        "speaker_id": spk_id,
                        "transcription": text
                    })
        
        if progress_callback: progress_callback(90)
        
        import csv
        metadata_path = os.path.join(output_dir, "metadata.csv")
        with open(metadata_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["file_path", "speaker_id", "transcription"], delimiter="|")
            writer.writeheader()
            for row in data_rows:
                writer.writerow(row)
        
        with open(os.path.join(output_dir, "list.txt"), "w", encoding="utf-8") as f:
            for row in data_rows:
                rel_path = os.path.relpath(row["file_path"], output_dir)
                f.write(f"{rel_path}|{row['transcription']}\n")

        self.logger.info(f"=== Pipeline Completed! Results in {output_dir} ===")
        if progress_callback: progress_callback(100)
        return metadata_path