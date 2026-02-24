import argparse
import os
import sys
import json
import logging
import platform
import subprocess
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("msst_runner")

from core.logger import set_log_level

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_INFO_PATH = os.path.join(SCRIPT_DIR, "data/models_info.json")
MODELS_INFO_BACKUP_PATH = os.path.join(SCRIPT_DIR, "data_backup/models_info.json")

def resolve_model_path(relative_path):
    if not relative_path:
        return relative_path
    if os.path.isabs(relative_path):
        return relative_path
    resolved = os.path.normpath(os.path.join(SCRIPT_DIR, relative_path))
    return resolved

def get_models_info_path():
    if os.path.exists(MODELS_INFO_PATH):
        return MODELS_INFO_PATH
    elif os.path.exists(MODELS_INFO_BACKUP_PATH):
        return MODELS_INFO_BACKUP_PATH
    else:
        logger.error("models_info.json not found in data or data_backup.")
        sys.exit(1)
        
def resolve_input_paths(input_val):
    if isinstance(input_val, list):
        return input_val
    if not input_val:
        return []
        
    if os.path.isdir(input_val):
        paths = []
        for root, _, files in os.walk(input_val):
            for f in files:
                if f.lower().endswith(('.wav', '.mp3', '.flac')):
                    paths.append(os.path.join(root, f))
        return sorted(paths)
        
    if ";" in str(input_val) or "," in str(input_val):
        delim = ";" if ";" in str(input_val) else ","
        raw_paths = [p.strip() for p in str(input_val).split(delim) if p.strip()]
        final_paths = []
        for p in raw_paths:
            if os.path.isdir(p):
                for root, _, files in os.walk(p):
                    for f in files:
                        if f.lower().endswith(('.wav', '.mp3', '.flac')):
                            final_paths.append(os.path.join(root, f))
            else:
                final_paths.append(p)
        return sorted(final_paths)
        
    return [input_val]

def load_models_info():
    path = get_models_info_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            info = json.load(f)
        for name, data in info.items():
            if "target_position" in data:
                data["target_position"] = resolve_model_path(data["target_position"])
            if "config_path" in data:
                data["config_path"] = resolve_model_path(data["config_path"])
        return info
    except Exception as e:
        logger.error(f"Failed to load models info from {path}: {e}")
        sys.exit(1)

def list_models():
    info = load_models_info()
    print(f"{'Model Name':<60} {'Type':<20} {'Size (MB)':<10}")
    print("-" * 90)
    for name, data in info.items():
        size_mb = f"{data.get('model_size', 0) / 1024 / 1024:.2f}"
        model_type = data.get('model_type', data.get('model_class', 'Unknown'))
        print(f"{name:<60} {model_type:<20} {size_mb:<10}")

def download_file(url, path, progress_callback=None):
    try:
        import urllib.request
        def reporthook(block_num, block_size, total_size):
            if progress_callback and total_size > 0:
                percent = int(block_num * block_size * 100 / total_size)
                progress_callback(min(percent, 100))
        
        urllib.request.urlretrieve(url, path, reporthook=reporthook)
        return True
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def download_model(model_name, progress_callback=None):
    info = load_models_info()
    if model_name not in info:
        logger.error(f"Model '{model_name}' not found in known models.")
        return

    model_data = info[model_name]
    target_path = model_data.get("target_position")
    link = model_data.get("link")

    if not target_path or not link:
        logger.error(f"Incomplete data for model '{model_name}'.")
        return

    model_dir = os.path.dirname(target_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(target_path):
        logger.info(f"Model '{model_name}' already exists at {target_path}.")
        if progress_callback: progress_callback(100)
        return

    logger.info(f"Downloading '{model_name}' from {link}...")
    if download_file(link, target_path, progress_callback=progress_callback):
        logger.info(f"Successfully downloaded '{model_name}' to {target_path}")
    else:
        logger.error(f"Failed to download '{model_name}'")

def run_inference(args, progress_callback=None):
    models_to_run = args.models if hasattr(args, "models") and args.models else []
    
    if not models_to_run and args.model_path:
        models_to_run = [{
            "model_name": args.model_name or "Custom",
            "model_path": args.model_path,
            "model_type": args.model_type,
            "config_path": args.config_path
        }]

    if not models_to_run:
        logger.error("No models specified for inference.")
        return

    audio_params = {
        "wav_bit_depth": args.wav_bit_depth,
        "flac_bit_depth": args.flac_bit_depth,
        "mp3_bit_rate": args.mp3_bit_rate
    }

    inference_params = {
        "batch_size": args.batch_size,
        "num_overlap": args.num_overlap,
        "chunk_size": args.chunk_size,
        "normalize": args.normalize
    }

    input_paths = resolve_input_paths(args.input)
    ensemble_mode = getattr(args, "ensemble", False)
    device_ids = [int(i.strip()) for i in args.device_ids.split(',')]
    
    import torch
    import librosa

    for idx, input_path in enumerate(input_paths):
        file_name, _ = os.path.splitext(os.path.basename(input_path))
        all_results_for_file = [] 
        
        for m_cfg in models_to_run:
            m_path = m_cfg["model_path"]
            m_type = m_cfg.get("model_type")
            m_name = m_cfg["model_name"]
            m_class = m_cfg.get("model_class", "")
            
            is_vr_model = (m_class == "VR_Models") or (not m_type and m_path.lower().endswith(".pth"))

            logger.info(f"Running model: {m_name} ({'VR' if is_vr_model else m_type}) on {input_path}")
            
            try:
                from core.separator import MSSeparator
                if is_vr_model:
                    from inference.vr_infer import VRSeparator as VRSep
                    
                    vr_params = {
                        "batch_size": int(args.batch_size) if args.batch_size else 4,
                        "window_size": 512,
                        "aggression": 5,
                        "enable_tta": args.use_tta,
                        "enable_post_process": False,
                        "post_process_threshold": 0.2,
                        "high_end_process": False
                    }
                    
                    separator = VRSep(
                        model_file=m_path,
                        output_dir=args.output_dir,
                        output_format=args.output_format,
                        use_cpu=(args.device == "cpu"),
                        vr_params=vr_params,
                        audio_params=audio_params,
                    )
                    
                    final_sr = 44100 
                    results = separator.separate(input_path)
                    
                    if ensemble_mode:
                        all_results_for_file.append(results)
                    else:
                        save_subdir = args.output_dir
                        if len(models_to_run) > 1:
                            save_subdir = os.path.join(args.output_dir, m_name.replace(" ", "_"))
                        os.makedirs(save_subdir, exist_ok=True)
                        for instr in results.keys():
                            separator.save_audio(results[instr], final_sr, f"{file_name}_{instr}", save_subdir)
                    
                    separator.del_cache()
                    del separator
                else:
                    c_path = m_cfg.get("config_path")
                    if not c_path:
                        c_path = m_path.replace("pretrain", "configs") + ".yaml"
                    c_path = resolve_model_path(c_path) if not os.path.isabs(c_path) else c_path

                    separator = MSSeparator(
                        model_type=m_type,
                        config_path=c_path,
                        model_path=m_path,
                        device=args.device,
                        device_ids=device_ids,
                        output_format=args.output_format,
                        use_tta=args.use_tta,
                        store_dirs=args.output_dir,
                        audio_params=audio_params,
                        inference_params=inference_params,
                        debug=args.debug
                    )
                    
                    final_sr = getattr(separator.config.audio, 'sample_rate', 44100)
                    mix, sr = librosa.load(input_path, sr=final_sr, mono=False)
                    
                    results = separator.separate(mix)
                    
                    if ensemble_mode:
                        all_results_for_file.append(results)
                    else:
                        save_subdir = args.output_dir
                        if len(models_to_run) > 1:
                            save_subdir = os.path.join(args.output_dir, m_name.replace(" ", "_"))
                        os.makedirs(save_subdir, exist_ok=True)
                        for instr in results.keys():
                            separator.save_audio(results[instr], final_sr, f"{file_name}_{instr}", save_subdir)
                    
                    separator.del_cache()
                    del separator
                
                import torch
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                
            except Exception as e:
                logger.error(f"Inference with model {m_name} failed: {e}", exc_info=True)
            
            if progress_callback:
                progress_callback(int((idx + 1) / len(input_paths) * 100))
            else:
                print(f"PROGRESS_UPDATE:{int((idx + 1) / len(input_paths) * 100)}", flush=True)

        if ensemble_mode and all_results_for_file:
            logger.info(f"Performing ensemble averaging for {file_name}...")
            instruments = all_results_for_file[0].keys()
            final_ensemble = {}
            for instr in instruments:
                waveforms = [res[instr] for res in all_results_for_file if instr in res]
                if waveforms:
                    final_ensemble[instr] = sum(waveforms) / len(waveforms)
            
            os.makedirs(args.output_dir, exist_ok=True)
            import soundfile as sf
            for instr, wave in final_ensemble.items():
                out_path = os.path.join(args.output_dir, f"{file_name}_{instr}_ensemble.wav")
                sf.write(out_path, wave, final_sr)
                logger.info(f"Saved ensemble {instr} to {out_path}")

    logger.info("Inference process completed.")

def run_diarization(config: dict, progress_callback=None):
    input_val = config.get("input")
    output_root = config.get("output_dir", "output_diarization")
    device = config.get("device", "cuda:0")
    import torch
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    input_files = resolve_input_paths(input_val)
    logger.info(f"Starting Speaker Diarization on {len(input_files)} files using {device}")
    
    for idx, input_file in enumerate(input_files):
        try:
            file_name = os.path.splitext(os.path.basename(input_file))[0].replace(" ", "_")
            current_output = output_root if len(input_files) == 1 else os.path.join(output_root, file_name)
            os.makedirs(current_output, exist_ok=True)
            
            logger.info(f"[{idx+1}/{len(input_files)}] Diarizing: {input_file} -> {current_output}")
            
            from core.diarizer import DiarizationInfer
            infer = DiarizationInfer(
                audio_path=input_file,
                output_dir=current_output,
                device=device,
                logger=logger
            )
            if "batch_size" in config and config["batch_size"]:
                infer.diarizen_batch_size = int(config["batch_size"])
            if "audiosr_steps" in config and config["audiosr_steps"]:
                infer.audiosr_steps = int(config["audiosr_steps"])
            if "process_overlap" in config:
                infer.process_overlap = config["process_overlap"]
                
            rttm_output = os.path.join(current_output, "final_refined.rttm")
            infer.run_all(rttm_output)
        except Exception as e:
            logger.error(f"Diarization failed for {input_file}: {e}")
        
        if progress_callback:
            progress_callback(int((idx + 1) / len(input_files) * 100))
        else:
            print(f"PROGRESS_UPDATE:{int((idx + 1) / len(input_files) * 100)}", flush=True)
    logger.info("Speaker Diarization completed successfully.")

def run_asr(config: dict, progress_callback=None):
    input_val = config.get("input")
    output_root = config.get("output_dir", "output_asr")
    device = config.get("device", "cuda:0")
    lang = config.get("language", "auto")
    model_type = config.get("model_type", "whisper")
    import torch
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    input_files = resolve_input_paths(input_val)
    logger.info(f"Starting ASR on {len(input_files)} files using {device}")
    
    input_base = None
    if isinstance(input_val, str) and ";" not in input_val and "," not in input_val:
        if os.path.exists(input_val):
            input_base = input_val if os.path.isdir(input_val) else os.path.dirname(input_val)

    for idx, input_file in enumerate(input_files):
        try:
            if input_base:
                rel_path = os.path.relpath(input_file, input_base)
                rel_dir = os.path.dirname(rel_path)
                current_output = os.path.join(output_root, rel_dir)
            else:
                file_name = os.path.splitext(os.path.basename(input_file))[0].replace(" ", "_")
                current_output = os.path.join(output_root, file_name)
            
            os.makedirs(current_output, exist_ok=True)
            
            logger.info(f"[{idx+1}/{len(input_files)}] ASR: {input_file} -> {current_output} (Lang: {lang})")
            
            from core.asr import ASRInfer
            infer = ASRInfer(
                audio_path=input_file,
                output_dir=current_output,
                device=device,
                language=lang,
                model_type=model_type,
                logger=logger
            )
            infer.run_all()
        except Exception as e:
            logger.error(f"ASR failed for {input_file}: {e}")
        
        if progress_callback:
            progress_callback(int((idx + 1) / len(input_files) * 100))
        else:
            print(f"PROGRESS_UPDATE:{int((idx + 1) / len(input_files) * 100)}", flush=True)
    logger.info("ASR completed successfully.")

def run_cleaner(config: dict, progress_callback=None):
    try:
        from core.cleaner import DatasetCleaner
        cleaner = DatasetCleaner(
            target_dir=config["target_dir"],
            min_duration=config.get("min_duration", 1.0),
            logger=logger
        )
        cleaner.run_all()
        if progress_callback:
            progress_callback(100)
        else:
            print("PROGRESS_UPDATE:100", flush=True)
        logger.info("Dataset cleaner finished successfully.")
    except Exception as e:
        logger.error(f"Cleaner failed: {e}")

def run_inference_api(config: dict, progress_callback=None):
    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)
            
    defaults = {
        "models": [],
        "ensemble": False,
        "model_path": None,
        "config_path": None,
        "model_name": None,
        "model_type": None,
        "input": None,
        "output_dir": "results",
        "device": "auto",
        "device_ids": "0",
        "output_format": "wav",
        "use_tta": False,
        "recursive": False,
        "debug": False,
        "batch_size": None,
        "num_overlap": None,
        "chunk_size": None,
        "normalize": False,
        "wav_bit_depth": "FLOAT",
        "flac_bit_depth": "PCM_24",
        "mp3_bit_rate": "320k"
    }
    
    final_config = defaults.copy()
    final_config.update(config)
    
    args = Args(**final_config)
    
    run_inference(args, progress_callback=progress_callback)

def add_custom_model(args):
    info = load_models_info()
    
    model_name = args.name
    model_path = os.path.abspath(args.path)
    model_type = args.type
    config_path = os.path.abspath(args.config) if args.config else None

    if not os.path.exists(model_path):
        logger.error(f"Model file not found at: {model_path}")
        return

    if config_path and not os.path.exists(config_path):
        logger.error(f"Config file not found at: {config_path}")
        return

    entry = {
        "model_name": model_name,
        "model_class": "custom",
        "model_type": model_type,
        "target_position": model_path,
        "is_installed": True,
        "link": ""
    }
    
    if config_path:
        entry["config_path"] = config_path

    if model_name in info:
        logger.info(f"Updating existing model entry '{model_name}'.")
    else:
        logger.info(f"Adding new model entry '{model_name}'.")

    info[model_name] = entry
    
    path = get_models_info_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4)
        logger.info(f"Model '{model_name}' successfully added/updated in {path}")
    except Exception as e:
        logger.error(f"Failed to save models info: {e}")

def main():
    parser = argparse.ArgumentParser(description="MSST Runner: Command Line Interface for Model Download and Inference", formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    dl_parser = subparsers.add_parser("download", help="Download models")
    dl_parser.add_argument("--list", action="store_true", help="List available models")
    dl_parser.add_argument("--model", type=str, help="Name of the model to download (see --list)")

    add_parser = subparsers.add_parser("add_model", help="Register a custom model")
    add_parser.add_argument("--name", type=str, required=True, help="Unique name for the model")
    add_parser.add_argument("--path", type=str, required=True, help="Path to the model checkpoint file")
    add_parser.add_argument("--type", type=str, required=True, help="Model architecture type")
    add_parser.add_argument("--config", type=str, help="Path to the config file")

    inf_parser = subparsers.add_parser("inference", help="Run inference")
    inf_parser.add_argument("--input", "-i", type=str, required=True, help="Input file or directory")
    inf_parser.add_argument("--output_dir", "-o", type=str, default="results", help="Output directory")
    inf_parser.add_argument("--model_name", "-n", type=str, help="Name of the model to use")
    inf_parser.add_argument("--model_path", "-p", type=str, help="Path to model file")
    inf_parser.add_argument("--config_path", "-c", type=str, help="Path to config file")
    inf_parser.add_argument("--model_type", "-t", type=str, help="Model type")
    
    inf_parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu", "mps"], help="Device to use")
    inf_parser.add_argument("--device_ids", type=str, default="0", help="Comma-separated device IDs")
    inf_parser.add_argument("--output_format", type=str, default="wav", choices=["wav", "flac", "mp3"], help="Output format")
    inf_parser.add_argument("--use_tta", action="store_true", help="Use Test Time Augmentation")
    inf_parser.add_argument("--recursive", "-r", action="store_true", help="Process input directory recursively")
    inf_parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    inf_parser.add_argument("--batch_size", type=int, help="Batch size for inference")
    inf_parser.add_argument("--num_overlap", type=int, help="Number of overlaps")
    inf_parser.add_argument("--chunk_size", type=int, help="Chunk size")
    inf_parser.add_argument("--normalize", action="store_true", help="Normalize output")
    
    inf_parser.add_argument("--wav_bit_depth", type=str, default="FLOAT", choices=["FLOAT", "PCM_16", "PCM_24"])
    inf_parser.add_argument("--flac_bit_depth", type=str, default="PCM_24", choices=["PCM_16", "PCM_24"])
    inf_parser.add_argument("--mp3_bit_rate", type=str, default="320k")

    dia_parser = subparsers.add_parser("diarization", help="Run speaker diarization")
    dia_parser.add_argument("--input", "-i", type=str, required=True, help="Input file")
    dia_parser.add_argument("--output_dir", "-o", type=str, default="output_diarization", help="Output directory")
    dia_parser.add_argument("--device", type=str, default="auto", help="Device to use")
    dia_parser.add_argument("--batch_size", type=int, help="Batch size for diarization phase 1")
    dia_parser.add_argument("--process_overlap", action="store_true", default=True, help="Process and keep overlapping segments")

    asr_parser = subparsers.add_parser("asr", help="Run ASR (OWSM-CTC v4)")
    asr_parser.add_argument("--input", "-i", type=str, required=True, help="Input file")
    asr_parser.add_argument("--output_dir", "-o", type=str, default="output_asr", help="Output directory")
    asr_parser.add_argument("--device", type=str, default="auto", help="Device to use")
    asr_parser.add_argument("--language", "-l", type=str, default="auto", help="ASR Language (e.g., auto, kor, eng)")
    asr_parser.add_argument("--model_type", type=str, default="whisper", choices=["whisper", "owsmv4"], help="ASR Model to use: whisper or owsmv4")

    json_parser = subparsers.add_parser("json_api", help="Run via JSON config (Internal IPC)")
    json_parser.add_argument("--task", type=str, required=True, choices=["inference", "diarization", "asr", "pipeline", "download", "clean"])
    json_parser.add_argument("--config", type=str, required=True, help="JSON string config")

    args = parser.parse_args()

    if args.command == "download":
        if args.list:
            list_models()
        elif args.model:
            download_model(args.model)
        else:
            dl_parser.print_help()
    elif args.command == "add_model":
        add_custom_model(args)
    elif args.command == "inference":
        if not args.model_name and not args.model_path:
            logger.error("You must specify either --model_name or --model_path")
            return
        run_inference(args)
    elif args.command == "diarization":
        config = {
            "input": args.input,
            "output_dir": args.output_dir,
            "device": args.device,
            "batch_size": args.batch_size,
            "process_overlap": args.process_overlap
        }
        run_diarization(config)
    elif args.command == "asr":
        config = {
            "input": args.input,
            "output_dir": args.output_dir,
            "device": args.device,
            "language": args.language,
            "model_type": args.model_type
        }
        run_asr(config)
    elif args.command == "json_api":
        import json
        config = json.loads(args.config)
        if args.task == "inference":
            run_inference_api(config)
        elif args.task == "diarization":
            run_diarization(config)
        elif args.task == "asr":
            run_asr(config)
        elif args.task == "clean":
            run_cleaner(config)
        elif args.task == "download":
            def prog(pct):
                print(f"PROGRESS_UPDATE:{pct}", flush=True)
            download_model(config["model"], progress_callback=prog)
        elif args.task == "pipeline":
            import pipeline
            pipe = pipeline.VoxUnravelPipeline(device="auto", logger=logging.getLogger())
            def prog(pct):
                print(f"PROGRESS_UPDATE:{pct}", flush=True)
            pipe.run_full_process(
                config["input_file"], 
                config["output_dir"], 
                sep_config=config["sep_config"], 
                asr_config=config["asr_config"], 
                progress_callback=prog
            )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()