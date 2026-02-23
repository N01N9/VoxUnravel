import os
import torch
import librosa
import logging
import soundfile as sf
import whisper

class ASRInfer:
    def __init__(self, audio_path, output_dir="output_asr", device='cuda:0', language='auto', model_type='whisper', logger=None):
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.device = device
        self.language = language
        self.model_type = model_type.lower()
        self.logger = logger or logging.getLogger("asr_infer")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.model = None

    def load_model(self):
        if self.model is None:
            if self.model_type == 'whisper':
                self.logger.info("Loading Whisper large-v3 model...")
                
                lang_msg = "auto" if self.language == 'auto' else self.language
                self.logger.info(f"Using language: {lang_msg}")

                device = self.device if torch.cuda.is_available() and 'cuda' in self.device else 'cpu'
                self.model = whisper.load_model("large-v3", device=device)
                self.logger.info("Whisper model loaded successfully.")
            elif self.model_type == 'owsmv4':
                import sys
                try:
                    import espnet2
                    from espnet2.bin.s2t_inference_ctc import Speech2TextGreedySearch
                except ImportError:
                    self.logger.error("espnet2 is not installed or could not be loaded. Please install requirements for owsmv4.")
                    raise
                    
                self.logger.info(f"Python executable: {sys.executable}")
                self.logger.info(f"ESPnet2 location: {espnet2.__file__}")
                
                self.logger.info("Loading OWSM-CTC v4 model (espnet/owsm_ctc_v4_1B)...")
                
                lang_sym = '<nolang>' if self.language == 'auto' else f'<{self.language}>'
                self.logger.info(f"Using language token: {lang_sym}")

                self.model = Speech2TextGreedySearch.from_pretrained(
                    model_tag="espnet/owsm_ctc_v4_1B",
                    device=self.device,
                    lang_sym=lang_sym,
                    task_sym='<asr>',
                    use_flash_attn=False
                )
                self.logger.info("OWSMv4 model loaded successfully.")
            else:
                raise ValueError(f"Unknown ASR model type: {self.model_type}")

    def _free_vram(self):
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def transcribe(self):
        self.load_model()
        
        self.logger.info(f"Loading audio: {self.audio_path}")
        
        self.logger.info("Starting ASR inference...")
        with torch.no_grad():
            try:
                # Load audio using librosa to gracefully handle WAV without external ffmpeg
                audio_np, rate = librosa.load(self.audio_path, sr=16000)
                
                if self.model_type == 'whisper':
                    lang = self.language if self.language != 'auto' else None
                    if lang:
                        lang_map = {'kor': 'ko', 'eng': 'en', 'jpn': 'ja', 'zho': 'zh'}
                        lang = lang_map.get(lang, lang)
                    results = self.model.transcribe(audio_np, language=lang)
                    text = results["text"].strip()
                elif self.model_type == 'owsmv4':
                    results = self.model(audio_np)
                    text = results[0][3] if (len(results[0]) > 3 and results[0][3] is not None) else results[0][0]
                else:
                    text = ""
            except Exception as e:
                self.logger.error(f"Inference error on {self.audio_path}: {e}")
                raise e
            
        base_name = os.path.splitext(os.path.basename(self.audio_path))[0]
        txt_path = os.path.join(self.output_dir, f"{base_name}.txt")
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
            
        self.logger.info(f"Transcription saved to: {txt_path}")
        
        del self.model
        self.model = None
        self._free_vram()
        
        return text

    def run_all(self):
        return self.transcribe()