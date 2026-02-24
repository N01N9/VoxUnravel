import os

# ==========================================
# [최종 패치] PyTorch 2.6+ 보안 정책(weights_only=True) 3중 무력화
# 1. 환경변수를 통한 전역 비활성화 (내부 라이브러리인 pytorch_lightning 등에 우선 적용됨)
# ==========================================
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import gc
import torch

# ==========================================
# 2. 명시적으로 weights_only=True 파라미터가 강제 주입되는 경우를 대비한 패치
# ==========================================
_original_load = torch.load
def override_load(*args, **kwargs):
    kwargs['weights_only'] = False 
    return _original_load(*args, **kwargs)
torch.load = override_load

# ==========================================
# 3. pyannote 객체를 안전한 전역 객체(safe_globals)로 화이트리스트 등록
# ==========================================
if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        from pyannote.audio.core.task import Specifications, Problem, Resolution
        torch.serialization.add_safe_globals([Specifications, Problem, Resolution])
    except Exception:
        pass
# ==========================================

import torchaudio
import numpy as np
import soundfile as sf
import tempfile
import librosa
from sklearn.metrics.pairwise import cosine_similarity
from diarizen.pipelines.inference import DiariZenPipeline
from modelscope.pipelines import pipeline
import logging

class DiarizationInfer:
    def __init__(self, audio_path, output_dir="output_speakers", device='cuda:0', logger=None):
        self.audio_path = audio_path
        self.output_dir = output_dir
        self.device = device
        self.sr = 16000
        self.logger = logger or logging.getLogger("diarization_infer")
        
        self.diarizen_batch_size = 8
        self.audiosr_steps = 50
        self.max_chunk_sec = 10.0
        self.threshold_new_spk = 0.30
        self.alpha = 0.15
        self.process_overlap = True
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.signal, fs = torchaudio.load(audio_path)
        if fs != self.sr:
            self.signal = torchaudio.transforms.Resample(fs, self.sr)(self.signal)
        if self.signal.shape[0] > 1:
            self.signal = torch.mean(self.signal, dim=0, keepdim=True)
        self.signal_np = self.signal.squeeze().numpy()
        
        self.diarizen = None
        self.mossformer = None
        self.verifier = None
        self.audiosr_model = None

    def _free_vram(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _extract_embedding(self, audio_array):
        tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_wav = tmp_file.name
        tmp_file.close()
        try:
            sf.write(tmp_wav, audio_array, self.sr)
            input_data = self.verifier.preprocess([tmp_wav, tmp_wav])
            model_out = self.verifier.forward(input_data)
            if isinstance(model_out, torch.Tensor): emb = model_out[0] 
            elif isinstance(model_out, tuple): emb = model_out[0][0]
            elif isinstance(model_out, dict):
                embs = model_out.get('spk_embedding', model_out.get('spk_embeddings'))
                emb = embs[0]
            if isinstance(emb, torch.Tensor): emb = emb.detach().cpu().numpy()
            return np.array(emb).flatten()
        finally:
            if os.path.exists(tmp_wav): os.remove(tmp_wav)

    def _split_by_vad(self, audio_array, frame_len=0.02, energy_thresh=0.005, min_dur=0.2, min_silence_dur=1.0):
        frame_samples = int(self.sr * frame_len)
        if len(audio_array) < frame_samples: return []
        num_frames = len(audio_array) // frame_samples
        energies = [np.sqrt(np.mean(audio_array[i*frame_samples : (i+1)*frame_samples]**2)) for i in range(num_frames)]
        active_mask = np.array(energies) > energy_thresh
        
        silence_limit = int(min_silence_dur / frame_len)
        
        sub_segments = []
        in_speech = False
        start_frame = 0
        silence_counter = 0
        
        for i, is_active in enumerate(active_mask):
            if is_active:
                if not in_speech:
                    in_speech = True
                    start_frame = i
                silence_counter = 0 
            else:
                if in_speech:
                    silence_counter += 1
                    if silence_counter >= silence_limit:
                        in_speech = False
                        end_frame = i - silence_counter + 1
                        start_samp, end_samp = start_frame * frame_samples, end_frame * frame_samples
                        if (end_samp - start_samp) / self.sr >= min_dur:
                            sub_segments.append({'audio': audio_array[start_samp:end_samp], 'offset_sec': start_samp / self.sr})
        
        if in_speech:
            start_samp = start_frame * frame_samples
            if (len(audio_array) - start_samp) / self.sr >= min_dur:
                sub_segments.append({'audio': audio_array[start_samp:], 'offset_sec': start_samp / self.sr})
        
        return sub_segments

    def _save_speaker_audio(self, audio_array, speaker_id, filename):
        spk_str = str(speaker_id)
        spk_dir = os.path.join(self.output_dir, spk_str)
        os.makedirs(spk_dir, exist_ok=True)
        sf.write(os.path.join(spk_dir, filename), audio_array, self.sr)

    def step1_initial_diarization(self):
        self.logger.info(f"[Phase 1] DiariZen scanning... (Batch size: {self.diarizen_batch_size})")
        self.diarizen = DiariZenPipeline.from_pretrained("BUT-FIT/diarizen-wavlm-large-s80-md-v2")
        
        if hasattr(self.diarizen, 'config'):
            self.diarizen.config['inference']['args']['batch_size'] = self.diarizen_batch_size
        
        with torch.no_grad():
            results = self.diarizen(self.audio_path)

        self.segments = []
        for turn, _, speaker in results.itertracks(yield_label=True):
            spk_str = f"Speaker_{speaker}" if isinstance(speaker, (int, np.integer)) else str(speaker)
            self.segments.append({'start': turn.start, 'end': turn.end, 'speaker': spk_str, 'is_overlap': False})
            
        self.segments.sort(key=lambda x: x['start'])
        for i in range(len(self.segments)):
            for j in range(i + 1, len(self.segments)):
                if self.segments[j]['start'] < self.segments[i]['end']:
                    self.segments[i]['is_overlap'] = True
                    self.segments[j]['is_overlap'] = True
                else: break
        
        del self.diarizen
        self._free_vram()
        self.logger.info(f"-> Phase 1 completed")

    def step2_process_and_refine(self):
        self.centroids = {}
        self.final_segments = []
        unknown_counter = 0
        chunk_samples = int(self.max_chunk_sec * self.sr)
        
        self.logger.info("[Phase 2] Pure segment verification (ERes2NetV2)...")
        self.verifier = pipeline(task='speaker-verification', model='iic/speech_eres2netv2_sv_zh-cn_16k-common')
        pure_segments = [seg for seg in self.segments if not seg['is_overlap']]
        for seg in pure_segments:
            audio_cut = self.signal_np[int(seg['start']*self.sr):int(seg['end']*self.sr)]
            if len(audio_cut) < int(self.sr * 0.3): continue
            embs = [self._extract_embedding(audio_cut[i:i+chunk_samples]) for i in range(0, len(audio_cut), chunk_samples) if len(audio_cut[i:i+chunk_samples]) >= int(self.sr * 0.3)]
            if not embs: continue
            emb = np.mean(embs, axis=0)
            if seg['speaker'] not in self.centroids: self.centroids[seg['speaker']] = emb
            else: self.centroids[seg['speaker']] = (1 - self.alpha) * self.centroids[seg['speaker']] + self.alpha * emb
            self._save_speaker_audio(audio_cut, seg['speaker'], f"pure_{seg['start']:.2f}.wav")
            self.final_segments.append(seg)
        del self.verifier
        self._free_vram()

        if self.process_overlap:
            self.logger.info("[Phase 3] Overlap separation and VAD segmentation (MossFormer)...")
            separated_items_8k = []
            overlap_segments = [seg for seg in self.segments if seg['is_overlap']]
            if overlap_segments:
                self.mossformer = pipeline(task='speech-separation', model='damo/speech_mossformer2_separation_temporal_8k')
                for seg in overlap_segments:
                    full_audio = self.signal_np[int(seg['start']*self.sr):int(seg['end']*self.sr)]
                    for o_idx in range(0, len(full_audio), chunk_samples):
                        chunk_audio = full_audio[o_idx:o_idx+chunk_samples]
                        if len(chunk_audio) < int(self.sr * 0.2): continue
                        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                        tmp_name = tmp.name
                        tmp.close()
                        try:
                            sf.write(tmp_name, librosa.resample(chunk_audio, orig_sr=self.sr, target_sr=8000), 8000)
                            res = self.mossformer(tmp_name)
                        finally:
                            if os.path.exists(tmp_name):
                                os.remove(tmp_name)
                        for sep_audio in res.get('output_pcm_list', []):
                            if isinstance(sep_audio, bytes): s_np = np.frombuffer(sep_audio, dtype=np.int16).astype(np.float32)/32768.0
                            elif isinstance(sep_audio, str): s_np = torchaudio.load(sep_audio)[0].squeeze().numpy()
                            else: s_np = np.array(sep_audio)
                            tmp_16k = librosa.resample(s_np, orig_sr=8000, target_sr=self.sr)
                            sub_chunks = self._split_by_vad(tmp_16k)
                            for sc in sub_chunks:
                                separated_items_8k.append({
                                    'base_start': seg['start'] + (o_idx/self.sr) + sc['offset_sec'],
                                    'audio_8k': librosa.resample(sc['audio'], orig_sr=self.sr, target_sr=8000)
                                })
                del self.mossformer
                self._free_vram()

            self.logger.info(f"[Phase 4] AudioSR super-resolution for {len(separated_items_8k)} VAD segments...")
            sr_items_48k = []
            if separated_items_8k:
                from audiosr import build_model, super_resolution
                self.audiosr_model = build_model(model_name="basic", device=self.device)
                target_len_8k = 81920 
                for item in separated_items_8k:
                    aud = item['audio_8k']
                    padded = np.pad(aud, (0, max(0, target_len_8k - len(aud))), 'constant')[:target_len_8k]
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp_name = tmp.name
                    tmp.close()
                    try:
                        sf.write(tmp_name, padded, 8000)
                        sr_wav = super_resolution(self.audiosr_model, tmp_name, guidance_scale=3.5, ddim_steps=self.audiosr_steps)
                    finally:
                        if os.path.exists(tmp_name):
                            os.remove(tmp_name)
                    sr_np = sr_wav.squeeze().detach().cpu().numpy() if isinstance(sr_wav, torch.Tensor) else np.array(sr_wav).squeeze()
                    sr_items_48k.append({'start': item['base_start'], 'audio_48k': sr_np[:int(len(aud)*(48000/8000))]})
                del self.audiosr_model
                self._free_vram()

            self.logger.info("[Phase 5] Final speaker identification...")
            if sr_items_48k:
                self.verifier = pipeline(task='speaker-verification', model='iic/speech_eres2netv2_sv_zh-cn_16k-common')
                for item in sr_items_48k:
                    aud_16k = librosa.resample(item['audio_48k'], orig_sr=48000, target_sr=self.sr)
                    if len(aud_16k) < int(self.sr * 0.2): continue
                    emb = self._extract_embedding(aud_16k)
                    sims = {spk: cosine_similarity([emb], [c])[0][0] for spk, c in self.centroids.items()}
                    if not sims: continue
                    best_spk = max(sims, key=sims.get)
                    if sims[best_spk] < self.threshold_new_spk:
                        best_spk = f"Unknown_{unknown_counter}"; unknown_counter += 1
                        self.centroids[best_spk] = emb
                    self._save_speaker_audio(aud_16k, best_spk, f"overlap_{item['start']:.2f}.wav")
                    self.final_segments.append({'start': item['start'], 'end': item['start']+(len(aud_16k)/self.sr), 'speaker': best_spk})
                del self.verifier
                self._free_vram()
        else:
            self.logger.info("[Phase 3-5] Skipping overlap processing (ignored by setting)")

    def export_rttm(self, path):
        self.final_segments.sort(key=lambda x: x['start'])
        with open(path, 'w') as f:
            for s in self.final_segments:
                f.write(f"SPEAKER audio 1 {s['start']:.3f} {s['end']-s['start']:.3f} <NA> <NA> {s['speaker']} <NA> <NA>\n")
        self.logger.info(f"💾 RTTM saved: {path}")

    def run_all(self, rttm_path=None):
        if rttm_path is None:
            rttm_path = os.path.join(self.output_dir, "final_refined.rttm")
        self.step1_initial_diarization()
        self.step2_process_and_refine()
        self.export_rttm(rttm_path)
        return rttm_path