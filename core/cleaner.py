import os
import csv
import soundfile as sf
import logging

class DatasetCleaner:
    def __init__(self, target_dir, min_duration=1.0, logger=None):
        self.target_dir = target_dir
        self.min_duration = float(min_duration)
        self.logger = logger or logging.getLogger("cleaner")
        
    def get_audio_duration(self, file_path):
        try:
            info = sf.info(file_path)
            return info.duration
        except Exception as e:
            self.logger.warning(f"Could not read duration for {file_path}: {e}")
            return 0.0

    def clean(self):
        self.logger.info(f"Starting cleanup in {self.target_dir}")
        self.logger.info(f"Criteria: Min Duration = {self.min_duration}s, Text must not be empty.")
        
        if not os.path.isdir(self.target_dir):
            self.logger.error(f"Directory not found: {self.target_dir}")
            return
            
        deleted_count = 0
        total_count = 0
        
        # 1. First, we handle the case where metadata.csv and list.txt exist.
        metadata_path = os.path.join(self.target_dir, "metadata.csv")
        list_path = os.path.join(self.target_dir, "list.txt")
        
        valid_rows_metadata = []
        valid_rows_list = []
        
        has_metadata = os.path.exists(metadata_path)
        has_list = os.path.exists(list_path)
        
        if has_metadata:
            self.logger.info("metadata.csv found. Processing via metadata listing...")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='|')
                for row in reader:
                    total_count += 1
                    audio_path = row.get("file_path", "")
                    text = row.get("transcription", "").strip()
                    
                    if not os.path.isabs(audio_path):
                        audio_path = os.path.join(self.target_dir, audio_path)
                        
                    should_delete = False
                    
                    if not text:
                        should_delete = True
                        reason = "Empty text"
                    elif os.path.exists(audio_path):
                        dur = self.get_audio_duration(audio_path)
                        if dur < self.min_duration:
                            should_delete = True
                            reason = f"Duration {dur:.2f}s < {self.min_duration}s"
                    else:
                        should_delete = True
                        reason = "Audio file missing"
                        
                    if should_delete:
                        self.logger.debug(f"Deleting item: {audio_path} | Reason: {reason}")
                        if os.path.exists(audio_path):
                            try:
                                os.remove(audio_path)
                                deleted_count += 1
                                # Try to delete accompanying text file if any
                                txt_path = os.path.splitext(audio_path)[0] + ".txt"
                                if os.path.exists(txt_path):
                                    os.remove(txt_path)
                                # Try to delete ASR result txt in asr_results if paths are matched
                                # Just a general check
                            except Exception as e:
                                self.logger.warning(f"Failed to delete {audio_path}: {e}")
                    else:
                        valid_rows_metadata.append(row)
                        
            # Rewrite metadata.csv
            with open(metadata_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=["file_path", "speaker_id", "transcription"], delimiter="|")
                writer.writeheader()
                for row in valid_rows_metadata:
                    writer.writerow(row)
                    
            # Rewrite list.txt
            if has_list:
                # We assume list.txt and metadata.csv have the same valid items
                with open(list_path, 'w', encoding='utf-8') as f:
                    for row in valid_rows_metadata:
                        rel_path = os.path.relpath(row["file_path"], self.target_dir) if os.path.isabs(row["file_path"]) else row["file_path"]
                        f.write(f"{rel_path}|{row['transcription']}\n")
                        
        else:
            self.logger.info("metadata.csv not found. Scanning directory for all audio files...")
            audio_exts = {'.wav', '.mp3', '.flac'}
            for root, dirs, files in os.walk(self.target_dir):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in audio_exts:
                        total_count += 1
                        audio_path = os.path.join(root, file)
                        txt_path = os.path.splitext(audio_path)[0] + ".txt"
                        
                        text = ""
                        if os.path.exists(txt_path):
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                text = f.read().strip()
                                
                        should_delete = False
                        if not text:
                            should_delete = True
                            reason = "Empty or missing text file"
                        else:
                            dur = self.get_audio_duration(audio_path)
                            if dur < self.min_duration:
                                should_delete = True
                                reason = f"Duration {dur:.2f}s < {self.min_duration}s"
                                
                        if should_delete:
                            self.logger.debug(f"Deleting item: {audio_path} | Reason: {reason}")
                            try:
                                os.remove(audio_path)
                                deleted_count += 1
                                if os.path.exists(txt_path):
                                    os.remove(txt_path)
                            except Exception as e:
                                self.logger.warning(f"Failed to delete {audio_path}: {e}")
                                
        self.logger.info(f"Cleanup finished. Checked {total_count} files, deleted {deleted_count} files.")
        
    def run_all(self):
        self.clean()
