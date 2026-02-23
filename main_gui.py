import sys
import os
import json
import logging
import subprocess
import time

# === Startup optimizations (non-blocking) ===
os.environ.setdefault("NUMEXPR_MAX_THREADS", "8")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QComboBox, QLineEdit, QPushButton, QCheckBox, 
                             QTextEdit, QFileDialog, QListWidget, QListWidgetItem, 
                             QProgressBar, QMessageBox, QGroupBox, QSplitter, QTabWidget, QDoubleSpinBox)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont, QIcon, QPalette, QColor

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
    return {"main": sys.executable, "inference": sys.executable, "separation": sys.executable, "diarization": sys.executable, "asr": sys.executable, "pipeline": sys.executable}


class LogSignaler(QThread):
    new_log = Signal(str)

class QueueHandler(logging.Handler):
    def __init__(self, signaler):
        super().__init__()
        self.signaler = signaler

    def emit(self, record):
        msg = self.format(record)
        self.signaler.new_log.emit(msg)

class ProcessWorker(QThread):
    task_success = Signal()
    progress = Signal(int)
    error = Signal(str)
    log = Signal(str)

    def __init__(self, task_type, config_dict):
        super().__init__()
        self.task_type = task_type
        self.config_dict = config_dict

    def run(self):
        try:
            envs = load_environments()
            env_key = self.task_type
            if env_key == "inference": env_key = "separation"
            if env_key in ["pipeline", "download"]: env_key = "main"
            if env_key == "clean": env_key = "asr"
            
            python_exe = envs.get(env_key, sys.executable)
            if not os.path.exists(python_exe) and python_exe != "python":
                python_exe = sys.executable

            runner_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner.py")
            config_str = json.dumps(self.config_dict)
            cmd = [python_exe, "-u", runner_script, "json_api", "--task", self.task_type, "--config", config_str]

            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

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
                        self.progress.emit(prog)
                    except: pass
                else:
                    self.log.emit(line)

            process.stdout.close()
            process.wait()
            
            if process.returncode != 0:
                self.error.emit(f"Process failed with code {process.returncode}")
            else:
                self.task_success.emit()
        except Exception as e:
            self.error.emit(str(e))

class InitWorker(QThread):
    progress_text = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def run(self):
        try:
            print("[InitWorker] Started")
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            
            self.progress_text.emit("Loading core modules...")
            import runner
            print("[InitWorker] Import runner OK")
            
            self.progress_text.emit("Warming up AI environments in background...")
            
            envs = load_environments()
            tasks = ["inference", "diarization", "asr"]
            processes = []
            
            for task in tasks:
                env_key = task
                if env_key == "inference": env_key = "separation"
                
                python_exe = envs.get(env_key, sys.executable)
                if not os.path.exists(python_exe) and python_exe != "python":
                    python_exe = sys.executable
                    
                startupinfo = None
                if os.name == 'nt':
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    
                init_script = "import torch, warnings; warnings.filterwarnings('ignore')"
                cmd = [python_exe, "-c", init_script]
                
                self.progress_text.emit(f"Warming up {env_key} environment...")
                p = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL,
                    startupinfo=startupinfo
                )
                processes.append(p)
                
            self.progress_text.emit("Waiting for environments to initialize...")
            for p in processes:
                try:
                    p.wait(timeout=15) # 타임아웃 15초 제한으로 무한 대기 원천 차단
                except subprocess.TimeoutExpired:
                    print(f"[InitWorker] Subprocess warmup timeout, killing process.")
                    p.kill()
            print("[InitWorker] Warmup done")

            self.progress_text.emit("Reading model database...")
            models_info = runner.load_models_info()
            print("[InitWorker] Loading models done")
            self.progress_text.emit("Initialization complete!")
            
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            
            self.finished.emit(models_info)
            
        except BaseException as e: # SystemExit(sys.exit)의 조용한 폭파 현상을 감지하기 위해 BaseException 사용
            os.environ.pop("HF_HUB_OFFLINE", None)
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            error_msg = str(e) if str(e) else "SystemExit or Unknown Error occurred"
            print(f"[InitWorker] Exception caught: {error_msg}")
            self.error.emit(error_msg)


class MainWindow(QMainWindow):
    def __init__(self, models_info=None):
        super().__init__()
        self.setWindowTitle("VoxUnravel - MSST")
        self.resize(1200, 800)
        self.model_info = models_info or {}
        
        self.apply_dark_theme()
        
        # 더 이상 내부 로딩 오버레이를 쓰지 않고 메인 UI를 직관적으로 구성합니다.
        self.main_content = QWidget()
        self.setCentralWidget(self.main_content)
        
        self.lang_dict = load_languages()
        self._build_main_ui()
        
        self.log_signaler = LogSignaler()
        self.log_signaler.new_log.connect(self.append_log)
        handler = QueueHandler(self.log_signaler)
        logging.getLogger().addHandler(handler)
        
        # 모델 정보 로드 반영
        self.filter_models()

    def _build_main_ui(self):
        main_layout = QHBoxLayout(self.main_content)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        self.left_panel = QWidget()
        left_layout = QVBoxLayout(self.left_panel)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        search_layout = QHBoxLayout()
        search_label = QLabel("Search:")
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter models...")
        self.search_input.textChanged.connect(self.filter_models)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.search_input)
        left_layout.addLayout(search_layout)
        
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.model_list.itemClicked.connect(self.on_model_selected)
        left_layout.addWidget(self.model_list)
        
        self.model_info_label = QLabel("Select a model to view details")
        self.model_info_label.setWordWrap(True)
        left_layout.addWidget(self.model_info_label)
        
        self.download_btn = QPushButton("Download Model (Via Main Env)")
        self.download_btn.clicked.connect(self.download_current_model)
        self.download_btn.setEnabled(False)
        left_layout.addWidget(self.download_btn)
        
        custom_group = QGroupBox("Add Custom Model")
        custom_layout = QVBoxLayout()
        self.custom_name = QLineEdit()
        self.custom_name.setPlaceholderText("Model Name")
        self.custom_path = QLineEdit()
        self.custom_path.setPlaceholderText("Model Path")
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.custom_path)
        browse_path_btn = QPushButton("...")
        browse_path_btn.clicked.connect(self.browse_custom_path)
        path_layout.addWidget(browse_path_btn)
        
        self.custom_type = QComboBox()
        self.custom_type.addItems(["bs_roformer", "mdx23c", "mel_band_roformer", "htdemucs", "scnet", "segm_models"])
        
        add_btn = QPushButton("Register Custom Model")
        add_btn.clicked.connect(self.register_custom_model)
        
        custom_layout.addWidget(self.custom_name)
        custom_layout.addLayout(path_layout)
        custom_layout.addWidget(self.custom_type)
        custom_layout.addWidget(add_btn)
        custom_group.setLayout(custom_layout)
        left_layout.addWidget(custom_group)
        
        splitter.addWidget(self.left_panel)
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(10, 10, 10, 10)
        
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        right_layout.addWidget(self.tabs)
        
        # --- TAB 1: Separation ---
        sep_tab = QWidget()
        sep_layout = QVBoxLayout(sep_tab)
        
        sep_io_group = QGroupBox("Input / Output")
        sep_io_layout = QVBoxLayout()
        
        sep_input_layout = QHBoxLayout()
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Select Audio File(s) or Folder")
        btn_sep_file = QPushButton("Files")
        btn_sep_file.clicked.connect(lambda: self.browse_generic(self.input_path, 'files'))
        btn_sep_folder = QPushButton("Folder")
        btn_sep_folder.clicked.connect(lambda: self.browse_generic(self.input_path, 'dir'))
        sep_input_layout.addWidget(self.input_path)
        sep_input_layout.addWidget(btn_sep_file)
        sep_input_layout.addWidget(btn_sep_folder)
        sep_io_layout.addWidget(QLabel("Input:"))
        sep_io_layout.addLayout(sep_input_layout)
        
        sep_output_layout = QHBoxLayout()
        self.output_dir = QLineEdit("results")
        browse_output_btn = QPushButton("Browse")
        browse_output_btn.clicked.connect(self.browse_output)
        sep_output_layout.addWidget(self.output_dir)
        sep_output_layout.addWidget(browse_output_btn)
        sep_io_layout.addWidget(QLabel("Output Directory:"))
        sep_io_layout.addLayout(sep_output_layout)
        
        sep_io_group.setLayout(sep_io_layout)
        sep_layout.addWidget(sep_io_group)
        
        sep_settings_group = QGroupBox("Settings")
        sep_settings_layout = QVBoxLayout()
        
        row1 = QHBoxLayout()
        self.output_format = QComboBox()
        self.output_format.addItems(["wav", "flac", "mp3"])
        row1.addWidget(QLabel("Format:"))
        row1.addWidget(self.output_format)
        
        self.device = QComboBox()
        self.device.addItems(["auto", "cuda", "cpu"])
        row1.addWidget(QLabel("Device:"))
        row1.addWidget(self.device)
        sep_settings_layout.addLayout(row1)
        
        self.normalize = QCheckBox("Normalize Output")
        self.use_tta = QCheckBox("Test Time Augmentation (TTA)")
        self.ensemble_mode = QCheckBox("Ensemble Mode (Average Selected Models)")
        sep_settings_layout.addWidget(self.normalize)
        sep_settings_layout.addWidget(self.use_tta)
        sep_settings_layout.addWidget(self.ensemble_mode)
        
        sep_settings_group.setLayout(sep_settings_layout)
        sep_layout.addWidget(sep_settings_group)
        
        self.run_btn = QPushButton("START SEPARATION")
        self.run_btn.setFixedHeight(50)
        self.run_btn.setStyleSheet("QPushButton { background-color: #2ea44f; color: white; font-size: 16px; font-weight: bold; border-radius: 6px; }")
        self.run_btn.clicked.connect(self.start_inference)
        sep_layout.addWidget(self.run_btn)
        sep_layout.addStretch()
        
        self.tabs.addTab(sep_tab, "Separation")
        
        # --- TAB 2: Diarization ---
        dia_tab = QWidget()
        dia_layout = QVBoxLayout(dia_tab)
        
        dia_io_group = QGroupBox("Input / Output")
        dia_io_layout = QVBoxLayout()
        
        dia_input_layout = QHBoxLayout()
        self.dia_input_path = QLineEdit()
        self.dia_input_path.setPlaceholderText("Select Audio File(s) or Folder")
        btn_dia_file = QPushButton("Files")
        btn_dia_file.clicked.connect(lambda: self.browse_generic(self.dia_input_path, 'files'))
        btn_dia_folder = QPushButton("Folder")
        btn_dia_folder.clicked.connect(lambda: self.browse_generic(self.dia_input_path, 'dir'))
        dia_input_layout.addWidget(self.dia_input_path)
        dia_input_layout.addWidget(btn_dia_file)
        dia_input_layout.addWidget(btn_dia_folder)
        dia_io_layout.addWidget(QLabel("Input:"))
        dia_io_layout.addLayout(dia_input_layout)
        
        dia_output_layout = QHBoxLayout()
        self.dia_output_dir = QLineEdit("output_diarization")
        browse_dia_output_btn = QPushButton("Browse")
        browse_dia_output_btn.clicked.connect(self.browse_dia_output)
        dia_output_layout.addWidget(self.dia_output_dir)
        dia_output_layout.addWidget(browse_dia_output_btn)
        dia_io_layout.addWidget(QLabel("Output Directory:"))
        dia_io_layout.addLayout(dia_output_layout)
        
        dia_io_group.setLayout(dia_io_layout)
        dia_layout.addWidget(dia_io_group)
        
        dia_settings_group = QGroupBox("Advanced Settings")
        dia_settings_layout = QVBoxLayout()
        
        row_dia = QHBoxLayout()
        self.dia_batch_size = QComboBox()
        self.dia_batch_size.addItems(["4", "8", "16", "32"])
        self.dia_batch_size.setCurrentText("8")
        row_dia.addWidget(QLabel("Phase 1 Batch Size:"))
        row_dia.addWidget(self.dia_batch_size)
        
        self.dia_device = QComboBox()
        self.dia_device.addItems(["auto", "cuda", "cpu"])
        row_dia.addWidget(QLabel("Device:"))
        row_dia.addWidget(self.dia_device)
        dia_settings_layout.addLayout(row_dia)
        
        self.dia_sr_steps = QLineEdit("50")
        dia_settings_layout.addWidget(QLabel("AudioSR Steps (Diarization Stage 4):"))
        dia_settings_layout.addWidget(self.dia_sr_steps)
        
        self.dia_process_overlap = QCheckBox("Process & Keep Overlapping Segments")
        self.dia_process_overlap.setChecked(True)
        dia_settings_layout.addWidget(self.dia_process_overlap)
        
        dia_settings_group.setLayout(dia_settings_layout)
        dia_layout.addWidget(dia_settings_group)
        
        self.dia_run_btn = QPushButton("START DIARIZATION")
        self.dia_run_btn.setFixedHeight(50)
        self.dia_run_btn.setStyleSheet("QPushButton { background-color: #1f6feb; color: white; font-size: 16px; font-weight: bold; border-radius: 6px; }")
        self.dia_run_btn.clicked.connect(self.start_diarization)
        dia_layout.addWidget(self.dia_run_btn)
        dia_layout.addStretch()
        
        self.tabs.addTab(dia_tab, "Speaker Diarization")
        
        # --- TAB 3: ASR (Speech-to-Text) ---
        asr_tab = QWidget()
        asr_layout = QVBoxLayout(asr_tab)
        
        asr_io_group = QGroupBox("Input / Output")
        asr_io_layout = QVBoxLayout()
        
        asr_input_layout = QHBoxLayout()
        self.asr_input_path = QLineEdit()
        self.asr_input_path.setPlaceholderText("Select Audio File(s) or Folder")
        btn_asr_file = QPushButton("Files")
        btn_asr_file.clicked.connect(lambda: self.browse_generic(self.asr_input_path, 'files'))
        btn_asr_folder = QPushButton("Folder")
        btn_asr_folder.clicked.connect(lambda: self.browse_generic(self.asr_input_path, 'dir'))
        asr_input_layout.addWidget(self.asr_input_path)
        asr_input_layout.addWidget(btn_asr_file)
        asr_input_layout.addWidget(btn_asr_folder)
        asr_io_layout.addWidget(QLabel("Input:"))
        asr_io_layout.addLayout(asr_input_layout)
        
        asr_output_layout = QHBoxLayout()
        self.asr_output_dir = QLineEdit("output_asr")
        browse_asr_output_btn = QPushButton("Browse")
        browse_asr_output_btn.clicked.connect(self.browse_asr_output)
        asr_output_layout.addWidget(self.asr_output_dir)
        asr_output_layout.addWidget(browse_asr_output_btn)
        asr_io_layout.addWidget(QLabel("Output Directory:"))
        asr_io_layout.addLayout(asr_output_layout)
        
        asr_io_group.setLayout(asr_io_layout)
        asr_layout.addWidget(asr_io_group)
        
        asr_settings_group = QGroupBox("Settings")
        asr_settings_layout = QVBoxLayout()
        
        row_asr = QHBoxLayout()
        self.asr_device = QComboBox()
        self.asr_device.addItems(["auto", "cuda", "cpu"])
        row_asr.addWidget(QLabel("Device:"))
        row_asr.addWidget(self.asr_device)
        
        self.asr_lang = QComboBox()
        for code, name in self.lang_dict.items():
            self.asr_lang.addItem(f"{name} ({code})", code)
        row_asr.addWidget(QLabel("Language:"))
        row_asr.addWidget(self.asr_lang)
        
        self.asr_model_combo = QComboBox()
        self.asr_model_combo.addItem("Whisper Large v3", "whisper")
        self.asr_model_combo.addItem("OWSM-CTC v4", "owsmv4")
        row_asr.addWidget(QLabel("Model:"))
        row_asr.addWidget(self.asr_model_combo)
        
        asr_settings_layout.addLayout(row_asr)
        
        asr_settings_group.setLayout(asr_settings_layout)
        asr_layout.addWidget(asr_settings_group)
        
        self.asr_run_btn = QPushButton("START ASR")
        self.asr_run_btn.setFixedHeight(50)
        self.asr_run_btn.setStyleSheet("QPushButton { background-color: #8957e5; color: white; font-size: 16px; font-weight: bold; border-radius: 6px; }")
        self.asr_run_btn.clicked.connect(self.start_asr)
        asr_layout.addWidget(self.asr_run_btn)
        asr_layout.addStretch()
        
        self.tabs.addTab(asr_tab, "ASR (Speech-to-Text)")
        
        # --- TAB 4: Auto TTS Dataset Builder ---
        auto_tab = QWidget()
        auto_layout = QVBoxLayout(auto_tab)
        
        auto_info = QLabel("<b>Integrated Pipeline:</b><br>"
                           "1. Separation (Vocal Extraction)<br>"
                           "2. Speaker Diarization<br>"
                           "3. ASR Transcription<br>"
                           "Produces metadata.csv and organized speaker folders for TTS training.")
        auto_info.setWordWrap(True)
        auto_layout.addWidget(auto_info)
        
        auto_io_group = QGroupBox("One-Click Dataset Creation")
        auto_io_layout = QVBoxLayout()
        
        auto_in_layout = QHBoxLayout()
        self.auto_input_path = QLineEdit()
        self.auto_input_path.setPlaceholderText("Select Audio File(s) or Folder")
        btn_auto_file = QPushButton("Files")
        btn_auto_file.clicked.connect(lambda: self.browse_generic(self.auto_input_path, 'files'))
        btn_auto_folder = QPushButton("Folder")
        btn_auto_folder.clicked.connect(lambda: self.browse_generic(self.auto_input_path, 'dir'))
        auto_in_layout.addWidget(self.auto_input_path)
        auto_in_layout.addWidget(btn_auto_file)
        auto_in_layout.addWidget(btn_auto_folder)
        auto_io_layout.addWidget(QLabel("Input Audio:"))
        auto_io_layout.addLayout(auto_in_layout)
        
        auto_out_layout = QHBoxLayout()
        self.auto_output_dir = QLineEdit("tts_dataset_output")
        btn_auto_out = QPushButton("Browse")
        btn_auto_out.clicked.connect(lambda: self.browse_generic(self.auto_output_dir, True))
        auto_out_layout.addWidget(self.auto_output_dir)
        auto_out_layout.addWidget(btn_auto_out)
        auto_io_layout.addWidget(QLabel("Output Root Directory:"))
        auto_io_layout.addLayout(auto_out_layout)
        
        lang_layout = QHBoxLayout()
        self.auto_lang = QComboBox()
        for code, name in self.lang_dict.items():
            self.auto_lang.addItem(f"{name} ({code})", code)
        lang_layout.addWidget(QLabel("ASR Language:"))
        lang_layout.addWidget(self.auto_lang)
        
        self.auto_asr_model = QComboBox()
        self.auto_asr_model.addItem("Whisper Large v3", "whisper")
        self.auto_asr_model.addItem("OWSM-CTC v4", "owsmv4")
        lang_layout.addWidget(QLabel("ASR Model:"))
        lang_layout.addWidget(self.auto_asr_model)
        
        auto_io_layout.addLayout(lang_layout)
        
        self.auto_ensemble_mode = QCheckBox("Enable Ensemble Mode for Separation Step")
        self.auto_ensemble_mode.setChecked(True)
        auto_io_layout.addWidget(self.auto_ensemble_mode)
        
        auto_io_group.setLayout(auto_io_layout)
        auto_layout.addWidget(auto_io_group)
        
        self.auto_run_btn = QPushButton("BUILD TTS DATASET")
        self.auto_run_btn.setFixedHeight(60)
        self.auto_run_btn.setStyleSheet("background-color: #d73a49; color: white; font-size: 18px; font-weight: bold; border-radius: 8px;")
        self.auto_run_btn.clicked.connect(self.start_auto_pipeline)
        auto_layout.addWidget(self.auto_run_btn)
        auto_layout.addStretch()
        
        self.tabs.addTab(auto_tab, "Auto TTS Builder")
        
        # --- TAB 5: Clean Dataset ---
        clean_tab = QWidget()
        clean_layout = QVBoxLayout(clean_tab)
        
        clean_info = QLabel("<b>Dataset Cleaner:</b><br>"
                            "Removes audio files shorter than the specified duration or those with empty text transcriptions.<br>"
                            "It also correctly cleans up <i>list.txt</i> and <i>metadata.csv</i> if they exist in the target folder.")
        clean_info.setWordWrap(True)
        clean_layout.addWidget(clean_info)
        
        clean_group = QGroupBox("Clean Settings")
        clean_group_layout = QVBoxLayout()
        
        clean_dir_layout = QHBoxLayout()
        self.clean_input_dir = QLineEdit()
        self.clean_input_dir.setPlaceholderText("Select TTS Dataset Directory to Clean")
        btn_clean_dir = QPushButton("Browse")
        btn_clean_dir.clicked.connect(lambda: self.browse_generic(self.clean_input_dir, True))
        clean_dir_layout.addWidget(QLabel("Target Directory:"))
        clean_dir_layout.addWidget(self.clean_input_dir)
        clean_dir_layout.addWidget(btn_clean_dir)
        
        clean_dur_layout = QHBoxLayout()
        self.clean_min_dur = QDoubleSpinBox()
        self.clean_min_dur.setRange(0.0, 10.0)
        self.clean_min_dur.setSingleStep(0.1)
        self.clean_min_dur.setValue(1.0)
        clean_dur_layout.addWidget(QLabel("Min Dur (sec):"))
        clean_dur_layout.addWidget(self.clean_min_dur)
        clean_dur_layout.addStretch()
        
        clean_group_layout.addLayout(clean_dir_layout)
        clean_group_layout.addLayout(clean_dur_layout)
        clean_group.setLayout(clean_group_layout)
        
        clean_layout.addWidget(clean_group)
        
        self.clean_run_btn = QPushButton("CLEAN DATASET")
        self.clean_run_btn.setFixedHeight(50)
        self.clean_run_btn.setStyleSheet("background-color: #fca311; color: white; font-size: 16px; font-weight: bold; border-radius: 6px;")
        self.clean_run_btn.clicked.connect(self.start_clean)
        clean_layout.addWidget(self.clean_run_btn)
        clean_layout.addStretch()
        
        self.tabs.addTab(clean_tab, "Clean Dataset")
    
        right_layout.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #30363d; border-radius: 5px; text-align: center; background-color: #0d1117; color: #c9d1d9; }
            QProgressBar::chunk { background-color: #58a6ff; border-radius: 4px; }
        """)
        right_layout.addWidget(self.progress_bar)
        
        right_layout.addWidget(QLabel("Logs:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #0d1117; color: #58a6ff; font-family: Consolas;")
        right_layout.addWidget(self.log_text)
        
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 2)

    def apply_dark_theme(self):
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(22, 27, 34))
        palette.setColor(QPalette.WindowText, QColor(201, 209, 217))
        palette.setColor(QPalette.Base, QColor(13, 17, 23))
        palette.setColor(QPalette.AlternateBase, QColor(22, 27, 34))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, QColor(201, 209, 217))
        palette.setColor(QPalette.Button, QColor(33, 38, 45))
        palette.setColor(QPalette.ButtonText, QColor(201, 209, 217))
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(88, 166, 255))
        palette.setColor(QPalette.Highlight, QColor(31, 111, 235))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        self.setPalette(palette)
        
        self.setStyleSheet("""
            QGroupBox { border: 1px solid #30363d; border-radius: 6px; margin-top: 20px; font-weight: bold; color: #58a6ff; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
            QLineEdit, QComboBox, QListWidget { border: 1px solid #30363d; border-radius: 6px; padding: 5px; background-color: #0d1117; color: #c9d1d9; }
            QLineEdit:focus, QComboBox:focus { border: 1px solid #58a6ff; }
            QPushButton { background-color: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 5px 10px; color: #c9d1d9; }
            QPushButton:hover { background-color: #30363d; border-color: #8b949e; }
        """)

    def filter_models(self):
        search_text = self.search_input.text().lower()
        self.model_list.clear()
        
        for name, data in self.model_info.items():
            if search_text in name.lower():
                item = QListWidgetItem(name)
                path = data.get('target_position')
                installed = os.path.exists(path) if path else False
                
                if installed:
                    item.setForeground(QColor("#3fb950"))
                    item.setText(f"{name} [Installed]")
                else:
                    item.setForeground(QColor("#8b949e"))
                    
                item.setData(Qt.UserRole, data)
                self.model_list.addItem(item)

    def on_model_selected(self, item):
        data = item.data(Qt.UserRole)
        name = data.get('model_name')
        model_type = data.get('model_type', 'Unknown')
        size = data.get('model_size', 0) / 1024 / 1024
        
        path = data.get('target_position')
        installed = os.path.exists(path) if path else False
        
        info_text = f"Name: {name}\nType: {model_type}\nSize: {size:.2f} MB\n"
        info_text += f"Installed: {'Yes' if installed else 'No'}"
        
        self.model_info_label.setText(info_text)
        
        if not installed and data.get('link'):
            self.download_btn.setEnabled(True)
            self.download_btn.setText(f"Download {name}")
        else:
            self.download_btn.setEnabled(False)
            self.download_btn.setText("Detailed Info Above")

    def download_current_model(self):
        current_item = self.model_list.currentItem()
        if not current_item:
            return
            
        data = current_item.data(Qt.UserRole)
        name = data.get('model_name')
        
        self.download_btn.setEnabled(False)
        self.append_log(f"Starting download: {name}...")
        
        config = {"model": name}
        
        self.progress_bar.setValue(0)
        # Use 'download' task which we'll add to runner.py
        self.worker = ProcessWorker("download", config)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.append_log)
        self.worker.task_success.connect(self.on_download_finished)
        self.worker.error.connect(self.on_download_error)
        self.worker.start()

    def on_download_finished(self):
        self.download_btn.setEnabled(True)
        self.filter_models() # Refreshes [Installed] status
        QMessageBox.information(self, "Download Finished", "The model has been downloaded and is ready to use.")

    def on_download_error(self, err_msg):
        self.append_log(f"Download Error: {err_msg}")
        self.download_btn.setEnabled(True)
        QMessageBox.warning(self, "Download Failed", f"Failed to download model:\n{err_msg}")

    def browse_custom_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model Checkpoint", "", "Checkpoints (*.ckpt *.th *.pth)")
        if path:
            self.custom_path.setText(path)

    def register_custom_model(self):
        name = self.custom_name.text()
        path = self.custom_path.text()
        m_type = self.custom_type.currentText()
        
        if not name or not path:
            QMessageBox.warning(self, "Error", "Name and Path are required")
            return
            
        class Args:
            def __init__(self):
                self.name = name
                self.path = path
                self.type = m_type
                self.config = "" 
        
        import runner
        try:
            runner.add_custom_model(Args())
            self.append_log(f"Custom model {name} registered.")
            self.model_info = runner.load_models_info()
            self.filter_models()
        except Exception as e:
            self.append_log(f"Error registering model: {e}")

    def browse_generic(self, line_edit, mode):
        if mode == 'dir':
            path = QFileDialog.getExistingDirectory(self, "Select Directory")
            if path:
                line_edit.setText(path)
        elif mode == 'files':
            paths, _ = QFileDialog.getOpenFileNames(self, "Select Audio Files", "", "Audio Files (*.wav *.mp3 *.flac)")
            if paths:
                line_edit.setText("; ".join(paths))
        elif mode == 'save_dir':
            path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if path:
                line_edit.setText(path)
        else: 
            path, _ = QFileDialog.getOpenFileName(self, "Select File", "", "Audio Files (*.wav *.mp3 *.flac)")
            if path:
                line_edit.setText(path)
                
    def browse_output(self):
        self.browse_generic(self.output_dir, 'dir')

    def browse_dia_output(self):
        self.browse_generic(self.dia_output_dir, 'dir')
        
    def browse_asr_output(self):
        self.browse_generic(self.asr_output_dir, 'dir')

    def start_diarization(self):
        input_file = self.dia_input_path.text()
        inputs = [i.strip() for i in input_file.split(";") if i.strip()]
        if not inputs or not all(os.path.exists(i) for i in inputs):
            QMessageBox.warning(self, "Error", "One or more input paths are invalid")
            return

        config = {
            "input": input_file,
            "output_dir": self.dia_output_dir.text(),
            "device": self.dia_device.currentText(),
            "batch_size": int(self.dia_batch_size.currentText()),
            "audiosr_steps": int(self.dia_sr_steps.text() or "50"),
            "process_overlap": self.dia_process_overlap.isChecked()
        }

        self.dia_run_btn.setEnabled(False)
        self.append_log(f"Starting Speaker Diarization on {input_file}...")
        
        self.progress_bar.setValue(0)
        self.worker = ProcessWorker("diarization", config)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.append_log)
        self.worker.task_success.connect(self.on_diarization_finished)
        self.worker.error.connect(self.on_dia_error)
        self.worker.start()

    def on_diarization_finished(self):
        self.progress_bar.setValue(100)
        self.dia_run_btn.setEnabled(True)
        QMessageBox.information(self, "Finished", "Speaker Diarization completed.")

    def on_dia_error(self, err_msg):
        self.append_log(f"Diarization Error: {err_msg}")
        self.dia_run_btn.setEnabled(True)

    def start_asr(self):
        input_file = self.asr_input_path.text()
        inputs = [i.strip() for i in input_file.split(";") if i.strip()]
        if not inputs or not all(os.path.exists(i) for i in inputs):
            QMessageBox.warning(self, "Error", "One or more input paths are invalid")
            return

        config = {
            "input": input_file,
            "output_dir": self.asr_output_dir.text(),
            "device": self.asr_device.currentText(),
            "language": self.asr_lang.currentData(),
            "model_type": self.asr_model_combo.currentData()
        }

        self.asr_run_btn.setEnabled(False)
        self.append_log(f"Starting ASR on {input_file}...")
        
        self.progress_bar.setValue(0)
        self.worker = ProcessWorker("asr", config)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.append_log)
        self.worker.task_success.connect(self.on_asr_finished)
        self.worker.error.connect(self.on_asr_error)
        self.worker.start()

    def on_asr_finished(self):
        self.progress_bar.setValue(100)
        self.asr_run_btn.setEnabled(True)
        QMessageBox.information(self, "Finished", "ASR completed successfully.")

    def on_asr_error(self, err_msg):
        self.append_log(f"ASR Error: {err_msg}")
        self.asr_run_btn.setEnabled(True)

    def start_auto_pipeline(self):
        input_file = self.auto_input_path.text()
        output_dir = self.auto_output_dir.text()
        
        inputs = [i.strip() for i in input_file.split(";") if i.strip()]
        if not inputs or not all(os.path.exists(i) for i in inputs):
            QMessageBox.warning(self, "Error", "One or more input paths are invalid")
            return
            
        selected_items = self.model_list.selectedItems()
        if not selected_items:
             QMessageBox.warning(self, "Error", "Please select at least one separation model from the left panel")
             return

        models_config = []
        for item in selected_items:
            data = item.data(Qt.UserRole)
            models_config.append({
                "model_name": data["model_name"],
                "model_path": data["target_position"],
                "model_type": data.get("model_type"),
                "model_class": data.get("model_class", "")
            })

        sep_config = {
            "models": models_config,
            "ensemble": self.auto_ensemble_mode.isChecked()
        }
        
        asr_config = {
            "language": self.auto_lang.currentData(),
            "model_type": self.auto_asr_model.currentData()
        }

        self.auto_run_btn.setEnabled(False)
        self.append_log(">>> Starting Integrated TTS Data Builder Pipeline...")
        
        self.progress_bar.setValue(0)
        config = {
            "input_file": input_file,
            "output_dir": output_dir,
            "sep_config": sep_config,
            "asr_config": asr_config
        }
        self.worker = ProcessWorker("pipeline", config)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.append_log)
        self.worker.task_success.connect(self.on_auto_finished)
        self.worker.error.connect(self.on_auto_error)
        self.worker.start()

    def on_auto_finished(self):
        self.progress_bar.setValue(100)
        self.auto_run_btn.setEnabled(True)
        QMessageBox.information(self, "Success", "TTS Dataset Builder completed.")

    def on_auto_error(self, err_msg):
        self.append_log(f"Pipeline Error: {err_msg}")
        self.auto_run_btn.setEnabled(True)

    def start_clean(self):
        target_dir = self.clean_input_dir.text().strip()
        if not target_dir or not os.path.isdir(target_dir):
            QMessageBox.warning(self, "Error", "Invalid dataset directory.")
            return
            
        config = {
            "target_dir": target_dir,
            "min_duration": self.clean_min_dur.value()
        }
        
        self.clean_run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.append_log(f">>> Cleaning Dataset in {target_dir}...")
        
        self.worker = ProcessWorker("clean", config)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.append_log)
        self.worker.task_success.connect(self.on_clean_finished)
        self.worker.error.connect(self.on_clean_error)
        self.worker.start()

    def on_clean_finished(self):
        self.progress_bar.setValue(100)
        self.clean_run_btn.setEnabled(True)
        QMessageBox.information(self, "Finished", "Dataset cleaning completed.")

    def on_clean_error(self, err_msg):
        self.append_log(f"Cleaner Error: {err_msg}")
        self.clean_run_btn.setEnabled(True)

    @Slot(int)
    def on_tab_changed(self, index):
        if index == 0 or index == 3: 
            self.left_panel.setVisible(True)
        else: 
            self.left_panel.setVisible(False)

    def start_inference(self):
        input_file = self.input_path.text()
        inputs = [i.strip() for i in input_file.split(";") if i.strip()]
        if not inputs or not all(os.path.exists(i) for i in inputs):
            QMessageBox.warning(self, "Error", "One or more input paths are invalid")
            return

        selected_items = self.model_list.selectedItems()
        if not selected_items:
             QMessageBox.warning(self, "Error", "Please select at least one model from the list")
             return
             
        models_config = []
        for item in selected_items:
            data = item.data(Qt.UserRole)
            model_name = data.get('model_name')
            model_path = data.get('target_position')
            
            if not model_path or not os.path.exists(model_path):
                 QMessageBox.warning(self, "Error", f"Model {model_name} is not installed. Please download it first.")
                 return
            
            models_config.append({
                "model_name": model_name,
                "model_path": model_path,
                "model_type": data.get('model_type'),
                "model_class": data.get('model_class', '')
            })
        
        config = {
            "input": input_file,
            "models": models_config,
            "ensemble": self.ensemble_mode.isChecked(),
            "output_dir": self.output_dir.text(),
            "device": self.device.currentText(),
            "output_format": self.output_format.currentText(),
            "normalize": self.normalize.isChecked(),
            "use_tta": self.use_tta.isChecked()
        }
        
        self.run_btn.setEnabled(False)
        model_names = ", ".join([m["model_name"] for m in models_config])
        self.append_log(f"Starting inference on {input_file} using {model_names}...")
        
        self.progress_bar.setValue(0)
        self.worker = ProcessWorker("inference", config)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.append_log)
        self.worker.task_success.connect(self.on_inference_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_inference_finished(self):
        self.progress_bar.setValue(100)
        self.run_btn.setEnabled(True)
        QMessageBox.information(self, "Finished", "Process completed. Check logs for details.")

    def on_error(self, err_msg):
        self.append_log(f"Error: {err_msg}")
        self.run_btn.setEnabled(True)
        self.download_btn.setEnabled(True)

    @Slot(str)
    def append_log(self, text):
        self.log_text.append(text)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    app = QApplication(sys.argv)
    
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # 1. 스플래시(로딩) 화면 생성
    splash = QWidget()
    splash.setWindowTitle("VoxUnravel")
    splash.setFixedSize(500, 350)
    splash.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
    splash.setStyleSheet("background-color: #0d1117; border: 1px solid #30363d;")
    
    screen = app.primaryScreen().availableGeometry()
    splash.move(screen.center() - splash.rect().center())
    
    splash_layout = QVBoxLayout(splash)
    splash_layout.setAlignment(Qt.AlignCenter)
    
    splash_title = QLabel("VoxUnravel")
    splash_title.setAlignment(Qt.AlignCenter)
    splash_title.setStyleSheet("font-size: 42px; font-weight: bold; color: #58a6ff;")
    splash_layout.addWidget(splash_title)
    
    splash_sub = QLabel("Audio Separation & Speaker Diarization")
    splash_sub.setAlignment(Qt.AlignCenter)
    splash_sub.setStyleSheet("font-size: 14px; color: #8b949e; margin-bottom: 20px;")
    splash_layout.addWidget(splash_sub)
    
    status_label = QLabel("Initializing engine...")
    status_label.setAlignment(Qt.AlignCenter)
    status_label.setStyleSheet("font-size: 13px; color: #c9d1d9; margin-bottom: 10px;")
    splash_layout.addWidget(status_label)
    
    splash_bar = QProgressBar()
    splash_bar.setRange(0, 0) 
    splash_bar.setFixedWidth(400)
    splash_bar.setFixedHeight(4)
    splash_bar.setTextVisible(False)
    splash_bar.setStyleSheet("""
        QProgressBar { border: none; background-color: #21262d; border-radius: 2px; }
        QProgressBar::chunk { background-color: #58a6ff; border-radius: 2px; }
    """)
    splash_layout.addWidget(splash_bar)
    
    splash.show()
    
    # 메인 윈도우를 담아둘 전역 변수
    main_window = None

    # 2. 로딩이 정상적으로 끝났을 때의 콜백 (이벤트 기반 비동기 처리)
    def on_init_finished(data):
        global main_window
        main_window = MainWindow(models_info=data)
        main_window.move(screen.center() - main_window.rect().center())
        main_window.show()
        main_window.append_log("System Ready. AI Libraries & Models loaded.")
        splash.close()

    # 3. 로딩 중 에러 발생 시의 콜백
    def on_init_error(err_msg):
        global main_window
        print(f"Initialization Error: {err_msg}")
        QMessageBox.critical(None, "Initialization Error", f"Failed to initialize background environments:\n{err_msg}")
        # 에러가 나더라도 사용자가 확인할 수 있도록 기본 UI는 띄움
        main_window = MainWindow(models_info={})
        main_window.move(screen.center() - main_window.rect().center())
        main_window.show()
        main_window.append_log(f"Critical Init Error: {err_msg}")
        splash.close()

    # 4. 백그라운드 워커 시작 (while 루프 완전 삭제됨)
    init_worker = InitWorker()
    init_worker.progress_text.connect(status_label.setText)
    init_worker.finished.connect(on_init_finished)
    init_worker.error.connect(on_init_error)
    init_worker.start()

    # Qt 이벤트 루프 진입
    sys.exit(app.exec())