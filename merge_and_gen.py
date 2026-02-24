import csv
import re
import os
import wave

metadata_file = r"d:\VoxUnravel\vox_test\metadata.csv"
output_dir = r"d:\VoxUnravel\docs\assets\speakers_merged"
os.makedirs(output_dir, exist_ok=True)

speaker_data = {}
with open(metadata_file, "r", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="|")
    header = next(reader)
    for row in reader:
        if len(row) < 3: continue
        file_path, speaker_id, trans = row
        file_path = file_path.replace("C:/Users/user/Downloads/vox_test", "d:/VoxUnravel/vox_test")
        
        filename = os.path.basename(file_path)
        m = re.search(r'_(\d+\.\d+)\.wav', filename)
        if m:
            time_val = float(m.group(1))
        else:
            time_val = 0.0
            
        if speaker_id not in speaker_data:
            speaker_data[speaker_id] = []
        speaker_data[speaker_id].append((time_val, file_path, trans))

html_output = []

# Sort speakers to display them nicely
speakers_sorted = sorted(speaker_data.keys())

for speaker in speakers_sorted:
    items = speaker_data[speaker]
    items.sort(key=lambda x: x[0])
    
    out_path = os.path.join(output_dir, f"{speaker}_merged.wav")
    
    in_files = [f for _, f, _ in items]
    if not in_files: continue
    
    try:
        with wave.open(in_files[0], 'rb') as w_in:
            params = w_in.getparams()
            nchannels = w_in.getnchannels()
            sampwidth = w_in.getsampwidth()
            framerate = w_in.getframerate()
            
        silence_frames = int(framerate * 0.5)
        silence_data = b'\x00' * (silence_frames * nchannels * sampwidth)
        
        with wave.open(out_path, 'wb') as w_out:
            w_out.setparams(params)
            for fpath in in_files:
                if not os.path.exists(fpath):
                    # fix path if needed
                    fpath = fpath.replace("\\", "/")
                try:
                    with wave.open(fpath, 'rb') as w_in:
                        data = w_in.readframes(w_in.getnframes())
                        w_out.writeframes(data)
                    w_out.writeframes(silence_data)
                except Exception as e:
                    print(f"Skipping {fpath}: {e}")
                
        full_text = " ".join([t for _,_, t in items])
        speaker_name = speaker
        if speaker == "Speaker_0": speaker_name += " (무전 내용)"
        if speaker == "Speaker_1": speaker_name += " (브루스 배너/헐크)"
        if speaker == "Speaker_2": speaker_name += " (토니 스타크/아이언맨)"
        if speaker == "Speaker_3": speaker_name += " (토르/로키)"
        if speaker == "Speaker_4": speaker_name += " (스티브 로저스/캡틴 아메리카)"
        if speaker == "Unknown_0": speaker_name += " (기타 소음/알수없음)"
        
        html_output.append(f'''
            <!-- {speaker} -->
            <div class="audio-item">
                <div class="speaker-info">
                    <span class="speaker-name">🗣️ {speaker_name}</span>
                    <span class="speaker-text">"{full_text}"</span>
                </div>
                <audio controls>
                    <source src="assets/speakers_merged/{speaker}_merged.wav" type="audio/wav">
                </audio>
            </div>''')
    except Exception as e:
        print(f"Error {speaker}: {e}")

html_content = "\n".join(html_output)
with open(r"d:\VoxUnravel\merged_html_snippet.txt", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Done")
