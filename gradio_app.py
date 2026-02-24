import gradio as gr
import os
import shutil
from pipeline import VoxUnravelPipeline

def process_audio(audio_file, out_dir_name, use_separation, asr_lang):
    if not audio_file:
        return "Please upload an audio file!"
    
    # Output path setting
    output_dir = os.path.join("/content/VoxUnravel/outputs", out_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Configuration setup
    sep_config = {"models": [{"model_name": "htdemucs"}], "ensemble": False} if use_separation else None
    dia_config = {"batch_size": 8, "audiosr_steps": 50, "process_overlap": True}
    asr_config = {"language": "auto" if asr_lang == "Auto" else asr_lang, "model_type": "whisper"}
    
    pipeline = VoxUnravelPipeline(device="cuda")
    
    try:
        pipeline.run_full_process(
            input_audio=audio_file, 
            output_dir=output_dir, 
            sep_config=sep_config, 
            dia_config=dia_config, 
            asr_config=asr_config
        )
        return f"✅ Processing completed successfully!\nOutput saved at: {output_dir}"
    except Exception as e:
        return f"❌ Error occurred: {str(e)}"

# Gradio Interface design
with gr.Blocks(title="VoxUnravel Web UI") as demo:
    gr.Markdown("# 🎧 VoxUnravel - Web GUI (Colab Only)")
    gr.Markdown("Pipeline GUI for vocal separation, speaker diarization, and speech-to-text (ASR).")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Audio File (.wav, .mp3)")
            output_name = gr.Textbox(value="my_project", label="Output Folder Name")
            
            use_sep = gr.Checkbox(label="Use Background/Vocal Separation", value=True)
            language = gr.Dropdown(choices=["Auto", "ko", "en", "ja"], value="Auto", label="ASR Language")
            
            run_btn = gr.Button("Run", variant="primary")
            
        with gr.Column():
            result_output = gr.Textbox(label="Execution Logs and Results", lines=10)

    # Trigger function on button click
    run_btn.click(
        fn=process_audio,
        inputs=[audio_input, output_name, use_sep, language],
        outputs=result_output
    )

if __name__ == "__main__":
    # share=True provides a public URL for external access.
    demo.launch(share=True, debug=True)
