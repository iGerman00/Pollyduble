print("Loading libraries...")

import os
import shutil
import subprocess
import json
import argparse
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
from faster_whisper import WhisperModel
from voicefixer import VoiceFixer
from audio_separator.separator import Separator

device = "cuda" if torch.cuda.is_available() else "cpu"

import warnings
warnings.filterwarnings("ignore")

# Constants
config = {
'OUTPUT_DIR': 'Pollyduble/output',
'INPUT_FILE': 'Pollyduble/input/trimmed.webm',
'REFERENCE_SPEAKER': 'Pollyduble/input/voice_sample2.mp4',
'CKPT_CONVERTER': 'OpenVoice/checkpoints_v2/converter',
'BASE_SPEAKERS': 'OpenVoice/checkpoints_v2/base_speakers/ses',
'TTS_LANGUAGE': 'EN_NEWEST',
'SPEED': 1.0
}

def create_voice_sample(input_file, output_file):
    subprocess.run(['ffmpeg', '-i', input_file, '-ss', '00:00:00', '-t', '00:00:15', '-vn', '-acodec', 'pcm_s16le', '-ac', '1', '-y', output_file], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_file

def args_parser():
    parser = argparse.ArgumentParser(description='Auto dubbing script')
    parser.add_argument('-i', '--input', help='Input audio/video file', required=True)
    parser.add_argument('-o', '--output', help='Output directory', required=False, default='Polydub/output')
    parser.add_argument('-v', '--voice', help='Voice sample. If not provided, a voice sample will be created from the input audio file', required=False)
    parser.add_argument('-f', '--fix', help='Fix audio using VoiceFixer', action='store_true')
    parser.add_argument('-s', '--separate', help='Separate audio using audio-separator (UVR-based). Implies --mux', action='store_true')
    parser.add_argument('-m', '--mux', help='Mux audio with separated instrumental audio (best used for video inputs). Implied by --separate', action='store_true')
    args = parser.parse_args()
    return args

def init_folders():
    print("Initializing folders at %s" % config['OUTPUT_DIR'])
    os.makedirs(f'{config["OUTPUT_DIR"]}', exist_ok=True)
    os.makedirs(f'{config["OUTPUT_DIR"]}/tmp', exist_ok=True)
    os.makedirs(f'{config["OUTPUT_DIR"]}/tmp/stretched', exist_ok=True)
    os.makedirs(f'{config["OUTPUT_DIR"]}/tts', exist_ok=True)

# Initialize whisper model
def init_whisper_model():
    print(f"Using device: {device}")
    whisper_model_size = "large-v2"
    return WhisperModel(whisper_model_size, device=device, compute_type="float16")

# Initialize tone color converter
def init_tone_color_converter():
    tone_color_converter = ToneColorConverter(f'{config["CKPT_CONVERTER"]}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{config["CKPT_CONVERTER"]}/checkpoint.pth')
    return tone_color_converter

# Get target speaker embedding
def get_target_se(tone_color_converter):
    return se_extractor.get_se(config['REFERENCE_SPEAKER'], tone_color_converter, vad=False)

# Transcribe audio
def transcribe_audio(whisper_model):
    language = None
    segments, info = whisper_model.transcribe(config['INPUT_FILE'], beam_size=5, language=language, task='translate')
    if language is None:
        print("Transcription/translation initialized with detected language '%s' (probability %f)" % (info.language, info.language_probability))
    print("Iterating...")
    # transcription_array = [{'start': segment.start, 'end': segment.end, 'text': segment.text} for segment in segments]
    transcription_array = []
    for segment in segments:
        print(f"Start: {segment.start}, End: {segment.end}, Text: {segment.text}")
        transcription_array += [{'start': segment.start, 'end': segment.end, 'text': segment.text}]
    return transcription_array

# Generate TTS audio
def generate_tts_audio(transcription_array, target_se, tone_color_converter):
    texts = [transcription['text'] for transcription in transcription_array]
    stretched_audio_files = []
    for idx, text in enumerate(texts):
        model = TTS(language=config['TTS_LANGUAGE'], device=device)
        speaker_ids = model.hps.data.spk2id
        for speaker_key in speaker_ids.keys():
            speaker_id = speaker_ids[speaker_key]
            speaker_key = speaker_key.lower().replace('_', '-')
            source_se = torch.load(f'{config["BASE_SPEAKERS"]}/{speaker_key}.pth', map_location=device)

            tts_temp_output = f'{config["OUTPUT_DIR"]}/tmp/tmp.wav'
            model.tts_to_file(text, speaker_id, tts_temp_output, speed=config['SPEED'])
            save_path = f'{config["OUTPUT_DIR"]}/tts/{idx}_{speaker_key}.wav'

            # Run the tone color converter
            # encode_message = "Powered by @MyShell's OpenVoice, Melo and German's auto-dubbing script"
            encode_message = "AI-cloned voice"
            tone_color_converter.convert(
                audio_src_path=tts_temp_output, 
                src_se=source_se, 
                tgt_se=target_se, 
                output_path=save_path,
                message=encode_message)
            print(f"Generated {save_path}")

            # Remove silence
            silence_removed_path = f'{config["OUTPUT_DIR"]}/tmp/{idx}_{speaker_key}_silence_removed.wav'
            subprocess.run(['ffmpeg', '-i', save_path, '-af', 'silenceremove=1:0:-50dB', '-y', silence_removed_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(save_path)
            shutil.move(silence_removed_path, save_path)

            # Stretch audio to match the target duration
            target_duration = transcription_array[idx]['end'] - transcription_array[idx]['start']
            probe = subprocess.check_output(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', save_path])
            probe = json.loads(probe)
            tts_output_duration = float(probe['format']['duration'])
            speed_multiplier = tts_output_duration / target_duration
            print(f"Speed multiplier: {speed_multiplier}")

            stretched_audio_path = f'{config["OUTPUT_DIR"]}/tmp/stretched/{idx}_{speaker_key}_stretched.wav'

            silence_padding = ''

            # Pad with silence if too slow
            if speed_multiplier < 0.7:
                speed_multiplier = 0.7
                padding_duration = (target_duration - tts_output_duration)
                padding_duration = round(padding_duration * 1-speed_multiplier, 4) 
                if padding_duration < 0:
                    padding_duration = 0
                silence_padding = f',apad=pad_dur={padding_duration}'

            if (idx + 1) < len(transcription_array):
                next_segment_start = transcription_array[idx + 1]['start']
                silence_duration = next_segment_start - transcription_array[idx]['end']
                # Add silence between segments if needed
                if silence_duration > 0.01:
                    silence_duration = round(silence_duration, 4)
                    if silence_duration < 0:
                        silence_duration = 0
                    silence_padding += f',apad=pad_dur={silence_duration}'

            speed_multiplier = str(round(speed_multiplier, 4))

            filter_string = f'atempo={speed_multiplier}{silence_padding}'
            print(f"Stretching audio with filter: {filter_string}")

            subprocess.run(['ffmpeg', '-i', save_path, '-filter:a', f'{filter_string}', '-y', stretched_audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            stretched_audio_files.append(stretched_audio_path)

    return stretched_audio_files

# Combine audio files
def combine_audio_files(stretched_audio_files):
    print("Combining audio files...")
    combined_audio_path = f'{config["OUTPUT_DIR"]}/concatenated_audio.wav'
    with open('concat_list.txt', 'w') as f:
        for file in stretched_audio_files:
            f.write(f"file '{file}'\n")
    subprocess.run(['ffmpeg', '-f', 'concat', '-safe', '0', '-i', 'concat_list.txt', '-c', 'copy', '-y', combined_audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Combined audio file saved to {combined_audio_path}")
    # ask user if they'd like to ffplay the combined audio
    ffplay = input("Do you want to play the combined audio? (y/n): ")
    if ffplay.lower() == 'y':
        subprocess.run(['ffplay', '-autoexit', combined_audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return combined_audio_path

def voicefixer_audio(input_file, output_file):
    is_cuda = torch.cuda.is_available()

    print("Initializing VoiceFixer...")
    voicefixer = VoiceFixer()
    voicefixer.restore(input=input_file, output=output_file, cuda=is_cuda, mode=0)

def separate_audio(input_file):
    separator = Separator()
    separator.load_model()

    separator.separate(input_file)

    output_files = separator.separate(input_file)

    vocal_file = [file for file in output_files if "(Vocals)" in file][0]
    instrumental_file = [file for file in output_files if "(Instrumental)" in file][0]
    shutil.move(vocal_file, f'{config["OUTPUT_DIR"]}/tmp/vocals.wav')
    shutil.move(instrumental_file, f'{config["OUTPUT_DIR"]}/tmp/instrumental.wav')
    files = {
        'vocals': f'{config["OUTPUT_DIR"]}/tmp/vocals.wav',
        'instrumental': f'{config["OUTPUT_DIR"]}/tmp/instrumental.wav'
    }
    return files

def mux_audio(video_file, vocals, instrumental, output_file):
    subprocess.run([
        'ffmpeg',
        '-i', video_file,
        '-i', vocals,
        '-i', instrumental,
        '-filter_complex', '[1:a]loudnorm=i=-23:tp=-2:lra=11[a];[a][2:a]amix=inputs=2:duration=longest,volume=2[out]',
        '-map', '0:v',
        '-map', '[out]',
        '-c:v', 'copy',
        '-ar', '48000',
        '-ac', '2',
        '-c:a', 'aac',
        output_file
    ])

    return output_file

# Main function
def main():
    args = args_parser()
    separated_audio_files = {}

    config['OUTPUT_DIR'] = os.path.abspath(args.output)
    config['INPUT_FILE'] = os.path.abspath(args.input)

    original_input_file = config['INPUT_FILE']

    init_folders()

    if args.separate or args.mux:
        separated_audio_files = separate_audio(config['INPUT_FILE'])
        config['INPUT_FILE'] = separated_audio_files['vocals']
        args.mux = True
        args.separate = True
    if args.voice:
        config['REFERENCE_SPEAKER'] = os.path.abspath(args.voice)
    else:
        config['REFERENCE_SPEAKER'] = os.path.abspath(f'{config["OUTPUT_DIR"]}/voice_sample.wav')

    # check_output_dir()

    if not args.voice:
        print("Creating voice sample...")
        create_voice_sample(config['INPUT_FILE'], f'{config["OUTPUT_DIR"]}/voice_sample.wav')

    whisper_model = init_whisper_model()
    tone_color_converter = init_tone_color_converter()
    target_se, audio_name = get_target_se(tone_color_converter)
    transcription_array = transcribe_audio(whisper_model)
    stretched_audio_files = generate_tts_audio(transcription_array, target_se, tone_color_converter)
    combine_audio_files(stretched_audio_files)

    if args.fix:
        voicefixer_audio(f'{config["OUTPUT_DIR"]}/concatenated_audio.wav', f'{config["OUTPUT_DIR"]}/voicefixed_audio.wav')

    if args.mux:
        # vocals file is our translated combined audio
        translated_vocals = f'{config["OUTPUT_DIR"]}/concatenated_audio.wav'
        if args.fix:
            translated_vocals = f'{config["OUTPUT_DIR"]}/voicefixed_audio.wav'

        mux_audio(original_input_file, translated_vocals, separated_audio_files['instrumental'], f'{config["OUTPUT_DIR"]}/muxed_output.mp4')

if __name__ == "__main__":
    main()