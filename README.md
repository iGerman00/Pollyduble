# Pollyduble
> Automatic Dubbing with Voice Cloning and Speech Recognition  
> Made possible thanks to [OpenVoice](https://github.com/myshell-ai/OpenVoice), [MeloTTS](https://github.com/myshell-ai/MeloTTS), [Faster Whisper](https://github.com/SYSTRAN/faster-whisper), [VoiceFixer](https://github.com/haoheliu/voicefixer), [python-audio-separator](https://github.com/karaokenerds/python-audio-separator) and [FFmpeg](https://ffmpeg.org/).

<p align="center">
  <img src="assets/polly.png" alt="Polly the Tadpole" width="200"/>
</p>
This is a highly experimental prototype of a script that aims to automatically dub English audio over a video file originally recorded in any language Whisper supports.  
Theoretically, with some modifications and different OpenVoice models, it should support any language supported by OpenVoice, however the translation would have to be handled by something other than Whisper

## Features
- Voice cloning and local text-to-speech synthesis
- Automatic speech recognition
- Audio separation
- Automatic synchronization of dubbed lines to the original speech
- Optional voice fixing to bring back some high frequencies lost during the voice cloning process
- Muxing the dubbed audio and extracted instrumental track back into the video

--- 

PRs are welcome, this is mostly just a proof-of-concept. Some good ideas for improvement include:
- Speaker diarization to separate the speech of different characters and automatically assign the correct dubbed lines to the correct characters
- Ability to load custom subtitles instead of relying on automatic speech recognition
- A translation neural network (local is highly preferred) or API to not rely on Whisper's shoddy translations

## Pre-requisites

- Python 3.9
- FFmpeg, FFprobe and FFplay installed on your system and **in PATH**
- Windows (only tested on Windows)
- A modern NVIDIA GPU with CUDA support is probably required
- Miniconda or Anaconda (optional, but recommended)

## Installation

0. Install FFmpeg, FFprobe and FFplay on your system and make sure they are in PATH. You can download them from [here](https://ffmpeg.org/download.html).

1. Make a new directory and clone this repository:
```bash
git clone https://github.com/igerman00/Pollyduble
cd Pollyduble
```

2. Create a new Conda environment:
```bash
conda create -n dubbing python=3.9
```

3. Activate the Conda environment:
```bash
conda activate dubbing
```

4. Clone the OpenVoice repository
```bash
git clone https://github.com/myshell-ai/OpenVoice
```
> Make sure the OpenVoice repository is in the same directory as this repository, it should be named "OpenVoice".

5. Install OpenVoice:
```bash
cd OpenVoice
pip install -e .
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -m unidic download
```

6. Install `torch` with GPU support (the index-url parameter should be optional for no GPU support):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

7. Install the other dependencies:
```bash
cd .. # Go back to the root directory of the repo
pip install -r requirements-win-cu118.txt
```

## Usage

1. Obtain a video file and place it anywhere on your computer, for this example we will assume it is in the same directory as our `demo.py` script, and it is named `video.mp4`.
2. Run the script:
```bash
python demo.py -i video.mp4 -s -m
```

The output will be stored in the `Pollyduble/output` directory by default. It will contain various files including the dubbed video, the separated audio, the dubbed audio, and the voice sample. Mostly, it should be one-click.
> Options include:
> - `-i` or `--input` to specify the input video file
> - `-o` or `--output` to specify the output directory (default is `Pollyduble/output`)
> - `-v` or `--voice` to specify a custom sample for the voice cloning. If not specified, one will be created from the first 15 seconds of the video
> - `-s` or `--separate` to enable audio separation, i.e. extracting the background music and the speech from the video separately
> - `-m` or `--mux` to enable muxing the separated audio back into the video with the dubbed speech
> - `-f` or `--fix` to enable voice fixing, i.e. improving the quality of the dubbed speech.  
> *^ Experimental and doesn't actually sound that good most of the time.*
> - `--help` to display the help message

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
