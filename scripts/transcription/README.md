# Transcription
Small CLI for recording audio, and transcribing it using a local Whisper model or a remote API.

Requirements
```
apt install libportaudio2 # for sounddevice python library
pip install -r requirements.txt
```

Usage
```
$ python transcribe.py --help

usage: transcribe.py [-h] [-lm {tiny.en,base.en,small.en}] [-d DURATION] [-p PROMPT] [-t TOKEN] [-pc] {local,remote}

Small OpenAI Whisper transcriber CLI for DBCLS BioHackathon 2023.

positional arguments:
  {local,remote}        local model runs on your CPU, remote model sends a request to OpenAI Whisper API

options:
  -h, --help            show this help message and exit
  -lm {tiny.en,base.en,small.en}, --local-model {tiny.en,base.en,small.en}
                        Whisper model to use, tiny=39M, base=74M, small=244M params
  -d DURATION, --duration DURATION
                        duration of recording in seconds
  -p PROMPT, --prompt PROMPT
                        prompt for the model to focus transcription
  -t TOKEN, --token TOKEN
                        OpenAPI token is using the remote model
  -pc, --preserve-recording
                        the recording is saved to a file and will be deleted, unless this option is used
```

Example
```
python transcribe.py local --local-model small.en --duration 5 --prompt "medical terminology"
```
