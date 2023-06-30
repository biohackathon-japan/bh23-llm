import sys
import time
import argparse
from pathlib import Path

import openai
import whisper
import sounddevice
from scipy.io.wavfile import write


def transcribe_remote(prompt: str = "", filepath: str = "", token: str = "") -> str:
    """Sends an audio file to OpenAI Whisper API."""
    print("TRANSCRIBING started")

    openai.api_key = token
    with open(filepath, "rb") as f:
        result = openai.Audio.transcribe(
            "whisper-1", # there is only one model available
            f,
            language="en",
            temperature=0.2,
            prompt=prompt,
        )

    print("TRANSCRIBING ended")
    return result["text"]


def transcribe_local(prompt: str = "", filepath: str = "", model: str = "") -> str:
    """Reads an audio file and transcribes it using a local Whisper model."""
    print("TRANSCRIBING started")

    model = whisper.load_model(model)
    result = model.transcribe(
        filepath,
        temperature=0.2,
        initial_prompt=prompt,
    )

    print("TRANSCRIBING ended")
    return result["text"]


def record_audio(duration: int = 0):
    """Records audio and saves it to timestamp.wav."""
    sample_rate = 44100
    filepath = f"{str(int(time.time()))}.wav"

    print(f"RECORDING now for {duration}s")
    recording = sounddevice.rec(
        int(sample_rate * duration),
        samplerate=sample_rate,
        channels=2,
    )
    sounddevice.wait()
    write(filepath, sample_rate, recording)
    print("RECORDING ended")

    return filepath


def parse_arguments(arguments):
    """Parse command line arguments and options."""
    parser = argparse.ArgumentParser(description="Small OpenAI Whisper transcriber CLI for DBCLS BioHackathon 2023.")
    parser.add_argument(
        "model",
        default="local",
        choices=["local", "remote"],
        help="local model runs on your CPU, remote model sends a request to OpenAI Whisper API",
    )
    parser.add_argument(
        "-lm",
        "--local-model",
        default="tiny.en",
        choices=["tiny.en", "base.en", "small.en"],
        help="Whisper model to use, tiny=39M, base=74M, small=244M params",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=0,
        help="duration of recording in seconds",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        help="prompt for the model to focus transcription",
    )
    parser.add_argument(
        "-t",
        "--token",
        help="OpenAPI token is using the remote model",
    )
    parser.add_argument(
        "-pc",
        "--preserve-recording",
        action="store_true",
        help="the recording is saved to a file and will be deleted, unless this option is used",
    )
    if len(sys.argv) <= 1:
        # If no command line arguments were given, print help text
        parser.print_help()
        sys.exit(0)
    return parser.parse_args(arguments)


def main(arguments=None):

    a = parse_arguments(arguments)

    filepath = record_audio(duration=a.duration)

    result = ""
    if a.model == "remote":
        result = transcribe_remote(prompt=a.prompt, filepath=filepath, token=a.token)
    else:
        result = transcribe_local(prompt=a.prompt, filepath=filepath, model=a.local_model)
    print(result)

    if not a.preserve_recording:
        Path(filepath).unlink()


if __name__ == "__main__":
    main()
