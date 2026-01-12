import audio_core
import numpy as np
import soundfile as sf
import os

# ---------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------
INPUT_FILE = "audio/test_reference.wav"
OUTPUT_FILE = "audio/loading_ref.wav"


def test_rust_loading():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Could not find file '{INPUT_FILE}'")
        return

    print(f"🦀 Rust Engine: Loading '{INPUT_FILE}'...")

    try:
        samples, sr = audio_core.load_track(INPUT_FILE)
    except Exception as e:
        print(f"❌ Rust Crash: {e}")
        return

    print("✅ Rust Load Complete!")
    print(f"   Sample Rate: {sr} Hz")
    print(f"   Shape: {samples.shape}")

    peak = np.max(np.abs(samples))
    print(f"   Peak Amplitude: {peak:.5f}")

    # ---------------------------------------------------------
    # 2. SAVE BACK TO DISK (Using soundfile)
    # ---------------------------------------------------------
    print("\n💾 Saving to WAV...")

    # Rust returns INTERLEAVED audio [L, R, L, R...] as a 1D array.
    # We must reshape it to 2D [Samples, Channels] for the WAV writer.
    channels = 2
    if len(samples) % channels == 0:
        stereo_audio = samples.reshape(-1, channels)

        sf.write(OUTPUT_FILE, stereo_audio, sr)
        print(f"✅ Saved to: {os.path.abspath(OUTPUT_FILE)}")
        print("🎧 Go listen to it! It should sound perfect.")
    else:
        print("❌ Error: Sample count is not divisible by 2. Is this Mono?")


if __name__ == "__main__":
    test_rust_loading()
