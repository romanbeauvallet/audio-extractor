import audio_core
import torch
import torchaudio
import os

# ---------------------------------------------------------
# 1. SETUP: Put the path to a REAL file on your laptop here
# ---------------------------------------------------------
INPUT_FILE = "my_test_song.mp3"  
OUTPUT_FILE = "rust_output_verified.wav"

def test_rust_loading():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Could not find file '{INPUT_FILE}'")
        print("   Please edit the script and provide a valid path.")
        return

    print(f"🦀 Rust Engine: Loading '{INPUT_FILE}'...")
    
    # CALL RUST
    # This runs the 'load_and_normalize' function we wrote in audio_ops.rs
    try:
        samples, sr = audio_core.load_track(INPUT_FILE)
    except Exception as e:
        print(f"❌ Rust Crash: {e}")
        return

    print("✅ Rust Load Complete!")
    print(f"   Sample Rate: {sr} Hz")
    print(f"   Shape: {samples.shape} (Samples,)")
    print(f"   Dtype: {samples.dtype}")

    # PHYSICS CHECK: Normalization
    # We normalized to ~0.891 (-1.0 dB). Let's verify.
    peak = abs(samples).max()
    print(f"   Peak Amplitude: {peak:.5f}")
    if 0.88 < peak < 0.90:
        print("   ✅ Normalization Logic: PASSED (-1.0 dB)")
    else:
        print("   ⚠️ Normalization Logic: DEVIATION (Expected ~0.891)")

    # ---------------------------------------------------------
    # 2. SAVE BACK TO DISK (To listen)
    # ---------------------------------------------------------
    print("\n💾 Saving to WAV to verify audio integrity...")
    
    # Convert Numpy -> Torch
    tensor = torch.from_numpy(samples)
    
    # Symphonia returns interleaved Stereo [L, R, L, R...]
    # We need to reshape for TorchAudio: [Channels, Time]
    if sr > 0: # Avoid div by zero
        # Reshape to (Time, 2)
        tensor = tensor.view(-1, 2)
        # Transpose to (2, Time)
        tensor = tensor.t()
    
    torchaudio.save(OUTPUT_FILE, tensor, sr)
    print(f"✅ Saved to: {os.path.abspath(OUTPUT_FILE)}")
    print("🎧 Open this file and verify it sounds correct!")

if __name__ == "__main__":
    test_rust_loading()