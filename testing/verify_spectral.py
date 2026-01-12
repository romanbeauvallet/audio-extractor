import audio_core
import soundfile as sf
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch, stft
import librosa

# ---------------------------------------------------------
# 1. SETUP: Change this to your audio file
# ---------------------------------------------------------
INPUT_FILE = "audio/test_reference.wav"
OUTPUT_FILE = "audio/spectral_roundtrip_output.wav"


def test_spectral_integrity():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: File '{INPUT_FILE}' not found.")
        return

    print("🧪 Starting Spectral Round-Trip Test...")
    print(f"   Input: '{INPUT_FILE}'")

    # ---------------------------------------------------------
    # 2. RUN RUST (Load -> STFT -> ISTFT -> Return)
    # ---------------------------------------------------------
    try:
        audio_data, sample_rate = audio_core.debug_spectral_roundtrip(INPUT_FILE)
    except Exception as e:
        print(f"❌ Rust Error: {e}")
        return

    print("✅ Rust finished processing.")
    print(f"   Output Samples: {len(audio_data)}")
    print(f"   Sample Rate: {sample_rate} Hz")

    # ---------------------------------------------------------
    # 3. SAVE AND VERIFY
    # ---------------------------------------------------------
    # Rust returns a flat 1D array [L, R, L, R...].
    # We need to guess channels to save it correctly as a WAV.

    # Try Stereo first (Standard for music)
    if len(audio_data) % 2 == 0:
        # channels = 2
        reshaped_audio = audio_data.reshape(-1, 2)
    else:
        # Fallback to Mono
        print("   ℹ️  Note: Odd number of samples detected. Saving as Mono.")
        # channels = 1
        reshaped_audio = audio_data

    print(f"💾 Saving to '{OUTPUT_FILE}'...")
    sf.write(OUTPUT_FILE, reshaped_audio, sample_rate)

    print("-" * 40)
    print(f"SUCCESS! File saved: {os.path.abspath(OUTPUT_FILE)}")
    print("🎧 LISTEN NOW: Compare the INPUT vs. the output.")
    print(
        "   They should sound IDENTICAL. If you hear 'robotic' artifacts, the windowing math is wrong."
    )
    print("-" * 40)


def load_mono(path):
    """
    Loads audio using Librosa (Independent of Rust).
    Replicates the Rust normalization logic so we can compare apples to apples.
    """
    data, sr = librosa.load(path, sr=None, mono=False)

    # 2. Handle Stereo -> Mono (Average)
    if data.ndim > 1:
        # data is (2, N), we want average across axis 0
        data = np.mean(data, axis=0)

    # 3. MANUAL NORMALIZATION (To match Rust's 0.905 peak)
    max_amp = np.max(np.abs(data))
    if max_amp > 1e-6:
        gain = 0.905 / max_amp
        data *= gain

    return data, sr


def normalize(sig):
    """Normalize to unit energy for fair PSD comparison"""
    return sig / np.max(np.abs(sig))


def plot_analysis():
    print(f"🦀 Calls Rust Engine to process: {INPUT_FILE}")

    # This triggers the 'println!' in Rust and runs the new WOLA math
    reconstructed_audio, sr = audio_core.debug_spectral_roundtrip(INPUT_FILE)

    # Save the NEW result to disk (Overwriting the old bad file)
    # Rust returns interleaved [L, R], reshape if stereo
    if len(reconstructed_audio) % 2 == 0:
        audio_to_save = reconstructed_audio.reshape(-1, 2)
    else:
        audio_to_save = reconstructed_audio

    sf.write(OUTPUT_FILE, audio_to_save, sr)
    print(f"✅ New WOLA output saved to: {OUTPUT_FILE}")

    print(f"📊 Analyzing: {INPUT_FILE} vs {OUTPUT_FILE}")

    # 1. Load Data
    sig_in, sr_in = load_mono(INPUT_FILE)
    sig_out, sr_out = load_mono(OUTPUT_FILE)

    assert sr_in == sr_out, "Sample rates must match!"
    sr = sr_in

    # 2. Alignment (Trim to shortest length)
    min_len = min(len(sig_in), len(sig_out))
    sig_in = normalize(sig_in[:min_len])
    sig_out = normalize(sig_out[:min_len])

    # -----------------------------------------------------
    # A. TIME DOMAIN (Residuals)
    # -----------------------------------------------------
    residual = sig_in - sig_out
    mse = np.mean(residual**2)
    print(f"📉 Time Domain MSE: {mse:.8f}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Plot 1: Waveforms
    axes[0].set_title("Time Domain: Signal Overlay")
    axes[0].plot(sig_in[10000:11000], label="INPUT", alpha=0.7)
    axes[0].plot(
        sig_out[10000:11000], label="Reconstruction", alpha=0.7, linestyle="--"
    )
    axes[0].legend()
    axes[0].set_ylabel("Amplitude")

    # -----------------------------------------------------
    # B. FREQUENCY DOMAIN (PSD using Welch's Method)
    # -----------------------------------------------------
    f_in, psd_in = welch(sig_in, sr, nperseg=4096)
    f_out, psd_out = welch(sig_out, sr, nperseg=4096)

    axes[1].set_title("Frequency Domain: Power Spectral Density (PSD)")
    axes[1].semilogy(f_in, psd_in, label="INPUT", linewidth=1.5)
    axes[1].semilogy(
        f_out, psd_out, label="Reconstruction", linestyle="--", linewidth=1.5
    )
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Power / Hz (dB)")
    axes[1].legend()
    axes[1].grid(True, which="both", ls="-", alpha=0.5)

    # -----------------------------------------------------
    # C. SPECTROGRAM DIFFERENCE
    # -----------------------------------------------------
    # Compute STFT
    f, t, Zxx_in = stft(sig_in, fs=sr, nperseg=1024)
    _, _, Zxx_out = stft(sig_out, fs=sr, nperseg=1024)

    # Magnitude Spectrograms
    spec_in = np.abs(Zxx_in)
    spec_out = np.abs(Zxx_out)

    # Log Difference (Visualizing the Error Floor)
    # We add a tiny epsilon to avoid log(0)
    diff = np.abs(spec_in - spec_out) + 1e-9
    log_diff = 20 * np.log10(diff)

    axes[2].set_title("Spectral Error Map (Difference Spectrogram)")
    im = axes[2].pcolormesh(t, f, log_diff, shading="gouraud", cmap="inferno")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[2].set_xlabel("Time (s)")
    fig.colorbar(im, ax=axes[2], label="Error Magnitude (dB)")

    plt.tight_layout()
    plt.savefig("physics_results/spectral_roundtrip_analysis.png")
    plt.show()


if __name__ == "__main__":
    plot_analysis()
