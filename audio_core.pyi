# audio_core.pyi
import numpy as np

def load_track(path: str) -> tuple[np.ndarray, int]:
    """
    Loads an audio file via Rust/Symphonia, mixes to Stereo,
    and Peak Normalizes to -1.0 dB.

    Args:
        path (str): Path to the audio file.

    Returns:
        tuple[np.ndarray, int]: (Audio Samples [Float32], Sample Rate)
    """
    ...

def debug_spectral_roundtrip(path: str) -> tuple[np.ndarray, int]:
    """
    Runs the full Physics Engine (Load -> STFT -> ISTFT) on a file.
    Returns the reconstructed audio.

    Args:
        path (str): Path to the audio file.

    Returns:
        tuple[np.ndarray, int]: (Reconstructed Audio, Sample Rate)
    """
    ...
