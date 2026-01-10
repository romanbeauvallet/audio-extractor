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