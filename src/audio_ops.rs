use anyhow::{Result, anyhow};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use ndarray::{Array2, Axis};
use rustfft::{Fft, FftPlanner, num_complex::Complex};

/// Loads any audio file, converts to f32, mixes to Stereo, and Normalizes.
/// Returns: (Interleaved Samples, Sample Rate)
pub fn load_and_normalize(path: &str) -> Result<(Vec<f32>, u32)> {
    // 1. OPEN THE INPUT
    let src = File::open(Path::new(path))?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());

    // 2. PROBE THE FORMAT (Physics: Detecting the signal type)
    // Symphonia checks magic bytes (headers) to see if it's MP3/WAV/etc.
    let hint = Hint::new();
    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;

    let mut format = probed.format;

    // 3. FIND THE AUDIO TRACK
    // A file might have video or subtitles. We want the first Audio wave.
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("No supported audio track found"))?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100);

    // 4. CREATE THE DECODER
    // This is the mathematical engine that unzips the compressed audio.
    let dec_opts: DecoderOptions = Default::default();
    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &dec_opts)?;

    // Vector to hold the raw audio energy
    let mut all_samples: Vec<f32> = Vec::new();

    // 5. DECODE LOOP (The Signal Flow)
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(_)) => break, // End of Stream (EOF)
            Err(e) => return Err(anyhow!("Error decoding packet: {}", e)),
        };

        if packet.track_id() == track_id {
            match decoder.decode(&packet) {
                Ok(decoded) => {
                    // We must standardize everything to Float32 [-1.0, 1.0]

                    // We create a specific f32 buffer to copy data into
                    let spec = *decoded.spec();
                    let duration = decoded.capacity() as u64;
                    let mut sample_buf =
                        symphonia::core::audio::SampleBuffer::<f32>::new(duration, spec);

                    // Copy and Convert (Int -> Float) happens here
                    sample_buf.copy_interleaved_ref(decoded);

                    all_samples.extend_from_slice(sample_buf.samples());
                }
                Err(Error::IoError(_)) => break,
                Err(e) => return Err(anyhow!("Error during decoding: {}", e)),
            }
        }
    }

    // 6. PHYSICS: PEAK NORMALIZATION
    // We want the loudest moment in the song to hit exactly -1.0 dB.
    // This maximizes Signal-to-Noise ratio for the Neural Network.

    // A. Find the Global Max Energy (Amplitude)
    let max_amp = all_samples.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));

    // B. Calculate Gain Factor
    // Target is -1.0 dB approx
    let target_peak = 0.905;

    // Avoid division by zero (silence protection)
    if max_amp > 1e-6 {
        let gain = target_peak / max_amp;

        // C. Apply Gain (Scalar Multiplication)
        for sample in all_samples.iter_mut() {
            *sample *= gain;
        }
    }

    Ok((all_samples, sample_rate))
}

// THE SPECTRAL ENGINE (New Code)
// src/audio_ops.rs

pub struct SpectralEngine {
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    fft_size: usize,
    hop_size: usize,
    scaling_factor: f32,
}

impl SpectralEngine {
    pub fn new(fft_size: usize, hop_size: usize) -> Self {
        println!("🦀 RUST: WOLA Spectral Engine Initialized (Fixed!)");
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);

        // 1. Generate Hanning Window
        // w[n] = 0.5 * (1 - cos(2*pi*n / N))
        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (fft_size as f32)).cos())
            })
            .collect();

        // 2. Calculate WOLA Scaling Factor
        // We apply the window TWICE (Analysis + Synthesis).
        // For COLA (Constant Overlap Add) compliance with Hanning:
        // Factor = Sum(Window^2) / Hop_Size
        let window_sq_sum: f32 = window.iter().map(|&x| x * x).sum();
        let scaling_factor = window_sq_sum / (hop_size as f32);

        Self {
            fft,
            ifft,
            window,
            fft_size,
            hop_size,
            scaling_factor,
        }
    }

    pub fn stft(&self, input: &[f32]) -> Array2<Complex<f32>> {
        let pad_len = self.fft_size / 2;

        // Create input with zeros at start/end
        let mut padded_input = vec![0.0f32; input.len() + 2 * pad_len];
        padded_input[pad_len..pad_len + input.len()].copy_from_slice(input);

        // Run FFT on padded data
        let n_frames = (padded_input.len() - self.fft_size) / self.hop_size + 1;
        let mut spectrogram = Array2::<Complex<f32>>::zeros((self.fft_size, n_frames));
        let mut buffer = vec![Complex::new(0.0, 0.0); self.fft_size];

        for (i, mut col) in spectrogram.axis_iter_mut(Axis(1)).enumerate() {
            let start = i * self.hop_size;
            // Safety check
            if start + self.fft_size > padded_input.len() {
                break;
            }

            // Apply Window
            for (j, (&x, &w)) in padded_input[start..start + self.fft_size]
                .iter()
                .zip(&self.window)
                .enumerate()
            {
                buffer[j] = Complex::new(x * w, 0.0);
            }

            self.fft.process(&mut buffer);

            for j in 0..self.fft_size {
                col[j] = buffer[j];
            }
        }
        spectrogram
    }

    // 2. ISTFT
    pub fn istft(&self, spectrogram: &Array2<Complex<f32>>) -> Vec<f32> {
        let n_frames = spectrogram.len_of(Axis(1));

        // Calculate expected length based on PADDED frames
        let padded_len = (n_frames - 1) * self.hop_size + self.fft_size;
        let mut output = vec![0.0f32; padded_len];
        let mut buffer = vec![Complex::new(0.0, 0.0); self.fft_size];

        for (i, col) in spectrogram.axis_iter(Axis(1)).enumerate() {
            let start = i * self.hop_size;

            // Load Freq Domain
            for j in 0..self.fft_size {
                buffer[j] = col[j];
            }

            // Inverse FFT -> Time Domain
            self.ifft.process(&mut buffer);

            // Weighted Overlap-Add (WOLA)
            for j in 0..self.fft_size {
                let val = buffer[j].re / (self.fft_size as f32);
                let windowed_val = val * self.window[j];
                output[start + j] += windowed_val;
            }
        }

        // Apply Global Scaling
        for sample in output.iter_mut() {
            *sample /= self.scaling_factor;
        }

        // REMOVE PADDING (Critical for Alignment)
        let pad_len = self.fft_size / 2;
        if output.len() > 2 * pad_len {
            // Trim the zeros we added at the start
            // And trim the end to match original length (approximately)
            output[pad_len..output.len() - pad_len].to_vec()
        } else {
            output
        }
    }
}
