use anyhow::{Result, anyhow};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

// Symphonia: The Pure Rust Audio Decoder
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use ndarray::{Array2, Axis};
use rustfft::{Fft, FftPlanner, num_complex::Complex};

/// 🚀 THE PHYSICS ENGINE
/// Loads any audio file, converts to f32, mixes to Stereo, and Normalizes.
/// Returns: (Interleaved Samples, Sample Rate)
pub fn load_and_normalize(path: &str) -> Result<(Vec<f32>, u32)> {
    // 1. OPEN THE INPUT
    // We create a "Media Source" from the file path.
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
                    // "decoded" is a raw AudioBuffer (could be Int16, Int24, Float32)
                    // We must standardize everything to Float32 [-1.0, 1.0]

                    // We create a specific f32 buffer to copy data into
                    let spec = *decoded.spec();
                    let duration = decoded.capacity() as u64;
                    let mut sample_buf =
                        symphonia::core::audio::SampleBuffer::<f32>::new(duration, spec);

                    // Copy and Convert (Int -> Float) happens here
                    sample_buf.copy_interleaved_ref(decoded);

                    // Extend our main storage
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

// ----------------------------------------------------------------
// 2. THE SPECTRAL ENGINE (New Code)
// ----------------------------------------------------------------
pub struct SpectralEngine {
    fft: Arc<dyn Fft<f32>>,
    ifft: Arc<dyn Fft<f32>>,
    window: Vec<f32>,
    fft_size: usize,
    hop_size: usize,
}

impl SpectralEngine {
    pub fn new(fft_size: usize, hop_size: usize) -> Self {
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);

        // Hanning Window: Reduces spectral leakage at edges
        // w[n] = 0.5 * (1 - cos(2*pi*n / N))
        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (fft_size as f32)).cos())
            })
            .collect();

        Self {
            fft,
            ifft,
            window,
            fft_size,
            hop_size,
        }
    }

    /// Forward Transform: Audio (Time) -> Spectrogram (Freq)
    pub fn stft(&self, input: &[f32]) -> Array2<Complex<f32>> {
        // Calculate number of frames
        // (Length - FFT_Size) / Hop + 1
        if input.len() < self.fft_size {
            return Array2::zeros((1, 1));
        }

        let n_frames = (input.len() - self.fft_size) / self.hop_size + 1;
        let mut spectrogram = Array2::<Complex<f32>>::zeros((self.fft_size, n_frames));
        let mut buffer = vec![Complex::new(0.0, 0.0); self.fft_size];

        for (i, mut col) in spectrogram.axis_iter_mut(Axis(1)).enumerate() {
            let start = i * self.hop_size;
            let end = start + self.fft_size;

            // Windowing
            for (j, (&x, &w)) in input[start..end].iter().zip(&self.window).enumerate() {
                buffer[j] = Complex::new(x * w, 0.0);
            }

            // FFT
            self.fft.process(&mut buffer);

            // Store in Matrix
            for j in 0..self.fft_size {
                col[j] = buffer[j];
            }
        }
        spectrogram
    }

    /// Inverse Transform: Spectrogram (Freq) -> Audio (Time)
    pub fn istft(&self, spectrogram: &Array2<Complex<f32>>) -> Vec<f32> {
        let n_frames = spectrogram.len_of(Axis(1));
        let output_len = (n_frames - 1) * self.hop_size + self.fft_size;
        let mut output = vec![0.0f32; output_len];

        // This buffer tracks how much "Window Energy" was added to each pixel.
        // We divide by this at the end to normalize the Overlap-Add.
        let mut normalization = vec![1e-10f32; output_len]; // Avoid div/0

        let mut buffer = vec![Complex::new(0.0, 0.0); self.fft_size];

        for (i, col) in spectrogram.axis_iter(Axis(1)).enumerate() {
            let start = i * self.hop_size;

            // Copy Frequency Data to Buffer
            for j in 0..self.fft_size {
                buffer[j] = col[j];
            }

            // Inverse FFT
            self.ifft.process(&mut buffer);

            // Overlap-Add Logic
            for j in 0..self.fft_size {
                // rustfft is unnormalized, so we scale by 1/N
                let val = buffer[j].re / (self.fft_size as f32);

                // Add to output buffer
                output[start + j] += val;

                // Track the window weight (OLA Normalization)
                // Note: We only windowed ONCE (during STFT).
                // Standard COLA constraint assumes the window sums to 1.
                // Since we used Hanning, we normalize by the window values added.
                normalization[start + j] += self.window[j];
            }
        }

        // Apply Normalization (Average the overlaps)
        for (out, norm) in output.iter_mut().zip(normalization.iter()) {
            *out /= norm;
        }

        output
    }
}
