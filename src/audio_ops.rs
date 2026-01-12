// src/audio_ops.rs
use anyhow::{Result, anyhow};
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use symphonia::core::codecs::CODEC_TYPE_NULL;
use symphonia::core::errors::Error;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;

use ndarray::{Array2, Axis};
use rustfft::{Fft, FftPlanner, num_complex::Complex};

/// Loads audio, ensures it is Stereo (Interleaved), and Normalizes.
/// Returns: (Interleaved Samples [L, R, L, R...], Sample Rate)
pub fn load_and_normalize(path: &str) -> Result<(Vec<f32>, u32)> {
    let src = File::open(Path::new(path))?;
    let mss = MediaSourceStream::new(Box::new(src), Default::default());
    let hint = Hint::new();

    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &Default::default(),
        &Default::default(),
    )?;

    let mut format = probed.format;

    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| anyhow!("WARNING: No supported audio track found"))?;

    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(44100); //ALAC type: 44100

    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &Default::default())?;
    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(Error::IoError(_)) => break,
            Err(e) => return Err(anyhow!("Error decoding packet: {}", e)),
        };

        if packet.track_id() == track_id {
            match decoder.decode(&packet) {
                Ok(decoded) => {
                    let spec = *decoded.spec();
                    let duration = decoded.capacity() as u64;
                    let mut sample_buf =
                        symphonia::core::audio::SampleBuffer::<f32>::new(duration, spec);
                    sample_buf.copy_interleaved_ref(decoded);
                    let samples = sample_buf.samples();

                    if spec.channels.count() == 1 {
                        // Case: Mono Input. We must duplicate to create Stereo [L, R]
                        for &sample in samples {
                            all_samples.push(sample); // Left
                            all_samples.push(sample); // Right
                        }
                    } else {
                        // Case: Already Stereo (or multi-channel). Keep as is.
                        all_samples.extend_from_slice(samples);
                    }
                }
                Err(Error::IoError(_)) => break,
                Err(e) => return Err(anyhow!("Error decoding: {}", e)),
            }
        }
    }

    // Normalization (Applied to Interleaved Stereo preserves balance)
    let max_amp = all_samples.iter().fold(0.0f32, |acc, &x| acc.max(x.abs()));
    if max_amp > 1e-6 {
        let gain = 0.905 / max_amp;
        for sample in all_samples.iter_mut() {
            *sample *= gain;
        }
    }

    Ok((all_samples, sample_rate))
}

/// Helper to run the SpectralEngine on Stereo data
pub fn process_stereo(engine: &SpectralEngine, interleaved_input: &[f32]) -> Vec<f32> {
    // 1. De-interleave [L, R, L, R...] -> [L...], [R...]
    let left_in: Vec<f32> = interleaved_input.iter().step_by(2).copied().collect();
    let right_in: Vec<f32> = interleaved_input
        .iter()
        .skip(1)
        .step_by(2)
        .copied()
        .collect();

    // 2. Process Left
    let spec_l = engine.stft(&left_in);
    let left_out = engine.istft(&spec_l);

    // 3. Process Right
    let spec_r = engine.stft(&right_in);
    let right_out = engine.istft(&spec_r);

    // 4. Re-interleave
    let len = left_out.len().min(right_out.len());
    let mut output = Vec::with_capacity(len * 2);

    for i in 0..len {
        output.push(left_out[i]);
        output.push(right_out[i]);
    }

    output
}

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
        println!("RUST: WOLA Spectral Engine Initialized");
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(fft_size);
        let ifft = planner.plan_fft_inverse(fft_size);

        let window: Vec<f32> = (0..fft_size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (fft_size as f32)).cos())
            })
            .collect();

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
        let mut padded_input = vec![0.0f32; input.len() + 2 * pad_len];
        padded_input[pad_len..pad_len + input.len()].copy_from_slice(input);

        let n_frames = (padded_input.len() - self.fft_size) / self.hop_size + 1;
        let mut spectrogram = Array2::<Complex<f32>>::zeros((self.fft_size, n_frames));
        let mut buffer = vec![Complex::new(0.0, 0.0); self.fft_size];

        for (i, mut col) in spectrogram.axis_iter_mut(Axis(1)).enumerate() {
            let start = i * self.hop_size;
            if start + self.fft_size > padded_input.len() {
                break;
            }
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

    pub fn istft(&self, spectrogram: &Array2<Complex<f32>>) -> Vec<f32> {
        let n_frames = spectrogram.len_of(Axis(1));
        let padded_len = (n_frames - 1) * self.hop_size + self.fft_size;
        let mut output = vec![0.0f32; padded_len];
        let mut buffer = vec![Complex::new(0.0, 0.0); self.fft_size];

        for (i, col) in spectrogram.axis_iter(Axis(1)).enumerate() {
            let start = i * self.hop_size;
            for j in 0..self.fft_size {
                buffer[j] = col[j];
            }
            self.ifft.process(&mut buffer);
            for j in 0..self.fft_size {
                output[start + j] += (buffer[j].re / self.fft_size as f32) * self.window[j];
            }
        }
        for sample in output.iter_mut() {
            *sample /= self.scaling_factor;
        }

        let pad_len = self.fft_size / 2;
        if output.len() > 2 * pad_len {
            output[pad_len..output.len() - pad_len].to_vec()
        } else {
            output
        }
    }
}
