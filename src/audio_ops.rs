use anyhow::{anyhow, Result};
use std::fs::File;
use std::path::Path;

// Symphonia: The Pure Rust Audio Decoder
use symphonia::core::audio::{AudioBuffer, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use symphonia::core::conv::IntoSample;

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

    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fmt_opts, &meta_opts)?;

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
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &dec_opts)?;

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
                    let mut sample_buf = symphonia::core::audio::SampleBuffer::<f32>::new(duration, spec);
                    
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
    // Target is -1.0 dB approx 0.89125
    let target_peak = 0.89125;
    
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