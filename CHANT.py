import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

class CHANTSynthesizer:
    def __init__(self, sr=44100):
        self.sr = sr
        
    def _generate_fof_grain(self, fc, bw, amp, phase, tex_seconds=None):
        """
        Generates a single FOF grain (one burst).
        
        Parameters:
        fc : Center Frequency (Hz)
        bw : Bandwidth (Hz)
        amp : Linear Amplitude
        phase : Initial phase (radians)
        tex_seconds : Duration of the attack skirt (seconds). 
                      If None, defaults to 1 period of fc (classic heuristic).
        """
        
        # 1. Calculate Decay Factor (Alpha)
        # Relates bandwidth to exponential decay: alpha = pi * bandwidth
        alpha = np.pi * bw
        
        # 2. Determine Grain Length
        # We stop generating when the envelope drops below -60dB (0.001)
        # e^(-alpha * t) = 0.001  =>  -alpha * t = ln(0.001)
        if alpha == 0: alpha = 0.001 # prevent div by zero
        decay_duration = -np.log(0.001) / alpha
        
        # Auto-calculate attack time if not provided (heuristic: 1/2 period to 1 period)
        if tex_seconds is None:
            tex_seconds = 1.0 / fc if fc > 0 else 0.001
            
        total_duration = decay_duration + tex_seconds
        num_samples = int(total_duration * self.sr)
        t = np.arange(num_samples) / self.sr
        
        # 3. Generate The Sinusoid
        omega = 2 * np.pi * fc
        sinusoid = np.sin(omega * t + phase)
        
        # 4. Generate The Envelope (The "FOF" shape)
        # Part A: The Skirt (Raised Cosine Attack)
        tex_samples = int(tex_seconds * self.sr)
        
        envelope = np.zeros(num_samples)
        
        # Attack region (0 to tex)
        # Formula: .5 * (1 - cos(pi * t / tex))
        if tex_samples > 0:
            t_att = t[:tex_samples]
            envelope[:tex_samples] = 0.5 * (1 - np.cos(np.pi * t_att / tex_seconds))
            # Ensure continuity at the peak of attack
            envelope[tex_samples:] = 1.0 
        else:
            envelope[:] = 1.0
            
        # Part B: The Exponential Decay (applied to whole, but effective after attack peak)
        # Note: Rodet's papers sometimes imply decay starts at t=0, sometimes at t=tex.
        # Accurate CHANT implementation usually applies decay *starting* from t=0 
        # but the attack window masks the start.
        decay = np.exp(-alpha * t)
        
        # Combine
        final_envelope = envelope * decay
        
        # Scale by amplitude
        return sinusoid * final_envelope * amp

    def synthesize_vowel(self, f0, duration, formants, vib_rate=5.0, vib_depth=0.0):
        """
        Synthesizes a sound by overlapping FOF grains.
        
        formants: List of dicts [{'freq': 600, 'bw': 80, 'amp_db': 0}, ...]
        """
        total_samples = int(duration * self.sr)
        output_buffer = np.zeros(total_samples + 44100) # padding for tails
        
        # Time pointer
        t_ptr = 0.0
        sample_idx = 0
        
        while sample_idx < total_samples:
            # Calculate current F0 (with simple Vibrato)
            current_time = sample_idx / self.sr
            # Vibrato calculation
            pitch_mod = 1.0 + (vib_depth * np.sin(2 * np.pi * vib_rate * current_time))
            current_f0 = f0 * pitch_mod
            
            period_samples = int(self.sr / current_f0)
            
            # Trigger FOFs for this period
            for fmt in formants:
                # Convert dB to linear
                linear_amp = 10 ** (fmt['amp_db'] / 20.0)
                
                grain = self._generate_fof_grain(
                    fc=fmt['freq'], 
                    bw=fmt['bw'], 
                    amp=linear_amp, 
                    phase=0
                )
                
                # Overlap-Add into buffer
                grain_len = len(grain)
                # Boundary check
                if sample_idx + grain_len < len(output_buffer):
                    output_buffer[sample_idx : sample_idx + grain_len] += grain
            
            # Advance time by one pitch period
            sample_idx += period_samples
            
        # Trim silence and normalize
        output = output_buffer[:total_samples]
        mx = np.max(np.abs(output))
        if mx > 0:
            output /= mx
            
        return output

# --- Example Configuration: Male Voice /a/ (Tenor) ---
# Parameters derived from standard CHANT libraries (Rodet)
# Formant 1: 650Hz, BW 80Hz, 0dB
# Formant 2: 1100Hz, BW 90Hz, -6dB
# Formant 3: 2860Hz, BW 120Hz, -20dB (The "Singing Formant" region)
# Formant 4: 3300Hz, BW 130Hz, -30dB
# Formant 5: 4500Hz, BW 140Hz, -40dB

male_a_vowel = [
    {'freq': 650,  'bw': 80,  'amp_db': 0},
    {'freq': 1100, 'bw': 90,  'amp_db': -6},
    {'freq': 2860, 'bw': 120, 'amp_db': -20}, # Resonance 1
    {'freq': 3300, 'bw': 130, 'amp_db': -30}, # Resonance 2
    {'freq': 4500, 'bw': 140, 'amp_db': -40}
]

# Instantiate and Generate
synth = CHANTSynthesizer(sr=44100)
print("Synthesizing CHANT voice...")
audio_data = synth.synthesize_vowel(
    f0=130.81,       # Note C3
    duration=2.0,    # Seconds
    formants=male_a_vowel,
    vib_rate=5.5,    # 5.5 Hz Vibrato
    vib_depth=0.005  # Slight pitch variation
)

# Save to file
wav.write("chant_fof_output.wav", 44100, (audio_data * 32767).astype(np.int16))
print("Done! Saved to chant_fof_output.wav")

# Visualization (Optional) - To see the granular structure
plt.figure(figsize=(10, 4))
plt.plot(audio_data[:2000])
plt.title("First 2000 samples (Showing Overlapping Grains)")
plt.show()