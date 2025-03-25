import numpy as np
import soundfile as sf
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.res_connection = nn.Conv1d(in_channels, out_channels,
                                        kernel_size=1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        residual = self.res_connection(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return self.relu(x + residual)
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()

        return quantized.permute(0, 2, 1).contiguous(), loss, encoding_indices

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.res1 = ResidualBlock(hidden_channels, hidden_channels)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.res2 = ResidualBlock(hidden_channels, hidden_channels)
        self.conv3 = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1)
        self.res3 = ResidualBlock(hidden_channels, hidden_channels)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.res1(x)
        x = F.relu(self.conv2(x))
        x = self.res2(x)
        x = F.relu(self.conv3(x))
        x = self.res3(x)
        return x
class Decoder(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(Decoder, self).__init__()
        self.res1 = ResidualBlock(hidden_channels, hidden_channels)
        self.conv1 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.res2 = ResidualBlock(hidden_channels, hidden_channels)
        self.conv2 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.res3 = ResidualBlock(hidden_channels, hidden_channels)
        self.conv3 = nn.ConvTranspose1d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.res1(x)
        x = F.relu(self.conv1(x))
        x = self.res2(x)
        x = F.relu(self.conv2(x))
        x = self.res3(x)
        x = self.conv3(x)
        return x


class VQVAE2(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, num_embeddings=512, embedding_dim=64):
        super(VQVAE2, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels)
        self.pre_vq_conv = nn.Conv1d(hidden_channels, embedding_dim, kernel_size=1)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, in_channels)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to produce smoother outputs without training"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        z = self.encoder(x)
        z = self.pre_vq_conv(z)
        quantized, vq_loss, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, quantized


class ImprovedAudioSteganography:
    def __init__(self):
        self.sample_rate = 44100
        self.duration = 10
        self.bit_duration = 0.005  # 5ms per bit

        # Using 4 frequency pairs for parallel encoding
        self.freq_pairs = [
            (17000, 17500),  # Pair 1
            (17750, 18250),  # Pair 2
            (18500, 19000),  # Pair 3
            (19250, 19750)  # Pair 4
        ]

        # Initialize VQ-VAE2 model with default parameters
        self.vqvae = VQVAE2()

        # Set device (CPU/GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vqvae.to(self.device)
        self.vqvae.eval()

        # Define a special end marker for messages
        self.end_marker = "§END§"

    def text_to_binary(self, text):
        return ''.join(format(ord(char), '08b') for char in text)

    def binary_to_text(self, binary):
        text = ''
        for i in range(0, len(binary), 8):
            byte = binary[i:i + 8]
            if len(byte) == 8:
                text += chr(int(byte, 2))
        return text

    def generate_carrier(self, frequency, duration):
        """Generate a carrier signal at specified frequency"""
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        return 0.05 * np.sin(2 * np.pi * frequency * t)  # Reduced amplitude

    # [Rest of the melody generation methods remain the same]
    def generate_improved_melody(self):
        """Generate a higher quality instrumental melody without relying on VQ-VAE2"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)
        melody = np.zeros_like(t)

        # Define a chord progression
        chord_progression = [
            ("C", "major"),  # C-E-G
            ("G", "major"),  # G-B-D
            ("A", "minor"),  # A-C-E
            ("F", "major"),  # F-A-C
            ("D", "minor"),  # D-F-A
            ("G", "dom7"),  # G-B-D-F
            ("C", "major"),  # C-E-G
            ("C", "major"),  # Repeat to fill duration
        ]

        # Define frequency for each note
        note_frequencies = {
            "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13, "E": 329.63,
            "F": 349.23, "F#": 369.99, "G": 392.00, "G#": 415.30, "A": 440.00,
            "A#": 466.16, "B": 493.88
        }

        # Define chord structures (semitones from root)
        chord_structures = {
            "major": [0, 4, 7],  # Root, major third, perfect fifth
            "minor": [0, 3, 7],  # Root, minor third, perfect fifth
            "major7": [0, 4, 7, 11],  # Root, major third, perfect fifth, major seventh
            "minor7": [0, 3, 7, 10],  # Root, minor third, perfect fifth, minor seventh
            "dom7": [0, 4, 7, 10]  # Root, major third, perfect fifth, minor seventh
        }

        # Add a bass line
        bass_pattern = [0, 0, 7, 0, 5, 7, 0, 0]  # Relative to chord root

        # Add a melody line
        melody_patterns = [
            [0, 2, 4, 7, 4, 2, 0, -1],
            [7, 4, 2, 0, 2, 4, 7, 9],
            [4, 7, 9, 7, 4, 2, 0, 2],
            [0, -3, -1, 0, 2, 4, 7, 4]
        ]

        chord_duration = self.duration / len(chord_progression)
        beat_duration = chord_duration / 8  # 8 beats per chord

        # Add rhythmic elements
        rhythm_pattern = [1, 0, 0.5, 0, 1, 0.5, 0.5, 0]  # Accent pattern

        for i, (root, chord_type) in enumerate(chord_progression):
            root_freq = note_frequencies[root]

            # Process each beat in the chord
            for beat in range(8):
                start_time = i * chord_duration + beat * beat_duration
                start_idx = int(start_time * self.sample_rate)
                beat_length = int(beat_duration * self.sample_rate)
                end_idx = min(start_idx + beat_length, len(melody))

                # Generate time array for this beat
                beat_t = np.linspace(0, beat_duration, beat_length, False)

                # Create smoother envelope for this beat (MODIFIED: improved envelope shape)
                envelope = np.ones_like(beat_t)
                attack = int(0.15 * len(envelope))  # Slightly longer attack
                release = int(0.25 * len(envelope))  # Slightly longer release
                # Use smoother curve for attack/release (quadratic instead of linear)
                envelope[:attack] = np.power(np.linspace(0, 1, attack), 2)
                envelope[-release:] = np.power(np.linspace(1, 0, release), 2)

                # Apply rhythm pattern
                accent = rhythm_pattern[beat]

                # Generate chord
                beat_signal = np.zeros_like(beat_t)

                # Add chord notes
                for semitone_offset in chord_structures[chord_type]:
                    # Calculate frequency for this note in the chord
                    note_freq = root_freq * (2 ** (semitone_offset / 12))

                    # Add fundamental and harmonics for richer sound (MODIFIED: balanced harmonics)
                    harmonic_mix = 0.2 * np.sin(2 * np.pi * note_freq * beat_t)
                    harmonic_mix += 0.08 * np.sin(2 * np.pi * note_freq * 2 * beat_t)  # 1st harmonic
                    harmonic_mix += 0.03 * np.sin(2 * np.pi * note_freq * 3 * beat_t)  # 2nd harmonic

                    # Apply slight detune for richness (MODIFIED: reduced detuning)
                    detune_amount = 1.002  # 0.2% detune (reduced from 0.3%)
                    harmonic_mix += 0.07 * np.sin(2 * np.pi * note_freq * detune_amount * beat_t)

                    # Add to beat signal
                    beat_signal += harmonic_mix * accent

                # Add bass note
                bass_note = root_freq * 0.5 * (2 ** (bass_pattern[beat] / 12))
                bass_signal = 0.3 * np.sin(2 * np.pi * bass_note * beat_t)
                bass_signal += 0.15 * np.sin(2 * np.pi * bass_note * 2 * beat_t)

                # Add melody note (choose a pattern based on chord index)
                pattern_idx = i % len(melody_patterns)
                melody_note_offset = melody_patterns[pattern_idx][beat]
                if melody_note_offset is not None:
                    melody_note = root_freq * (2 ** (melody_note_offset / 12))
                    melody_signal = 0.25 * np.sin(2 * np.pi * melody_note * beat_t)

                    # Add vibrato to melody (MODIFIED: smoother vibrato)
                    vibrato_rate = 5  # Hz (slowed down)
                    vibrato_depth = 0.004  # Reduced depth
                    vibrato = vibrato_depth * np.sin(2 * np.pi * vibrato_rate * beat_t)
                    melody_signal += 0.18 * np.sin(2 * np.pi * melody_note * (1 + vibrato) * beat_t)

                    beat_signal += melody_signal

                # Add bass to signal
                beat_signal += bass_signal

                # Apply envelope to the beat
                beat_signal *= envelope

                # Add to main melody
                if end_idx > start_idx:
                    actual_length = min(len(beat_signal), end_idx - start_idx)
                    melody[start_idx:start_idx + actual_length] += beat_signal[:actual_length]

        # MODIFIED: Improved reverb parameters
        # Add more subtle reverb effect with longer decay for natural feel
        reverb_length = int(0.35 * self.sample_rate)  # 350ms reverb
        # Exponential decay with smoother curve
        reverb_envelope = np.exp(-np.linspace(0, 6, reverb_length))

        # Normalize reverb envelope
        reverb_envelope = reverb_envelope / np.sum(reverb_envelope)

        # Apply reverb
        melody_with_reverb = np.convolve(melody, reverb_envelope, mode='full')[:len(melody)]

        # MODIFIED: Removed random noise addition to reduce noise
        # Instead add very subtle, filtered noise for natural analog warmth
        subtle_noise = np.random.normal(0, 0.0008, len(melody_with_reverb))  # Reduced amplitude

        # Apply bandpass filter to noise to make it less harsh
        sos_noise = signal.butter(4, [100, 3000], 'bandpass', fs=self.sample_rate, output='sos')
        filtered_noise = signal.sosfilt(sos_noise, subtle_noise)

        # Add filtered noise at very low level
        melody_with_reverb += filtered_noise * 0.1

        # Apply gentle low-pass filter to melody to reduce harshness
        sos_smooth = signal.butter(3, 12000, 'lowpass', fs=self.sample_rate, output='sos')
        melody_with_reverb = signal.sosfilt(sos_smooth, melody_with_reverb)

        # MODIFIED: Apply subtle compression for consistent level
        # Simple compressor parameters
        threshold = 0.5
        ratio = 1.5  # Gentle ratio
        attack = 0.01  # 10ms
        release = 0.1  # 100ms

        # Time constants
        attack_coef = np.exp(-1.0 / (self.sample_rate * attack))
        release_coef = np.exp(-1.0 / (self.sample_rate * release))

        # Envelope follower
        env = 0
        compressed = np.zeros_like(melody_with_reverb)

        for i, sample in enumerate(melody_with_reverb):
            # Envelope detection
            env_in = abs(sample)
            if env_in > env:
                env = attack_coef * env + (1 - attack_coef) * env_in
            else:
                env = release_coef * env + (1 - release_coef) * env_in

            # Gain computation
            if env <= threshold:
                gain = 1.0
            else:
                gain = 1.0 + (1.0 / ratio - 1.0) * (env - threshold) / env

            # Apply gain
            compressed[i] = sample * gain

        melody_with_reverb = compressed

        # MODIFIED: Added gentle multiband EQ for cleaner sound
        # Apply high-shelf reduction above 10kHz to reduce harshness
        sos_highshelf = signal.butter(2, 10000, 'lowpass', fs=self.sample_rate, output='sos')
        melody_high = signal.sosfilt(sos_highshelf, melody_with_reverb)

        # Boost mids slightly for warmth (between 300Hz and 3kHz)
        sos_mid = signal.butter(2, [300, 3000], 'bandpass', fs=self.sample_rate, output='sos')
        melody_mid = signal.sosfilt(sos_mid, melody_with_reverb)

        # Combine with slight mid boost
        melody_with_reverb = 0.85 * melody_with_reverb + 0.15 * melody_mid

        # Normalize the final melody
        melody_with_reverb = melody_with_reverb / np.max(np.abs(melody_with_reverb)) * 0.8

        return melody_with_reverb

    def embed_message(self, message, output_file="output.wav"):
        """Generate audio with embedded message, adding an end marker"""
        # Add the end marker to the message
        message_with_marker = message + self.end_marker

        binary_message = self.text_to_binary(message_with_marker)
        print(f"Binary message length: {len(binary_message)} bits")

        # Calculate maximum message length (4x capacity due to parallel encoding)
        bits_per_channel = int(self.duration / self.bit_duration)
        max_bits = bits_per_channel * len(self.freq_pairs)
        max_chars = max_bits // 8
        print(f"Maximum capacity: {max_chars} characters")

        if len(binary_message) > max_bits:
            raise ValueError(f"Message too long. Maximum length is {max_chars - len(self.end_marker)} characters")

        # Generate improved instrumental melody
        print("Generating improved melody...")
        melody = self.generate_improved_melody()

        # Apply simple psychoacoustic masking to determine optimal data signal amplitude
        # This helps hide the data signal in the melody based on human hearing perception
        def calculate_masking_threshold(audio_segment, freq):
            # Simple psychoacoustic model - loud sounds at similar frequencies mask quieter ones
            fft = np.abs(np.fft.rfft(audio_segment))
            freqs = np.fft.rfftfreq(len(audio_segment), 1 / self.sample_rate)

            # Find the nearest frequency bin
            idx = np.argmin(np.abs(freqs - freq))

            # Calculate a masking threshold based on energy in nearby bins
            # (simplified model - real psychoacoustic models are more complex)
            window_size = 10  # Look at 10 bins on each side
            start_idx = max(0, idx - window_size)
            end_idx = min(len(fft), idx + window_size)

            nearby_energy = np.mean(fft[start_idx:end_idx])

            # Return a scaling factor based on nearby energy
            # Higher energy means we can hide more data (use higher amplitude)
            return min(0.3, max(0.05, nearby_energy / np.max(fft) * 0.2))

        # Generate data track with message
        data_signal = np.zeros(len(melody))
        samples_per_bit = int(self.bit_duration * self.sample_rate)

        # Pad binary message to be divisible by 4
        padding_length = (-len(binary_message)) % 4
        binary_message += '0' * padding_length

        # Process 4 bits at a time using parallel frequency bands
        for i in range(0, len(binary_message), 4):
            bit_group = binary_message[i:i + 4]
            start = (i // 4) * samples_per_bit
            end = start + samples_per_bit

            # Skip if we're past the end of the audio
            if start >= len(melody):
                break

            # Generate carriers for each bit using different frequency pairs
            for j, bit in enumerate(bit_group):
                freq = self.freq_pairs[j][1] if bit == '1' else self.freq_pairs[j][0]

                # Calculate optimal amplitude based on psychoacoustic masking
                segment = melody[start:end] if end <= len(melody) else melody[start:]
                mask_level = calculate_masking_threshold(segment, freq)

                # Generate carrier with adaptive amplitude
                carrier = mask_level * np.sin(
                    2 * np.pi * freq * np.linspace(0, self.bit_duration, samples_per_bit, False))

                # Apply a short fade in/out to reduce clicking
                fade_samples = min(int(0.0005 * self.sample_rate), len(carrier) // 10)  # 0.5ms or 10% of bit length
                carrier[:fade_samples] *= np.linspace(0, 1, fade_samples)
                carrier[-fade_samples:] *= np.linspace(1, 0, fade_samples)

                # Add carrier to data signal
                if end <= len(data_signal):
                    data_signal[start:end] += carrier
                else:
                    data_signal[start:] += carrier[:len(data_signal) - start]

        # MODIFIED: Improved filtering for melody
        # Use a more precise band-pass filter on melody to reduce interference with high-frequency data signal
        # Higher order filter with gentler slopes to preserve musical quality
        sos = signal.butter(6, [40, 15000], 'bandpass', fs=self.sample_rate, output='sos')
        filtered_melody = signal.sosfilt(sos, melody)

        # Normalize filtered melody
        filtered_melody = filtered_melody / np.max(np.abs(filtered_melody)) * 0.82  # Slightly higher level

        # MODIFIED: Improved anti-aliasing filter for data signal
        # Add an anti-aliasing filter to data signal with more precise cutoff
        sos_aa = signal.butter(6, [16500, 20000], 'bandpass', fs=self.sample_rate, output='sos')
        filtered_data = signal.sosfilt(sos_aa, data_signal)

        # Combine signals
        audio = filtered_melody + filtered_data

        # MODIFIED: Improved normalization and limiting
        # Apply a smoother multiband limiter to prevent clipping
        threshold = 0.92
        ratio = 3.0  # compression ratio

        # Split audio into frequency bands for multiband processing
        sos_low = signal.butter(3, 500, 'lowpass', fs=self.sample_rate, output='sos')
        sos_mid = signal.butter(3, [500, 8000], 'bandpass', fs=self.sample_rate, output='sos')
        sos_high = signal.butter(3, 8000, 'highpass', fs=self.sample_rate, output='sos')

        audio_low = signal.sosfilt(sos_low, audio)
        audio_mid = signal.sosfilt(sos_mid, audio)
        audio_high = signal.sosfilt(sos_high, audio)

        # Apply limiting to each band
        for band in [audio_low, audio_mid, audio_high]:
            mask = np.abs(band) > threshold
            if np.any(mask):
                # Smoother transition at threshold
                band[mask] = np.sign(band[mask]) * (threshold + (np.abs(band[mask]) - threshold) / ratio)

        # Recombine bands
        audio = audio_low + audio_mid + audio_high

        # Final normalization with headroom
        audio = audio / np.max(np.abs(audio)) * 0.95

        # MODIFIED: Add subtle dithering to mask quantization noise
        # This can actually improve perceptual quality despite adding very low-level noise
        dither_amplitude = 1.0 / (2 ** 15)  # 16-bit dithering
        dither = np.random.uniform(-dither_amplitude, dither_amplitude, len(audio))
        audio += dither

        # Save the audio - MODIFIED: using 24-bit depth for better quality
        sf.write(output_file, audio, self.sample_rate, subtype='PCM_24')
        return audio

    def extract_message(self, audio_file, max_bits=None):
        """Extract hidden message without knowing the message length by detecting an end marker"""
        audio, _ = sf.read(audio_file)

        # Apply high-pass filter to isolate the high-frequency data signals
        sos = signal.butter(8, 16000, 'highpass', fs=self.sample_rate, output='sos')
        filtered_audio = signal.sosfilt(sos, audio)

        samples_per_bit = int(self.bit_duration * self.sample_rate)
        binary_message = ''

        # If max_bits is not provided, use the maximum possible bits based on audio length
        if max_bits is None:
            # Estimate maximum number of bits based on audio length
            # Each bit group (4 bits) takes one bit_duration of time
            max_bit_groups = len(filtered_audio) // samples_per_bit
            max_bits = max_bit_groups * 4  # 4 bits per group

        # Process bit groups until we reach max_bits or the end of audio
        for i in range((max_bits + 3) // 4):  # Round up to nearest group of 4
            start = i * samples_per_bit
            end = start + samples_per_bit

            if end > len(filtered_audio):
                break

            segment = filtered_audio[start:end]

            # Apply window function to reduce spectral leakage
            segment = segment * np.hanning(len(segment))

            # Perform FFT with zero-padding for better frequency resolution
            n_fft = 2048  # Increased FFT size for better resolution
            freqs = np.fft.rfftfreq(n_fft, 1 / self.sample_rate)
            fft = np.abs(np.fft.rfft(segment, n=n_fft))

            # Decode 4 bits in parallel
            for j, (freq_0, freq_1) in enumerate(self.freq_pairs):
                # Find magnitude at each frequency
                idx_0 = np.argmin(np.abs(freqs - freq_0))
                idx_1 = np.argmin(np.abs(freqs - freq_1))

                # Get energy around each frequency (use a small window)
                window = 2
                energy_0 = np.sum(fft[max(0, idx_0 - window):idx_0 + window + 1])
                energy_1 = np.sum(fft[max(0, idx_1 - window):idx_1 + window + 1])

                # Compare energies to determine bit
                binary_message += '1' if energy_1 > energy_0 else '0'

                # Check if we've reached max_bits
                if len(binary_message) >= max_bits:
                    break

            # Check if we've reached max_bits
            if len(binary_message) >= max_bits:
                break

        # Convert binary to text
        full_text = self.binary_to_text(binary_message)

        # Look for the end marker
        if self.end_marker in full_text:
            # Extract message up to the end marker
            extracted_message = full_text.split(self.end_marker)[0]
            return extracted_message
        else:
            # If no end marker is found, return everything (with a warning)
            print("Warning: End marker not found. Returning all extracted data.")
            return full_text


def test_improved_steganography(message):
    """Test function to verify encoding/decoding with improved melody generation"""
    stego = ImprovedAudioSteganography()

    try:
        print("Generating audio with hidden message using improved melody generation...")
        audio = stego.embed_message(message, "hidden_message_improved.wav")

        print("\nExtracting hidden message without knowing its length...")
        extracted_message = stego.extract_message("hidden_message_improved.wav")

        print(f"\nOriginal message: {message}")
        print(f"Extracted message: {extracted_message}")
        print(f"\nMessage length: {len(message)} characters")
        print(f"Success: {message == extracted_message}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    # Test with a message
    test_message = "hello everyone!."
    test_improved_steganography(test_message)