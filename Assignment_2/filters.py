import numpy as np

def ram_lak_filter(fft, num_t_values, l_mul_factor=1):
    freqs = np.fft.fftfreq(num_t_values).reshape(-1,1)
    filter = np.abs(freqs)

    l_cutoff = np.max(np.abs(freqs)) * l_mul_factor
    
    rectangle = np.where(np.abs(freqs) <= l_cutoff, 1, 0)   
    filter *= rectangle

    filtered_fft =  fft * filter
    return filtered_fft


def shepp_logan_filter(fft, num_t_values, l_mul_factor=1):
    freqs = np.fft.fftfreq(num_t_values).reshape(-1,1)
    l_cutoff = np.max(np.abs(freqs)) * l_mul_factor
    C = np.sinc(0.5 * freqs / l_cutoff)
    rectangle = np.where(np.abs(freqs) <= l_cutoff, 1, 0)
    filter = np.abs(freqs) * rectangle * C

    filtered_fft = fft * filter
    return filtered_fft

def cosine_filter(fft, num_t_values, l_mul_factor=1):
    freqs = np.fft.fftfreq(num_t_values).reshape(-1,1)
    l_cutoff = np.max(np.abs(freqs)) * l_mul_factor
    C = np.cos((0.5 * np.pi * freqs) / l_cutoff)
    rectangle = np.where(np.abs(freqs) <= l_cutoff, 1, 0)
    filter = np.abs(freqs) * rectangle * C

    filtered_fft = fft * filter
    return filtered_fft

def unfiltered(fft, num_t_values, l_mul_factor=1):
    return fft

