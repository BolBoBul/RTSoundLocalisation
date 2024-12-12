# %% [markdown]
# # Signal Processing Project: real-time sound localisation

# %% [markdown]
# ## 1 Offline system
# ### 1.1 Data generation and dataset

# %%
import numpy as np
import matplotlib.pyplot as plt

def create_sine_wave(f, A, fs, N):
    """Function to create a sine wave signal

    Args:
        f (int): Frequency of the sine wave
        A (int): Amplitude peak-to-peak of the sine wave
        fs (int): Sampling frequency
        N (int): Number of samples

    Returns:
        ndarray: Samples of the sine wave signal
    """
    
    t = np.linspace(0, N/fs, N)
    out = A/2 * np.sin(2 * np.pi * f * t)

    return out

# call and test your function here #
fs = 44100
N = 8000
freq = 20
amplitude = 8

your_signal = create_sine_wave(freq, amplitude, fs, N)
plt.plot(your_signal)


# %%
from glob import glob
import scipy.io.wavfile as wf

def read_wavefile(path):
    """Function to read a wave file
    
    Args:
        path (str): Path to the wave file
        
    Returns:
        tuple: Tuple containing the sampling rate and the samples of the wave file
    """

    out = wf.read(path)

    return out

# call and test your function here #
LocateClaps = "resources/LocateClaps/"
files = glob(f"{LocateClaps}/*.wav")
# select the ith file
i = 10
# the second part of the array represents the value and the first element is the sampling rate
your_wave = read_wavefile(files[i])[1]
plt.title(f"One-clap sound #{i}")
plt.plot(your_wave)

# %% [markdown]
# ### 1.2 Buffering

# %%
from collections import deque

# When the buffer is full, the oldest element is removed
class RingBuffer:
    """Ring buffer class
    """
    def __init__(self, max_length):
        self.max_length = max_length
        self.data = deque(maxlen=max_length)

    def append(self, x):
        self.data.append(x)

    def get(self):
        return self.data
    
def create_ringbuffer(maxlen):
    """Function to create a ring RingBuffer

    Args:
        maxlen (int): Maximum length of the buffer

    Returns:
        RingBuffer: Ring buffer instance which is a deque
    """
    
    out = RingBuffer(maxlen)

    return out

# call and test your function here #
maxlen = 750

your_buffer = create_ringbuffer(maxlen)

'''# reading your signal as a stream:
for i, sample in enumerate(your_signal):
    your_buffer.append(sample)
    
'''
def display_buffer_after_X_seconds(buffer: RingBuffer, time, signal):
    """Display the content of the buffer after a certain amount of time

    Args:
        ring_buffer (RingBuffer): Ring buffer instance
        time (float): Time in seconds
        signal (list): List of samples of the signal
    """
    
    # we create a variable to store the index of the buffer
    idx = 0
    # we iterate over the samples of the signal
    for i, sample in enumerate(signal):
        # we append the sample to the buffer
        buffer.append(sample)
        # we check if the time has passed
        if i/fs >= time:
            idx = i
            break
    # we plot the content of the buffer
    plt.plot(buffer.get())
    plt.title(f"Buffer content after {time} seconds")
    plt.show()

display_buffer_after_X_seconds(RingBuffer(maxlen), 1.5, your_wave)

# %% [markdown]
# ### 1.3 Pre-processing
# #### 1.3.1 Normalisation

# %%
def normalise(s):
    """Normalise the signal between -1 and 1

    Args:
        s (ndarray): Signal to normalise

    Returns:
        ndarray: Normalised signal
    """
    
    # we divide the signal by its maximum value to get a signal between -1 and 1
    
    out = s / np.max(np.abs(s))

    return out

# call and test your function here #
plt.plot(normalise(your_wave), 'b')
plt.show()

# %% [markdown]
# #### 1.3.2 Downsampling

# %%
## 1 - spectral analysis via spectrogram
# The y-axis represents the frequency, the x-axis the time and the color the amplitude (in dB)
your_signal1 = create_sine_wave(5500, 10, fs, N)
your_signal2 = create_sine_wave(7500, 1000, fs, N)
your_other_signal = your_signal1 + your_signal2

figure, axis = plt.subplots(1, 2)
axis[0].specgram(x=your_other_signal, Fs=fs, NFFT=1024, noverlap=512, cmap="inferno")
axis[0].set_title("Spectrogram of the generated sine wave")

axis[1].specgram(x=your_wave, Fs=fs, NFFT=1024, noverlap=512, cmap="inferno")
axis[1].set_title("Spectrogram of the one-clap sound")
plt.colorbar(axis[1].get_children()[0], ax=axis[1])
plt.show()


# Analyse du signal:
# Sur le spectrogramme du clap, on voit qu'au-delà d'environ 8000Hz, l'intensité des fréquences est très faible, on peut donc se dire qu'une fréquence d'échantillonnage de 16000Hz est suffisante pour capturer les informations importantes de ce signal.

## 2 - Anti-aliasing filter synthesis
# %%
import scipy.signal as sc

def create_filter_cheby(wp, ws, gpass, gstop, fs):
    """
    Design a Chebyshev Type I low-pass filter.
    
    Parameters:
        wp: float  - Passband edge frequency (Hz).
        ws: float  - Stopband edge frequency (Hz).
        gpass: float - Maximum ripple in the passband (dB).
        gstop: float - Minimum attenuation in the stopband (dB).
        fs: float   - Sampling frequency (Hz).
        
    Returns:
        B, A: ndarray - Numerator and denominator coefficients of the filter.
    """


    
    N, Wn = sc.cheb2ord(wp, ws, gpass, gstop, fs=fs)
    B, A = sc.cheby2(N, gstop, Wn, btype='low', fs=fs)
    
    return B, A

def create_filter_cauer(wp, ws, gpass, gstop, fs):
    """
    Design an Elliptic (Cauer) low-pass filter.
    
    Parameters:
        wp: float  - Passband edge frequency (Hz).
        ws: float  - Stopband edge frequency (Hz).
        gpass: float - Maximum ripple in the passband (dB).
        gstop: float - Minimum attenuation in the stopband (dB).
        fs: float   - Sampling frequency (Hz).
        
    Returns:
        B, A: ndarray - Numerator and denominator coefficients of the filter.
    """
    
    N, Wn = sc.ellipord(wp, ws, gpass, gstop, fs=fs)
    B, A = sc.ellip(N, gpass, gstop, Wn, btype='low', fs=fs)
    
    return B, A


N=8000
sine1 = create_sine_wave(8500, 1000, 44100, N)
sine2 = create_sine_wave(7500, 20, 44100, N)
main_signal = sine1 + sine2


# call and test your function here #
fs = 24000
wp = 8000
ws = 8500
gpass = 1
gstop = 40

# create a Chebyshev Type I low-pass filter
B_cheby, A_cheby = create_filter_cheby(8000, 8500, 1, 40, fs=fs)
# create an Elliptic (Cauer) low-pass filter
B_cauer, A_cauer = create_filter_cauer(8000, 8500, 1, 40, fs=fs)

# plot the frequency response of the filters
w_cheby, h_cheby = sc.freqz(B_cheby, A_cheby, worN=2048, fs=fs)
w_cauer, h_cauer = sc.freqz(B_cauer, A_cauer, worN=2048, fs=fs)

h_cheby = 20*np.log10(abs(h_cheby))
h_cauer = 20*np.log10(abs(h_cauer))

decimated_signal = main_signal[::3]
# we plot the sine signals filtered by the Chebyshev filter
plt.plot(sc.lfilter(B_cheby, A_cheby, decimated_signal), 'b')
plt.show()


# %%
## 3 - Decimation
def downsampling(sig,B, A, M):
    """Decimate a signal by keeping one sample every M samples

    Args:
        sig (ndaray): Signal to decimate
        M (int): factor of decimation

    Returns:
        ndarray: Decimated signal
    """
    filtered_sig = sc.lfilter(B, A, sig)
    out = filtered_sig[::M]
    return out

fs=16000
N=8000
sinus1 = create_sine_wave(8500, 1000, fs, N)
sinus2 = create_sine_wave(7500, 20, fs, N)

# wave1 is from the microphone at angle 0 of device 1
your_wave1 = wf.read(files[1])[1]
# wave11 is from the microphone at angle 30 of device 1
your_wave11 = wf.read(files[2])[1]
# wave2 is from the microphone at angle 0 of device 2
your_wave2 = wf.read(files[12])[1]

# call and test your function here #
M = 10
signal = sinus1 + sinus2
downsampled_signal = simple_downsampling(signal, M)

plt.plot(signal, 'orange')
# we stretch the signal to see the difference
plt.plot(np.linspace(0, len(signal), len(downsampled_signal)), downsampled_signal, 'b')
plt.legend(["Original signal", "Downsampled signal"])


# %% [markdown]
# ### 1.4 Cross-correlation

# %%
## 1.4
import scipy.signal as sc
import numpy as np

def fftxcorr(in1, in2):
    """Cross-correlation of two signals using the Fourier Transform
    
    Args:
        in1 (ndarray): First input signal
        in2 (ndarray): Second input signal
        
    Returns:
        ndarray: Cross-correlation of the two signals
    """
    input1 = np.asarray(in1)
    input2 = np.asarray(in2)
    input1 = np.append(input1, np.zeros(len(in2)))
    input2 = np.append(np.zeros(len(in1)), input2)
    out = np.fft.ifft(np.fft.fft(input1) * np.conj(np.fft.fft(input2))).real
    
    # we use np.fft.fft to compute the Fourier Transform of the input signals
    # we apply the inverse Fourier Transform (np.fft.ifft) to the product of the Fourier Transform of in1 and the complex conjugate (np.conj) of the Fourier Transform of in2
    
    return out
    
# call and test your function here #
# Verify your implementation, compare your output with that of the function fftconvolve when computing the auto-correlation of your sine wave signal. Remember that you need to counter the flip that convolution does on the second signal
normalised_wave1 = normalise(your_wave1)
normalised_wave11 = normalise(your_wave11)
normalised_wave2 = normalise(your_wave2)

# if the shape of the 2 signals is not the same, we pad the smallest signal with zeros
def pad_signal(signal1, signal2):
    if len(signal1) > len(signal2):
        signal2 = np.pad(signal2, (0, len(signal1)-len(signal2)), 'constant')
    else:
        signal1 = np.pad(signal1, (0, len(signal2)-len(signal1)), 'constant')
    return signal1, signal2

normalised_wave1, normalised_wave2 = pad_signal(normalised_wave1, normalised_wave2)

your_signal3 = create_sine_wave(21, 4, 44100, 8000)
your_signal4 = create_sine_wave(21, 4, 44100, 8000)

plt.plot(fftxcorr(your_signal3, your_signal4), 'b')
plt.plot(sc.fftconvolve(your_signal3, your_signal4[::-1]), 'orange')
plt.show()

# we see that the blue graph is the same as the orange one but shifted by half of the length of the signal. Also, the fact that the blue graph has its highest value at the middle of the signal is normal since the signal is the same as the input signal

# %% [markdown]
# ### 1.5 Localisation
# #### 1.5.1 TDOA

# %%
def TDOA(xcorr):
    
    # One way to measure the time-shift is to find the index of the maximum value of the  ross-correlation
    # we know that the delay between the two signals reception is the difference between the index of the maximum value and the middle of the cross-correlation
    out = np.where(xcorr == np.max(xcorr))[0][0] - len(xcorr)//2
    
    return out

# call and test your function here #
print(f"{TDOA(fftxcorr(your_signal1, your_signal1))}") 

# %%
beforeProcess = {}
afterProcess = {}

def process_sample(sample):
    # we find the first sample where the amplitude is higher than 3000
    start = np.argmax(sample > 3000)
    # we find the last sample where the amplitude is higher than 3000
    end = len(sample) - np.argmax(sample[::-1] > 1000)
    # we keep only the samples between start and end
    out = sample[start-100:end]
    return out

for f in files:
    beforeProcess[f] = wf.read(f)[1]
    afterProcess[f] = process_sample(beforeProcess[f])
    
# find the longest signal among the processed signals
maxlen = 0
for f in afterProcess:
    if len(afterProcess[f]) > maxlen:
        maxlen = len(afterProcess[f])        

# we pad the signals with zeros to have the same length
for f in afterProcess:
    afterProcess[f] = np.pad(afterProcess[f], (0, maxlen-len(afterProcess[f])), 'constant')
        
# call and test your function here #
your_wave1 = afterProcess[files[4]]
your_wave2 = afterProcess[files[23]]





# %% [markdown]
# #### 1.5.2 Equation system

# %%
from scipy.optimize import root

# mic coordinates in meters
MICS = [{'x': 0, 'y': 0.0487}, {'x': 0.0425, 'y': -0.025}, {'x': -0.0425, 'y': -0.025}] 

def equations(p, deltas):
    """System of equations to solve the problem of localisation

    Args:
        p (tuple): coordinates of the source of the sound
        deltas (list): list of the TDOA between the microphones

    Returns:
        tuple: system of equations
    """
    # speed of sound in m/s
    v = 343
    # p is the coordinates of the source of the sound
    x, y = p
    
    # we calculate the angles between the microphones #0 & #1 and between the microphones #0 & #2
    alpha = np.arctan2((MICS[1]['y'] - MICS[0]['y']), (MICS[1]['x'] - MICS[0]['x']))
    beta = np.arctan2((MICS[2]['y'] - MICS[0]['y']), (MICS[2]['x'] - MICS[0]['x']))
    
    # we create the system of equations to solve the problem of localisation
    eq1 = v*deltas[0] - (np.sqrt((MICS[1]['x'] - MICS[0]['x'])**2 + (MICS[1]['y'] - MICS[0]['y'])**2) * np.sqrt((x)**2 + (y)**2) * np.cos(alpha-np.arctan2(y, x)))
    eq2 = v*deltas[1] - (np.sqrt((MICS[2]['x'] - MICS[0]['x'])**2 + (MICS[2]['y'] - MICS[0]['y'])**2) * np.sqrt((x)**2 + (y)**2) * np.cos(beta-np.arctan2(y, x)))
    return (eq1, eq2)
    
def localize_sound(deltas):
    """Localize the source of the sound given the time differences of arrival (TDOA) between the microphones

    Args:
        deltas (list): list of the TDOA between the microphones

    Returns:
        tuple: x and y coordinates of the source
    """

    sol = root(equations, [0, 0], (deltas), tol=10)
    return sol.x

def source_angle(coordinates):
    """Output the angle of the source from the x-axis in degrees

    Args:
        coordinates (tuples): x and y coordinates of the source

    Returns:
        float: angle in degrees
    """
    import math
    
    unit_vector_src = np.array(coordinates)/np.linalg.norm(coordinates)
    unit_vector_x_axis = [1,0] / np.linalg.norm([1,0])
    dot_product = np.dot(unit_vector_src, unit_vector_x_axis)
    out = math.degrees(np.arccos(dot_product))
    
    if coordinates[1] < 0:
        out = 360 - out

    return out

# call and test your function here #
your_wave1, your_wave11 = pad_signal(your_wave1, your_wave11)

deltas = list(map(TDOA, [fftxcorr(your_wave1, your_wave11), fftxcorr(your_wave1, your_wave1), fftxcorr(your_wave11, your_wave11)]))
coordinates = localize_sound(deltas)
print(f"Coordinates: {coordinates}")
print(f"Angle: {round(source_angle(coordinates), 2)}°")


# %% [markdown]
# ### 1.6 System accuracy and speed

# %%
## 1.6.1
def accuracy(pred_angle, gt_angle, threshold):
    
    # your code here #
    
    out = min(abs(pred_angle - gt_angle), 360 - abs(pred_angle - gt_angle)) < threshold

    return out

## 1.6.2
possible_angle = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
            
locate_claps = "resources/LocateClaps/"
files = glob(f"{locate_claps}/*.wav")
B, A = create_filter_cauer(8000, 8500, 1, 40, fs=44100)
threshold = 11
for angle in possible_angle:
    m1_wavfile = []
    m2_wavfile = []
    m3_wavfile = []
    for f in files:
        if f'_{angle}.' in f:
            mic = f.split('/')[-1].split('_')[0]
            if mic == 'M1':
                m1_wavfile = read_wavefile(f)[1]
            elif mic == 'M2':
                m2_wavfile = read_wavefile(f)[1]
            elif mic == 'M3':
                m3_wavfile = read_wavefile(f)[1]
            else:
                print("Error")
                break
                
    # Preprocess
    m1_wavfile = normalise(m1_wavfile)
    m2_wavfile = normalise(m2_wavfile)
    m3_wavfile = normalise(m3_wavfile)
    
    #Downsampling
    wp=16000/2
    ws = wp + 1000
    gpass = 1
    gstop = 40
    M = 3
    m1_wavfile = downsampling(m1_wavfile, B, A, M)
    m2_wavfile = downsampling(m2_wavfile, B, A, M)
    m3_wavfile = downsampling(m3_wavfile, B, A, M)
    #X-correlation
    m12_xcorr = fftxcorr(m1_wavfile, m2_wavfile)
    m13_xcorr = fftxcorr(m1_wavfile, m3_wavfile)
    # localisation
    m12_time_shift_value = TDOA(m12_xcorr)/16000
    m13_time_shift_value = TDOA(m13_xcorr)/16000
    
    # Equations systems
    coords_system = localize_sound([m12_time_shift_value, m13_time_shift_value])
    pred_angle = source_angle(coords_system)
    print(f'Angle: {angle}°, Prediction: {pred_angle}°, Accuracy: {accuracy(pred_angle, angle, threshold)}')



## 1.6.3
from time import time_ns, sleep

def func_example(a, b):
    return a*b

def time_delay(func, args):
    start_time = time_ns()
    out = func(*args)
    end_time = time_ns()
    print(f"{func.__name__} in {end_time - start_time} ns")
    return out

product = time_delay(func_example, [2, 10])

# call and test your previous functions here #
signal_test = read_wavefile(files[0])[1]
signal_test_norm = time_delay(normalise, [signal_test])
signal_test_down = time_delay(downsampling, [signal_test_norm, B, A, 3])
signal_test_xcorr = time_delay(fftxcorr, [signal_test_down, signal_test_down])
signal_test_tdoa = time_delay(TDOA, [signal_test_xcorr])
signal_test_localisation = time_delay(localize_sound, [[signal_test_tdoa, signal_test_tdoa]])
signal_test_angle = time_delay(source_angle, [signal_test_localisation])

# %% [markdown]
# ## 2 Real-time localisation

# %% [markdown]
# ### 2.1 Research one Raspberry Pi application

# %% [markdown]
# ### 2.2 Data acquisition and processing

# %%
#### Callback 
import pyaudio

RESPEAKER_CHANNELS = 8
BUFFERS = []

def callback(in_data, frame_count, time_info, flag):
    global BUFFERS
    data = np.frombuffer(in_data, dtype=np.int16)
    BUFFERS[0].extend(data[0::RESPEAKER_CHANNELS])
    BUFFERS[1].extend(data[2::RESPEAKER_CHANNELS])
    BUFFERS[2].extend(data[4::RESPEAKER_CHANNELS])
    return (None, pyaudio.paContinue)

#### Stream management

RATE = 44100
RESPEAKER_WIDTH = 2
CHUNK_SIZE = 2048

def init_stream():
    print("========= Stream opened =========")
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)

        if device_info['maxInputChannels'] == 8:
            INDEX = i
            break

        if i == p.get_device_count()-1:
            # Sound card not found
            raise OSError('Invalid number of channels')

    stream = p.open(rate=RATE, channels=RESPEAKER_CHANNELS, format=p.get_format_from_width(RESPEAKER_WIDTH), input=True, input_device_index=INDEX,
                    frames_per_buffer=CHUNK_SIZE, stream_callback=callback)

    return stream



def close_stream(stream):
    print("========= Stream closed =========")
    stream.stop_stream()
    stream.close()

#### Detection and visual feedback
def detection(stream):
    global BUFFERS, pixel_ring
    
    if stream.is_active():
        print("========= Recording =========")

    while stream.is_active():
        try:
            if len(BUFFERS[0]) > CHUNK_SIZE:
                st = time_ns()
                deltas = [TDOA(fftxcorr(BUFFERS[0], BUFFERS[1])), TDOA(fftxcorr(BUFFERS[0], BUFFERS[2]))] 

                x, y = localize_sound(deltas)
                hyp = np.sqrt(x**2+y**2)
                
                ang_cos = round(np.arccos(x/hyp)*180/np.pi, 2)
                ang_sin = round(np.arcsin(y/hyp)*180/np.pi, 2)

                if ang_cos == ang_sin:
                    ang = ang_cos
                else:
                    ang = np.max([ang_cos, ang_sin])
                    if ang_cos < 0 or ang_sin < 0:
                        ang *= -1
                ang *= -1

                print((time_ns() - st)/1e9, ang)

                print(np.max(BUFFERS, axis=-1))

                if (np.max(BUFFERS, axis=-1) > 3000).any():
                    pixel_ring.wakeup(ang)
                else:
                    pixel_ring.off()

                sleep(0.5)

        except KeyboardInterrupt:
            print("========= Recording stopped =========")
            break

#### Launch detection
from pixel_ring.apa102_pixel_ring import PixelRing
from gpiozero import LED


USED_CHANNELS = 3


power = LED(5)
power.on()

pixel_ring = PixelRing(pattern='soundloc')

pixel_ring.set_brightness(10)

for i in range(USED_CHANNELS):
    BUFFERS.append(create_ringbuffer(3 * CHUNK_SIZE))
    
stream = init_stream()

while True:
    try:
        detection(stream)
        sleep(0.5)
    except KeyboardInterrupt:
        break

close_stream(stream)

power.off()


