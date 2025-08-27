#!/usr/bin/env python
import numpy as np
from pycbc.filter import highpass_fir, lowpass_fir
from pycbc.waveform import get_td_waveform
from pycbc.psd import aLIGOZeroDetHighPower, interpolate, welch
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q
from pycbc.noise import noise_from_psd


fs = 4096
duration = 1024
length = int(duration * fs)
mcmin, mcmax = 30.0, 60.0
qmin, qmax = 1.0, 4.0
ninjection = 4
tclist = np.random.randint(1, 31, (ninjection,)) * 32.0

delta_f = 1.0 / 4
flength = int(fs / delta_f)
low_frequency_cutoff = 20.0
psd = aLIGOZeroDetHighPower(flength, delta_f, low_frequency_cutoff)
# k_zeropsd = psd.data == 0.0
# psd.data[k_zeropsd] = 1e+10
noise = noise_from_psd(length, 1.0 / fs, psd)

outputfile = 'data/test_strain.npy'
mclist = np.random.uniform(mcmin, mcmax, (ninjection,))
qlist = np.random.uniform(qmin, qmax, (ninjection,))
m1list = mass1_from_mchirp_q(mclist, qlist)
m2list = mass2_from_mchirp_q(mclist, qlist)

all_injection_strains = 0

for i in range(ninjection):
    tc = tclist[i]
    hp, hc = get_td_waveform(
        approximant='IMRPhenomD',
        mass1=m1list[i],
        mass2=m2list[i],
        delta_t=1.0 / fs,
        f_lower=15.0,
        distance=1000
    )
    tstart = hp.start_time
    hp.append_zeros(length - len(hp))
    hp.roll(int(tstart * fs))
    hp.start_time = 0.0
    hp.roll(int(tc * fs))

    all_injection_strains += hp

simulated_signal = all_injection_strains + noise
simulated_signal = highpass_fir(simulated_signal, 15, 8)
psd_estimated = interpolate(welch(simulated_signal, seg_len=4 * fs, seg_stride=2 * fs, avg_method='median-mean'), 1.0 / simulated_signal.duration)
mask = np.ones((len(psd_estimated)))
k = psd_estimated.sample_frequencies.data <= 20.0
mask[k] = 0.0
white_strain = (mask * simulated_signal.to_frequencyseries() / psd_estimated ** 0.5).to_timeseries()
# remove some of the high and low
smooth = highpass_fir(white_strain, 20, 8)
smooth = lowpass_fir(smooth, 1024, 8)

np.save(outputfile, np.array(smooth.data))
