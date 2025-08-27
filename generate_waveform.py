#!/usr/bin/env python
import numpy as np
import torch
from tqdm import tqdm
# from scipy.signal.windows import tukey
from pycbc.filter import highpass
from pycbc.waveform import get_td_waveform
from pycbc.psd import aLIGOZeroDetHighPower, interpolate
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q


fs = 4096
duration = 1.2
merger_time = 0.9
length = int(duration * fs)
length_before_merger = int(merger_time * fs)
length_after_merger = length - length_before_merger
mcmin, mcmax = 30.0, 60.0
qmin, qmax = 1.0, 4.0
nsample_dict = {
    'train': 1000,
    'validate': 100,
}

delta_f = 1.0 / 4
flength = int(fs / delta_f)
low_frequency_cutoff = 20.0
psd = aLIGOZeroDetHighPower(flength, delta_f, low_frequency_cutoff)
k_zeropsd = psd.data == 0.0
psd.data[k_zeropsd] = 1e+10


for mode in ['train', 'validate', 'test']:
    outputfile = f'data/inputs_{mode}.pth'
    nsample = nsample_dict[mode]
    waveforms = torch.zeros((nsample, 2, length), dtype=torch.float32)
    mclist = np.random.uniform(mcmin, mcmax, (nsample,))
    qlist = np.random.uniform(qmin, qmax, (nsample,))
    m1list = mass1_from_mchirp_q(mclist, qlist)
    m2list = mass2_from_mchirp_q(mclist, qlist)
    for i in tqdm(range(nsample)):
        hp, hc = get_td_waveform(
            approximant='IMRPhenomD',
            mass1=m1list[i],
            mass2=m2list[i],
            delta_t=1.0 / fs,
            f_lower=15.0
        )
        # window = (1 + np.tanh((np.arange(len(hp)) / len(hp) - 0.2) / 0.03)) / 2.0
        hp_wh = (hp.to_frequencyseries() / interpolate(psd, 1.0 / hp.duration, len(hp) // 2 + 1) ** 0.5).to_timeseries()
        hc_wh = (hc.to_frequencyseries() / interpolate(psd, 1.0 / hp.duration, len(hc) // 2 + 1) ** 0.5).to_timeseries()
        hp_wh = highpass(hp_wh, 20.0)
        hc_wh = highpass(hc_wh, 20.0)

        # normalize
        amp = np.max(np.sqrt(hp_wh ** 2 + hc_wh**2).data)
        hp_wh /= amp
        hc_wh /= amp

        waveform_duration = hp_wh.duration
        signal_length = len(hp_wh)
        idx_merger = np.argmin(abs(hp_wh.sample_times))
        signal_length_before_merger = idx_merger
        signal_length_after_merger = signal_length - signal_length_before_merger
        if (signal_length_before_merger >= length_before_merger) and (signal_length_after_merger >= length_after_merger):
            k0 = signal_length_before_merger - length_before_merger
            k1 = k0 + length
            waveforms[i, 0] = torch.tensor(hp_wh.data[k0: k1], dtype=torch.float32)
            waveforms[i, 1] = torch.tensor(hc_wh.data[k0: k1], dtype=torch.float32)
        elif (signal_length_before_merger >= length_before_merger) and (signal_length_after_merger < length_after_merger):
            k0 = signal_length_before_merger - length_before_merger
            k1 = length_before_merger + signal_length_after_merger
            waveforms[i, 0, :k1] = torch.tensor(hp_wh.data[k0:], dtype=torch.float32)
            waveforms[i, 1, :k1] = torch.tensor(hc_wh.data[k0:], dtype=torch.float32)
        elif (signal_length_before_merger < length_before_merger) and (signal_length_after_merger >= length_after_merger):
            k0 = length_before_merger - signal_length_before_merger
            k1 = signal_length_before_merger + length_after_merger
            waveforms[i, 0, k0:] = torch.tensor(hp_wh.data[:k1], dtype=torch.float32)
            waveforms[i, 1, k0:] = torch.tensor(hc_wh.data[:k1], dtype=torch.float32)
        else:
            k0 = length_before_merger - signal_length_before_merger
            k1 = k0 + signal_length
            waveforms[i, 0, k0: k1] = torch.tensor(hp_wh.data, dtype=torch.float32)
            waveforms[i, 1, k0: k1] = torch.tensor(hc_wh.data, dtype=torch.float32)

    torch.save(waveforms, outputfile)
