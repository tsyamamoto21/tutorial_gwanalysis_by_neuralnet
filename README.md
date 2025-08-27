# Tutorial for GW annalysis by neural network

## Dependencies

- `numpy`
- `matplotlib`
- `pycbc`: https://pycbc.org/
- `sbi`: https://github.com/sbi-dev/sbi

## Notebooks

- `bbh_search.ipynb`: searching GWs from BBH
- `damped_sinusoid.ipynb`: Ringdown analysis with SBI
- `spectroscopy.ipynb`: Ringdown with two modes with SBI

## Instruction for BBH search

- Create conda environment

```
conda env create -f requirement.yaml
conda activate dl4gw
```

- Generate training dataset

```
mkdir data/
./generate_waveform.py
```

- Generate test dataset

```
./generate_testdata.py
```

- Run notebook `bbh_search.ipynb`

