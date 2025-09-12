from __future__ import annotations
"""Dataset generators for QLSTM reproduction.

Adapted from Quantum_Long_Short_Term_Memory (MIT License).
"""
import numpy as np, torch, os, csv, math
from sklearn.preprocessing import MinMaxScaler
from scipy.integrate import odeint
from scipy.special import jv
from scipy.stats import poisson

class BaseGenerator:
    def __init__(self, name:str):
        self.name=name; self.scaler=MinMaxScaler(feature_range=(-1,1)); self._raw=None; self._time=None
    def generate_raw(self):
        raise NotImplementedError
    def _ensure(self):
        if self._raw is None:
            self._time, self._raw = self.generate_raw()
        return self._time, self._raw
    def normalized(self):
        _, raw = self._ensure(); arr = raw.reshape(-1,1) if raw.ndim==1 else raw
        return self.scaler.fit_transform(arr)[:,0]
    def get_data(self, seq_len:int=4, target_dim: int = 1):
        norm = self.normalized(); xs=[]; ys=[]
        for i in range(len(norm)-seq_len-1):
            xs.append(norm[i:i+seq_len]); ys.append([norm[i+seq_len]])  # keep explicit dim
        x = torch.tensor(np.array(xs), dtype=torch.double)
        y = torch.tensor(np.array(ys), dtype=torch.double)
        if target_dim==1 and y.ndim==2: # shape (N,1)
            return x,y.squeeze(-1)
        return x,y

class SinGenerator(BaseGenerator):
    def __init__(self, frequency=0.2, amplitude=1.0, phase=0.0, t_max=20, n_points=240, noise_std=0.0):
        super().__init__('Sine'); self.frequency=frequency; self.amplitude=amplitude; self.phase=phase; self.t_max=t_max; self.n_points=n_points; self.noise_std=noise_std
    def generate_raw(self):
        t = np.linspace(0,self.t_max,self.n_points)
        data = self.amplitude * np.sin(2*np.pi*self.frequency*t + self.phase)
        if self.noise_std>0: data += np.random.normal(0,self.noise_std,len(data))
        return t,data

class DampedSHMGenerator(BaseGenerator):
    def __init__(self, b=0.15, g=9.81, l=1, m=1, theta_0=None, t_max=20, n_points=240):
        super().__init__('Damped SHM'); self.b=b; self.g=g; self.l=l; self.m=m; self.theta_0=theta_0 if theta_0 is not None else [0,3]; self.t_max=t_max; self.n_points=n_points
    def generate_raw(self):
        t = np.linspace(0,self.t_max,self.n_points)
        def system(theta, t_, b,g,l,m):
            th1, th2 = theta; return [th2, -(b/m)*th2 - g*math.sin(th1)]
        theta = odeint(system, self.theta_0, t, args=(self.b,self.g,self.l,self.m))
        return t, theta[:,1]

class CosGenerator(BaseGenerator):
    def __init__(self, frequency=0.2, amplitude=1.0, phase=0.0, t_max=20, n_points=240, noise_std=0.0):
        super().__init__('Cosine'); self.frequency=frequency; self.amplitude=amplitude; self.phase=phase; self.t_max=t_max; self.n_points=n_points; self.noise_std=noise_std
    def generate_raw(self):
        t = np.linspace(0,self.t_max,self.n_points)
        data = self.amplitude * np.cos(2*np.pi*self.frequency*t + self.phase)
        if self.noise_std>0: data += np.random.normal(0,self.noise_std,len(data))
        return t,data

class LinearGenerator(BaseGenerator):
    def __init__(self, slope=1.0, intercept=0.0, t_max=20, n_points=240, noise_std=0.0):
        super().__init__('Linear'); self.slope=slope; self.intercept=intercept; self.t_max=t_max; self.n_points=n_points; self.noise_std=noise_std
    def generate_raw(self):
        t = np.linspace(0,self.t_max,self.n_points)
        data = self.slope * t + self.intercept
        if self.noise_std>0: data += np.random.normal(0,self.noise_std,len(data))
        return t,data

class ExponentialGenerator(BaseGenerator):
    def __init__(self, growth_rate=0.1, initial_value=1.0, t_max=20, n_points=240, noise_std=0.0):
        super().__init__('Exponential'); self.growth_rate=growth_rate; self.initial_value=initial_value; self.t_max=t_max; self.n_points=n_points; self.noise_std=noise_std
    def generate_raw(self):
        t = np.linspace(0,self.t_max,self.n_points)
        data = self.initial_value * np.exp(self.growth_rate * t)
        if self.noise_std>0: data += np.random.normal(0,self.noise_std,len(data))
        return t,data

class BesselJ2Generator(BaseGenerator):
    def __init__(self, amplitude=1.0, x_scale=5.0, x_max=20, n_points=240, noise_std=0.0):
        super().__init__('Bessel J2'); self.amplitude=amplitude; self.x_scale=x_scale; self.x_max=x_max; self.n_points=n_points; self.noise_std=noise_std
    def generate_raw(self):
        x = np.linspace(0.1, self.x_max, self.n_points)
        scaled_x = self.x_scale * x
        data = self.amplitude * jv(2, scaled_x)
        if self.noise_std>0: data += np.random.normal(0,self.noise_std,len(data))
        return x, data

class PopulationInversionGenerator(BaseGenerator):
    def __init__(self, omega=1.0, amplitude=1.0, t_max=1000, n_points=240, noise_std=0.0):
        super().__init__('Population Inversion'); self.omega=omega; self.amplitude=amplitude; self.t_max=t_max; self.n_points=n_points; self.noise_std=noise_std
    def generate_raw(self):
        t = np.linspace(0,self.t_max,self.n_points)
        data = self.amplitude * np.cos(self.omega * t)
        if self.noise_std>0: data += np.random.normal(0,self.noise_std,len(data))
        return t, data

class PopulationInversionCollapseRevivalGenerator(BaseGenerator):
    def __init__(self, mean_n=40, g=1.0, t_max=200, n_points=2500, n_max=100, noise_std=0.0):
        super().__init__('Population Inversion CR'); self.mean_n=mean_n; self.g=g; self.t_max=t_max; self.n_points=n_points; self.n_max=n_max if n_max is not None else int(mean_n + 8*np.sqrt(mean_n)); self.noise_std=noise_std
    def generate_raw(self):
        t = np.linspace(0,self.t_max,self.n_points)
        n_vals = np.arange(0, self.n_max+1)
        P_n = poisson.pmf(n_vals, self.mean_n)
        cos_terms = np.cos(2 * self.g * np.outer(t, np.sqrt(n_vals+1)))
        W_t = cos_terms @ P_n
        if self.noise_std>0: W_t += np.random.normal(0,self.noise_std,len(W_t))
        return t, W_t

class LogSineGenerator(BaseGenerator):
    """Log-amplitude modulated sine: sin(t) * log(t+1)."""
    def __init__(self, t_max=20, n_points=240):
        super().__init__('LogSine'); self.t_max=t_max; self.n_points=n_points
    def generate_raw(self):
        t = np.linspace(0,self.t_max,self.n_points)
        data = np.sin(t) * np.log(t+1)
        return t,data

class MovingAverageNoiseGenerator(BaseGenerator):
    """White noise smoothed by simple moving average to create autocorrelation."""
    def __init__(self, t_max=20, n_points=240, window=5, seed=0):
        super().__init__('MA Noise'); self.t_max=t_max; self.n_points=n_points; self.window=window; self.seed=seed
    def generate_raw(self):
        rng = np.random.default_rng(self.seed)
        t = np.linspace(0,self.t_max,self.n_points)
        noise = rng.normal(0,1,self.n_points)
        if self.window>1:
            kernel = np.ones(self.window)/self.window
            smooth = np.convolve(noise, kernel, mode='same')
        else:
            smooth = noise
        return t, smooth

class CSVSeriesGenerator(BaseGenerator):
    """Load a univariate time series from a CSV file with a single column or 'value' column."""
    def __init__(self, path:str):
        super().__init__('CSVSeries'); self.path=path
    def generate_raw(self):
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        values=[]
        with open(self.path) as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                # try second column if exists
                candidate = None
                if len(row) > 1:
                    candidate = row[1]
                else:
                    candidate = row[0]
                try:
                    v = float(candidate)
                except ValueError:
                    continue  # likely header
                values.append(v)
        t = np.arange(len(values))
        return t, np.array(values, dtype=float)

class DataFactory:
    _registry = {
        'sin': SinGenerator,
        'cos': CosGenerator,
        'linear': LinearGenerator,
        'exp': ExponentialGenerator,
        'besselj2': BesselJ2Generator,
        'pop_inv': PopulationInversionGenerator,
        'pop_inv_cr': PopulationInversionCollapseRevivalGenerator,
        'damped_shm': DampedSHMGenerator,
        'logsine': LogSineGenerator,
        'ma_noise': MovingAverageNoiseGenerator,
        # CSV handled specially
    }
    @classmethod
    def get(cls, name:str, **kw):
        if name == 'csv':
            path = kw.get('path')
            if not path:
                raise ValueError("CSV generator requires 'path' argument")
            return CSVSeriesGenerator(path)
        if name not in cls._registry:
            raise ValueError(f'Unknown generator {name}; available: {list(cls._registry)+['csv']}')
        return cls._registry[name](**kw)
    @classmethod
    def list(cls):
        return list(cls._registry)+['csv']

data = DataFactory
