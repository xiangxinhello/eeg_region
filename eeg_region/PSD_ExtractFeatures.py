import torch
import torch.nn.functional as F
import scipy
import numpy as np
from scipy.fftpack import fft, rfft, fftfreq, irfft, ifft, rfftfreq
from scipy import signal as ss


def _periodogram(X: torch.Tensor, fs, detrend, scaling):
    if X.dim() > 2:
        X = torch.squeeze(X)
    elif X.dim() == 1:
        X = X.unsqueeze(0)

    if detrend:
        X -= X.mean(-1, keepdim=True)

    N = X.size(-1)
    assert N % 2 == 0

    df = fs / N
    dt = df
    f = torch.arange(0, N / 2 + 1) * df  # [0:df:f/2]

    dual_side = np.fft.fft(X)  # 双边谱
    dual_side = torch.tensor(dual_side)
    half_idx = int(N / 2 + 1)
    single_side = dual_side[:, 0:half_idx]
    win = torch.abs(single_side)

    ps = win ** 2
    if scaling == 'density':  # 计算功率谱密度
        scale = N * fs
    elif scaling == 'spectrum':  # 计算功率谱
        scale = N ** 2
    elif scaling is None:  # 不做缩放
        scale = 1
    else:
        raise ValueError('Unknown scaling: %r' % scaling)
    Pxy = ps / scale

    Pxy[:, 1:-1] *= 2  # 能量2倍;直流分量不用二倍, 中心频率点不用二倍

    return f, Pxy.squeeze()


def periodogram(X: torch.Tensor, fs=256, detrend=False, scaling='density', no_grad=True):
    """计算信号单边 PSD, 基本等价于 scipy.signal.periodogram

        Parameters:
        ----------
            - `X`:          torch.Tensor, EEG, [T]/[N, T]
            - `fs`:         int, 采样率, Hz
            - `detrend`:    bool, 是否去趋势 (去除直流分量)
            - `scaling`:    { 'density', 'spectrum' }, 可选
                - 'density':    计算功率谱密度 `(V ** 2 / Hz)`
                - 'spectrum':    计算功率谱 `(V ** 2)`
            - `no_grad`:    bool, 是否启用 no_grad() 模式

        Returns:
        ----------
            - `Pxy`:    Tensor, 单边功率谱
    """
    if no_grad:
        with torch.no_grad():
            return _periodogram(X, fs, detrend, scaling)
    else:
        return _periodogram(X, fs, detrend, scaling)

def _get_window(window, nwlen, device):
    if window == 'hann':
        window = torch.hann_window(
            nwlen, dtype=torch.float32, device=device, periodic=False
        )
    elif window == 'hamming':
        window = torch.hamming_window(
            nwlen, dtype=torch.float32, device=device, periodic=False
        )
    elif window == 'blackman':
        window = torch.blackman_window(
            nwlen, dtype=torch.float32, device=device, periodic=False
        )
    elif window == 'boxcar':
        window = torch.ones(nwlen, dtype=torch.float32, device=device)
    else:
        raise ValueError('Invalid Window {}' % window)
    return window


def _pwelch(X: torch.Tensor, fs, detrend, scaling, window, nwlen, nhop):
    # X = ensure_3dim(X)
    if scaling == 'density':
        scale = (fs * (window * window).sum().item())
    elif scaling == 'spectrum':
        scale = window.sum().item() ** 2
    else:
        raise ValueError('Unknown scaling: %r' % scaling)
    # --------------- Fold and windowing --------------- #
    N, T = X.size(0), X.size(-1)
    X = X.view(N, 1, 1, T)
    X_fold = F.unfold(X, (1, nwlen), stride=nhop)  # [N, 1, 1, T] -> [N, nwlen, win_cnt]
    if detrend:
        X_fold -= X_fold.mean(1, keepdim=True) # 各个窗口各自detrend
    window = window.view(1, -1, 1)  # [1, nwlen, 1]
    X_windowed = X_fold * window  # [N, nwlen, win_cnt]
    win_cnt = X_windowed.size(-1)

    # --------------- Pwelch --------------- #
    X_windowed = X_windowed.transpose(1, 2).contiguous()  # [N, win_cnt, nwlen]
    X_windowed = X_windowed.view(N * win_cnt, nwlen)  # [N * win_cnt, nwlen]
    f, pxx = _periodogram(
        X_windowed, fs, detrend=False, scaling=None
    )  # [N * win_cnt, nwlen // 2 + 1]
    pxx /= scale
    pxx = pxx.view(N, win_cnt, -1)  # [N, win_cnt, nwlen // 2 + 1]
    pxx = torch.mean(pxx, dim=1)  # [N, nwlen // 2 + 1]
    return f, pxx


def pwelch(
        X: torch.Tensor,
        fs=256,
        detrend=False,
        scaling='density',
        window='hann',
        nwlen=256,
        nhop=None,
        no_grad=True,
):
    """Pwelch 方法，大致相当于 scipy.signal.welch

        Parameters:
        ----------
            - `X`:          torch.Tensor, EEG, [T]/[N, T]
            - `fs`:         int, 采样率, Hz
            - `detrend`:    bool, 是否去趋势 (去除直流分量)
            - `scaling`:    { 'density', 'spectrum' }, 可选
                - 'density':    计算功率谱密度 `(V ** 2 / Hz)`
                - 'spectrum':    计算功率谱 `(V ** 2)`
            - `window`:     str, 窗函数名称
            - `nwlen`:      int, 窗函数长度 (点的个数)
            - `nhop`:       int, 窗函数移动步长, 即 nwlen - noverlap (点的个数)
                            如果为 None，则默认为 `nwlen // 2`
            - `no_grad`:    bool, 是否启用 no_grad() 模式

        Returns:
        ----------
            - `Pxy`:    Tensor, 单边功率谱
    """
    nhop = nwlen // 2 if nhop is None else nhop
    X = torch.tensor(X)
    window = _get_window(window, nwlen, X.device)
    if no_grad:
        with torch.no_grad():
            return _pwelch(X, fs, detrend, scaling, window, nwlen, nhop)
    else:
        return _pwelch(X, fs, detrend, scaling, window, nwlen, nhop)



def bandpower(x, fs, fmin, fmax):
    # x = x.numpy()
    # fs = fs.numpy()
    f, Pxx = pwelch(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax)
    a = Pxx[:, ind_min: ind_max]
    b = f[ind_min: ind_max]
    res = np.trapz(Pxx[:, ind_min: ind_max], f[ind_min: ind_max])
    return torch.tensor(res)

def PSD_Etract(eeg, fs=1000):
    f, pxx_t = pwelch(eeg, fs=1000)
    power_delta = bandpower(eeg, 1000, 0.5, 4)
    power_theta = bandpower(eeg, 1000, 4, 8)
    power_alpha = bandpower(eeg, 1000, 8, 14)
    power_beta = bandpower(eeg, 1000, 14, 30)
    power_gamma = bandpower(eeg, 1000, 30, 70)
    pxx_t_mean = torch.mean(pxx_t, 0)
    DE_delta = torch.log2(power_delta)
    DE_theta = torch.log2(power_theta)
    DE_alpha = torch.log2(power_alpha)
    DE_beta = torch.log2(power_beta)
    DE_gamma = torch.log2(power_gamma)
    DE_mean = torch.log2(pxx_t_mean)
    # PSD_data = np.concatenate([power_delta,power_theta,power_alpha,power_beta,power_gamma,pxx_t_mean[:20]],)
    # PSD_data = np.concatenate([power_alpha,power_beta,power_gamma,pxx_t_mean[:20]],)
    # DE_data = np.concatenate([DE_delta,DE_theta,DE_alpha,DE_beta,DE_gamma,DE_mean[:20]],)
    DE_data = np.concatenate([DE_delta.unsqueeze(0),DE_theta.unsqueeze(0),DE_alpha.unsqueeze(0),DE_beta.unsqueeze(0),DE_gamma.unsqueeze(0),DE_mean[:128].unsqueeze(0)],)
    return DE_data

