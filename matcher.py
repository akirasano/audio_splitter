from itertools import filterfalse
from joblib import Parallel, delayed
from numpy.core.shape_base import stack
from scipy import interpolate
import scipy
import matplotlib.pyplot as plt
import numpy as np
import time
import array
from pydub import AudioSegment, audio_segment
import simpleaudio
import split


def play_numpy(a: np.ndarray, base: AudioSegment, dur=-1):
    """numpy.ndarrayを直接再生する

    Args:
        a (numpy.ndarray): [description]
    """
    po = simpleaudio.play_buffer(
        array.array(base.array_type, a),
        num_channels=1,
        bytes_per_sample=base.sample_width,
        sample_rate=base.frame_rate
    )

    if dur > 0:
        time.sleep(dur)
        po.stop()
    else:
        po.wait_done()


def correlation(data: np.ndarray, tmpl: np.ndarray):
    assert data.ndim == 1
    assert tmpl.ndim == 1
    assert data.size >= tmpl.size

    ns = data.size - tmpl.size
    # coeff = np.zeros((ns, ))
    # padded_tml = np.ndarray((data.size, ))

    def calc_coeff(data, tmpl, shift):
        # tmp = data.copy()
        tmp = data[shift:shift+tmpl.size] - tmpl
        return np.linalg.norm(tmp, ord=1) / tmpl.size

    def calc_cross_correlation(data, tmpl, shift):
        tmp = data[shift:shift+tmpl.size]
        return -np.abs(np.sum(tmp * tmpl))

    # procs = [delayed(calc_coeff)(data, tmpl, s) for s in range(ns)]
    procs = [delayed(calc_cross_correlation)(data, tmpl, s) for s in range(ns)]

    # for s in range(ns):
    #     tmp = data.copy()
    #     tmp[s:s+tmpl.size] -= tmpl
    #     # padded_tml[:] = 0
    #     # padded_tml[s:s+tmpl.size] = tmpl
    #     coeff[s] = np.linalg.norm(tmp, ord=2)
    return Parallel(n_jobs=-1, verbose=10, batch_size=1024)(procs)
    # return Parallel(n_j bs=-1)(procs)


def calc_correlation_pyramid(
        data: np.ndarray, tmpl: np.ndarray, l: int,
        sampling: bool = True):

    if sampling:
        # dx = np.arange(data.size)
        # sdx = np.arange(0, data.size, l)
        # sdata = scipy.interpolate.interp1d(dx, data)(sdx)
        sdata = data[::l]
        sdata = sdata - sdata.mean()

        # tx = np.arange(tmpl.size)
        # stx = np.arange(0, tmpl.size, l)
        # stmpl = scipy.interpolate.interp1d(tx, tmpl)(stx)
        stmpl = tmpl[::l]
        stmpl = stmpl - stmpl.mean()
    else:
        # binning
        # lの整数倍。後ろは捨てる

        ld = data.size // l * l
        sdata = data[:ld].reshape(-1, l).sum(axis=1) / l

        ls = tmpl.size // l * l
        stmpl = data[:ls].reshape(-1, l).sum(axis=1) / l

    return np.array(correlation(sdata, stmpl))


if __name__ == '__main__':
    dsound = AudioSegment.from_file('20210813_Turn.wav')
    sound = split.split_to_mono_numpy(dsound)[0]
    # dtemplate = AudioSegment.from_file('Turn_jingle_02.wav')
    dtemplate = AudioSegment.from_file('20200531_ですよ.m4a')
    template = split.split_to_mono_numpy(dtemplate)[0]

    from scipy import signal

    def lpf(wave, th, n=5):
        b, a = signal.butter(1, th, btype='low')
        for i in range(0, n):
            wave = signal.filtfilt(b, a, wave)
        return wave

    # fsound = lpf(sound, 1.0/16.0)
    fsound = sound
    # ftemplate = lpf(template, 1.0/16.0)
    ftemplate = template

    # play_numpy(template, dtemplate, dur=3)

    # level_exps = [10, 11]
    # levels = [2**e for e in level_exps]
    levels = [1024]
    proc_times = []
    coeffs = []
    for i, l in enumerate(levels):
        s = time.perf_counter()
        coeff = calc_correlation_pyramid(fsound, ftemplate, l)
        coeffs.append(coeff)
        e = time.perf_counter()
        proc_times.append(e - s)

    cmap = plt.get_cmap('tab10')
    nr = len(levels) + 2
    fig, ax = plt.subplots(nr, 1, figsize=(8, nr*2))
    t = np.arange(sound.size) / dsound.frame_rate
    ax[0].plot(t, sound, color='gray', alpha=0.5)
    ax[0].set_xlim((1150, 1200))
    ly0 = ax[0].get_ylim()
    for i, l in enumerate(levels):
        ai = i + 1
        coeff = coeffs[i]
        amax = np.argmin(coeff)
        x = np.arange(coeff.size) / dsound.frame_rate * l
        ax[ai].plot(x, coeff, label=f'{l}', alpha=0.5)

        args = np.argsort(coeff)
        for j in range(5):
            amax = args[j]
            # if amax > 100:
            ax[ai].plot([x[amax], x[amax]], ax[ai].get_ylim(),
                        color=cmap(j), alpha=0.5, label=f'{j} th.')

            def s_to_timestr(s):
                min = s // 60
                h = int(min // 60)
                min = int(min % 60)
                s = int(s % 60)
                return f'{h:02d}\:{min:02d}\:{s:02d}'
            print('play', amax, x[amax], s_to_timestr(x[amax]))
            # play_numpy(sound[amax*l:], dsound, dur=3)
            # break

        r0 = i / len(levels)
        r1 = (i + 1) / len(levels)
        hy = ly0[1] - ly0[0]
        ly = [ly0[0] + hy*r0, ly0[0] + hy*r1]
        ax[0].plot([x[amax], x[amax]], ly,
                   color=cmap(i), alpha=0.9, label=f'{l}')
    ax[-1].bar(np.arange(len(proc_times)), proc_times, tick_label=levels)
    ax[-1].set_xlabel('Level')
    ax[-1].set_ylabel('Proc. time [s]')
    for a in ax.flat:
        a.grid()
        a.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig('match_test.png')
