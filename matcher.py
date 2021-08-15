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

    def calc_cross_correlation(data, tmpl, shift):
        tmp = data[shift:shift+tmpl.size]
        return np.abs(np.sum(tmp * tmpl))

    procs = [delayed(calc_cross_correlation)(data, tmpl, s) for s in range(ns)]

    return Parallel(n_jobs=-1, verbose=10, batch_size=1024)(procs)


def calc_correlation_pyramid(
        data: np.ndarray, tmpl: np.ndarray, l: int):

    # sampling
    sdata = data[::l]
    stmpl = tmpl[::l]
    return np.array(correlation(sdata, stmpl))


if __name__ == '__main__':
    import os
    markers = [
        os.path.join(
            'markers', 'Audrey', f'{i+1:02d}.wav')
        for i in range(8)
    ]
    dsound = AudioSegment.from_file('20210808_Audrey.m4a')
    for tmpl_filename in markers:
        sound = split.split_to_mono_numpy(dsound)[0]
        dtemplate = AudioSegment.from_file(tmpl_filename)
        template = split.split_to_mono_numpy(dtemplate)[0]

        def normalize(v):
            v = v.astype(float)
            return 2 * (v - v.min()) / (v.max() - v.min()) - 1
        fsound = normalize(sound)
        ftemplate = normalize(template)

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
        ly0 = ax[0].get_ylim()
        for i, l in enumerate(levels):
            ai = i + 1
            coeff = coeffs[i]
            amax = np.argmax(coeff)
            x = np.arange(coeff.size) / dsound.frame_rate * l
            ax[ai].plot(x, coeff, label=f'{l}', alpha=0.5)

            args = np.argsort(coeff)
            for j in range(1):
                amax = args[-1-j]
                # if amax > 100:
                # ax[ai].plot([x[amax], x[amax]], ax[ai].get_ylim(),
                #             color=cmap(j), alpha=0.5, label=f'{j} th.')

                def s_to_timestr(s):
                    min = s // 60
                    h = int(min // 60)
                    min = int(min % 60)
                    s = int(s % 60)
                    return f'{h:02d}:{min:02d}:{s:02d}'
                print('play', coeff[amax], amax,
                      x[amax], s_to_timestr(x[amax]))

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
        figout = 'match_test_' + \
            os.path.splitext(
                os.path.basename(tmpl_filename))[0] + '.png'

        fig.savefig(figout)
        plt.close(fig)
