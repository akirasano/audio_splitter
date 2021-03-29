# %%
from joblib import Parallel, delayed
from scipy import interpolate
import scipy
import matplotlib.pyplot as plt
import numpy as np
import time
import array
import numpy as np
from pydub import AudioSegment
import simpleaudio
import split
# %%
data = AudioSegment.from_file('20200531_ですよ.m4a')
# %%
ln, rn = split.split_to_mono_numpy(data)
ss = 3*data.frame_rate
se = 5*data.frame_rate
ts = 4*data.frame_rate
te = int(4.2*data.frame_rate)
sound = ln[ss:se]
template = ln[ts:te]

# %%


def play_numpy(a: np.ndarray, base: AudioSegment, dur=-1):
    """numpy.ndarrayを直接再生する

    Args:
        a (numpy.ndarray): [description]
    """
    po = simpleaudio.play_buffer(
        array.array(base.array_type, a),
        num_channels=1,
        bytes_per_sample=data.sample_width,
        sample_rate=data.frame_rate
    )

    if dur > 0:
        time.sleep(dur)
        po.stop()
    else:
        po.wait_done()


play_numpy(template, data)

# %%

# %%
ns = ln.size
x = np.arange(ns)
f = interpolate.interp1d(x, ln)
x1 = x[::100]
fig, ax = plt.subplots(3, 1, figsize=(8, 8))
ax[0].plot(x, ln, '.')
ax[1].plot(x1, f(x1), '.')
ax[2].plot(x, ln, '.')
ax[2].plot(x1, f(x1), '.')
for a in ax.flat:
    a.set_xlim([3e+4, 3.5e4])
plt.show()
plt.close(fig)

# %%


def correlation(data: np.ndarray, tmpl: np.ndarray):
    assert data.ndim == 1
    assert tmpl.ndim == 1
    assert data.size >= tmpl.size

    ns = data.size - tmpl.size
    # coeff = np.zeros((ns, ))
    # padded_tml = np.ndarray((data.size, ))

    def calc_coeff(data, tmpl, shift):
        tmp = data.copy()
        tmp[shift:shift+tmpl.size] -= tmpl
        return np.linalg.norm(tmp, ord=2)
    procs = [delayed(calc_coeff)(data, tmpl, s) for s in range(ns)]
    # for s in range(ns):
    #     tmp = data.copy()
    #     tmp[s:s+tmpl.size] -= tmpl
    #     # padded_tml[:] = 0
    #     # padded_tml[s:s+tmpl.size] = tmpl
    #     coeff[s] = np.linalg.norm(tmp, ord=2)
    return Parallel(n_jobs=-1, verbose=10, batch_size=16)(procs)


def calc_correlation_pyramid(
    data: np.ndarray,
    tmpl: np.ndarray,
    # levels=[16, 8, 4, 2, 1]):
        l):

    dx = np.arange(data.size)
    tx = np.arange(tmpl.size)
    # for l in levels:
    sdx = np.arange(0, data.size, l)
    sdata = scipy.interpolate.interp1d(dx, data)(sdx)

    stx = np.arange(0, tmpl.size, l)
    stmpl = scipy.interpolate.interp1d(tx, tmpl)(stx)

    return np.array(correlation(sdata, stmpl))


fig, ax = plt.subplots(3, 1, figsize=(4, 8))
xdata = np.arange(ln.size) / data.frame_rate + 3
ax[0].plot(xdata, ln)
xtemplate = np.arange(template.size) / data.frame_rate + 4
ax[1].plot(xtemplate, template)
# ax[2].plot(correlation(ln, template))
l = 64
coeff = calc_correlation_pyramid(ln, template, l)
xcoeff = np.arange(coeff.size) / data.frame_rate * l
ax[2].plot(xcoeff, coeff)
amin = np.argmin(coeff)
print(amin / data.frame_rate * l)
fig.tight_layout()
plt.show()
plt.close(fig)


# %%
fig, ax = plt.subplots(5, 1, figsize=(8, 12))
for i, l in enumerate([64, 128, 256, 512, 1024]):
    coeff = calc_correlation_pyramid(ln, template, l)
    amin = np.argmin(coeff)
    print(l, amin / data.frame_rate * l)
    x = np.arange(coeff.size) / data.frame_rate * l
    ax[i].plot(x, coeff, label=f'{l}')
    ax[i].plot([x[amin], x[amin]], ax[i].get_ylim())
    ax[i].grid()
    ax[i].legend()
plt.show(fig)