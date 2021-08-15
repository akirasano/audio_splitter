import sys
import os
from glob import glob
import numpy as np
import subprocess
from numpy.core.numeric import indices
from tqdm import tqdm
from pydub import AudioSegment
from split import split_to_mono_numpy
from matcher import calc_correlation_pyramid


def ms_to_timestr(ms):
    s = ms // 1000
    min = s // 60
    h = min // 60
    min = min % 60
    s = s % 60

    return f'{h:02d}:{min:02d}:{s:02d}'


def write_meta(out, marker_times_ms, end_time_ms):
    with open(out, 'w') as fp:
        fp.write(';FFMETADATA\n\n')

        for i in range(len(marker_times_ms)):
            t0 = marker_times_ms[i]
            if i == len(marker_times_ms) - 1:
                t1 = end_time_ms
            else:
                t1 = marker_times_ms[i + 1]
            ts0 = ms_to_timestr(t0)
            fp.write(
                '[CHAPTER]\n'
                'TIMEBASE=1/1000\n'
                f'START={t0:d}\n'
                f'END={t1:d}\n'
                f'title=Chapter{i:d}: {ts0}\n'
                '\n'
            )


def make_figure(figout, audio_data, coeffs, cindices, l):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('tab10')

    audio = split_to_mono_numpy(audio_data)[0]
    nr = len(coeffs) + 1

    fig, ax = plt.subplots(nr, 1, figsize=(8, nr*2))

    t = np.arange(audio.size) / audio_data.frame_rate
    ax[0].plot(t, audio, color='gray', alpha=0.5)

    ly0 = ax[0].get_ylim()
    for i, (coeff, idx) in enumerate(zip(coeffs, cindices)):
        ai = i + 1

        x = np.arange(coeff.size) / audio_data.frame_rate * l
        ax[ai].plot(x, coeff, label=f'{i}', alpha=0.5)

        ax[0].annotate(
            f'{i}', xy=(t[idx], 0),
            xytext=(t[idx], ly0[1]),
            arrowprops=dict(arrowstyle='->', color=cmap(i))
        )

    for a in ax.flat:
        a.grid()
        a.legend(loc='lower left')

    fig.tight_layout()
    fig.savefig(figout)
    plt.close(fig)


def calc_marker_position(wave_lch, frame_rate, level, markers):
    def normalize(v):
        v = v.astype(float)
        return 2 * (v - v.min()) / (v.max() - v.min()) - 1

    wave_lch = normalize(wave_lch)

    chapter_indices = []
    coeffs = []

    last_sample_pos = 0
    for m in tqdm(markers):
        marker_data = AudioSegment.from_file(m)
        assert(marker_data.frame_rate == frame_rate)
        marker_wave_lch = split_to_mono_numpy(marker_data)[0]
        marker_wave_lch = normalize(marker_wave_lch)

        coeff = calc_correlation_pyramid(
            wave_lch, marker_wave_lch, level)
        coeffs.append(coeff)
        args = np.argsort(coeff)
        for j in range(args.size):
            sample_pos = args[-1-j] * level
            if sample_pos > last_sample_pos:
                chapter_indices.append(sample_pos)
                last_sample_pos = sample_pos
                break

    marker_times_ms = [0] + [
        int(i / frame_rate * 1000)
        for i in chapter_indices
    ]

    return marker_times_ms, coeffs, chapter_indices


def add_chapter(src, dst, meta_file, markers, level=256):

    audio_data = AudioSegment.from_file(src)
    wave_lch = split_to_mono_numpy(audio_data)[0]

    marker_times_ms, coeffs, chapter_indices = \
        calc_marker_position(
            wave_lch, audio_data.frame_rate,
            level, markers)

    # avconvがあるとffmpegよりも優先して使う
    write_meta(meta_file, marker_times_ms,
               len(wave_lch) // audio_data.frame_rate * 1000)

    if True:
        base = os.path.splitext(os.path.basename(src))[0]
        make_figure(
            os.path.join(base, 'match.png'),
            audio_data, coeffs, chapter_indices, level)

    #  '-acodec', 'copy',
    # 48kのファイルはQuicktimeやbookでうまく再生できないので
    # 44.1にリサンプルする
    ret = subprocess.run(
        ['ffmpeg',
         '-y',
         '-i', src,
         '-i', meta_file,
         '-map_chapters', '1',
         '-acodec', 'libfdk_aac',
         '-profile:a', 'aac_he_v2',
         #  '-ab', '48k',
         '-ab', '32k',
         '-ar', '44.1k',
         dst]
    )

    return audio_data, coeffs, chapter_indices


def show_params(**kwargs):
    for k, v in kwargs.items():
        print(f'{k:12s}: {v}')


if __name__ == '__main__':

    src = sys.argv[1]
    title = src.split('_')[-1].split('.')[0]
    marker_dir = os.path.join('markers', title)

    base = os.path.splitext(os.path.basename(src))[0]
    os.makedirs(base, exist_ok=True)
    dst = os.path.join(base, base + '.m4b')
    meta_file = os.path.join(base, 'meta.txt')

    show_params(src=src, title=title, base=base,
                dst=dst, meta_file=meta_file, marker_dir=marker_dir)

    markers = sorted(list(glob(os.path.join(marker_dir, '*.wav'))))
    print(markers)

    add_chapter(src, dst, meta_file, markers, level=256)
