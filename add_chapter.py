import os
import numpy as np
import subprocess
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


if __name__ == '__main__':

    src = '20210813_Turn.m4a'
    dst = os.path.splitext(os.path.basename(src))[0] + '.m4b'
    meta_file = 'tmp_meta.txt'

    markers = [
        'Turn_jingle_01.wav',
        'Turn_jingle_02.wav',
        'Turn_jingle_03.wav',
        'Turn_jingle_04.wav',
    ]

    chapter_indices = []

    def normalize(v):
        v = v.astype(float)
        return 2 * (v - v.min()) / (v.max() - v.min()) - 1

    # avconvがあるとffmpegよりも優先して使う
    audio_data = AudioSegment.from_file(src)
    wave_lch = split_to_mono_numpy(audio_data)[0]
    wave_lch = normalize(wave_lch)

    level = 1024

    last_sample_pos = 0
    for m in tqdm(markers):
        marker_data = AudioSegment.from_file(m)
        assert(marker_data.frame_rate == audio_data.frame_rate)
        marker_wave_lch = split_to_mono_numpy(marker_data)[0]
        marker_wave_lch = normalize(marker_wave_lch)

        coeff = calc_correlation_pyramid(wave_lch, marker_wave_lch, level)
        args = np.argsort(coeff)
        for j in range(args.size):
            sample_pos = args[-1-j] * level
            if sample_pos > last_sample_pos:
                chapter_indices.append(sample_pos)
                last_sample_pos = sample_pos
                break

    marker_times_ms = [0] + [
        int(i / audio_data.frame_rate * 1000)
        for i in chapter_indices
    ]
    write_meta(meta_file, marker_times_ms,
               len(wave_lch) // audio_data.frame_rate * 1000)

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
