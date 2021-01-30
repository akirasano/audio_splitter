import time
import array
import numpy as np
from pydub import AudioSegment
import simpleaudio


def play(data, dur=-1):
    """simpleaudioを使って、pydub.AudioSegmentを再生する

    Args:
        data (pydub.AudioSegment): An audio data.
        dur (int, optional):
            The duration of playback.
            If dur is zero or negative, play entire audio data.
            Defaults to -1.
    """
    po = simpleaudio.play_buffer(
        data.raw_data,
        num_channels=data.channels,
        bytes_per_sample=data.sample_width,
        sample_rate=data.frame_rate
    )

    if dur > 0:
        time.sleep(dur)
        po.stop()
    else:
        po.wait_done()


def print_info(data):
    """Print information of pydub.AudioSegmentの情報をprintする

    Args:
        data (pydub.AudioSegment): An audio data.
    """
    print(data.channels)
    print(data.sample_width)
    print(data.frame_rate)
    print(data.frame_width)
    print(type(data.raw_data))
    print(data.frame_count())
    print(len(data) / 1000.0)
    print(data.duration_seconds)
    print(data.frame_count() / data.frame_rate)


def split_to_mono_numpy(data):
    """Split pydub.AudioSegment to L and R channel data, and
       convert to numpy.ndarray respectively.

    Args:
        data (pydub.AudioSegment): An audio data.

    Returns:
        tuple of numpy.ndarray: (L ch., R ch.)
    """
    ld, rd = data.split_to_mono()
    la = ld.get_array_of_samples()
    ln = np.array(la)

    ra = rd.get_array_of_samples()
    rn = np.array(ra)

    return ln, rn


def combine_to_audiosegment(ln, rn, base):
    """Convert numpy.ndarray to pydub.AudioSegment

    Args:
        ln (numpy.ndarray): L ch. data.
        rn (numpy.ndarray): R ch. data.
        base (pydub.AudioSegment): original data.

    Returns:
        [type]: [description]
    """
    sn = np.column_stack((ln, rn)).flatten()
    sa = array.array(base.array_type, sn)
    return base._spawn(sa)


data = AudioSegment.from_file('20200531_ですよ.m4a')

print_info(data)

# 5秒再生
play(data, 5)

# チャネルを分けて、numpyに変換
ln, rn = split_to_mono_numpy(data)

# 最初5秒カット
skip = 5
skip_frames = skip * data.frame_rate # 5s * 48KHz
ln = ln[skip_frames:] # l, r それぞれ処理
rn = rn[skip_frames:]

# l, rのndarrayをAudioSegmentに変換
new_data = combine_to_audiosegment(ln, rn, data)

# 5秒再生
play(new_data, 5)
