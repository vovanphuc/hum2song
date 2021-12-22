from pydub import AudioSegment, effects

def detect_leading_silence(sound, silence_threshold=-19.0, chunk_size=10):
    '''
    sound is a pydub.AudioSegment
    silence_threshold in dB
    chunk_size in ms

    iterate over chunks until you find the first one with sound
    '''
    trim_ms = 0 # ms
    assert chunk_size > 0 # to avoid infinite loop
    while sound[trim_ms:trim_ms+chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
        trim_ms += chunk_size
    return trim_ms

def get_valid_interval(sound: AudioSegment):
    normalizedsound = effects.normalize(sound)  
    head_sil = detect_leading_silence(normalizedsound)
    tail_sil = len(normalizedsound) - detect_leading_silence(normalizedsound.reverse())
    return (head_sil, tail_sil)