from sound_processing import WavProcessor
from sound_reader import WavFileReader


class SoundChain:
    def __init__(self, reader: WavFileReader, *processors: WavProcessor) -> None:
        self._reader = reader
        self._processors = processors

    def run(self, fname):
        wav = self._reader.read(fname)
        for proc in self._processors:
            wav = proc.process(wav)
        return wav
