from sound_processing import WavProcessor
from sound_reader import WavFileReader


class SoundChain:
    def __init__(self, reader: WavFileReader, *processors: WavProcessor) -> None:
        self._reader = reader
        self._processors = processors

    def run(self, fname):
        wav = self._reader.read(fname)
        for proc in self._processors:
            try:
                wav = proc.process(wav)
            except Exception as e:
                print("Can't process file {} with {}: {}".format(fname, proc.__class__.__name__, e))
                continue  # TODO: maybe stop?
        return wav
