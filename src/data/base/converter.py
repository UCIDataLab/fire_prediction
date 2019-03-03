"""
Base class for data converters.
"""


class Converter(object):
    def __init__(self):
        pass

    @staticmethod
    def load(src):
        raise NotImplementedError()

    @staticmethod
    def transform(data):
        raise NotImplementedError()

    @staticmethod
    def save(dest, data):
        raise NotImplementedError()

    def convert(self, data_dir, dest=None):
        """
        Converts data using self.transform. 
        
        If 'dest' is None, uses 'INPUT_FMT' and 'OUTPUT_FMT' class variables to select file(s) to load/store. 
        Else uses 'data_dir' and 'dest' explicitly.
        """
        # Choose which file selection mode is used
        # if dest:
        src = data_dir
        # else:
        #    src = self.INPUT_FMT.run(data_dir)
        #    dest = self.INPUT_FMT.run(data_dir)

        data = self.load(src)
        converted_data = self.transform(data)
        self.save(dest, converted_data)
