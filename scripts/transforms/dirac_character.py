
class DiracCharacter(object):

    def __init__(self):

        self.U = None
        self.block_size = None
        self.block_start_idx = None
        self.block_idx = None

    @property
    def n_irreps(self):
        return len(self.block_size)
