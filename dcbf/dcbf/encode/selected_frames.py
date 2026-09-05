from ase.io import iread


class SelectedFrames:
    """Original trajectory length and random lookup for only selected frames."""
    def __init__(self, path, indices):
        wanted = set(int(index) for index in indices)
        self.frames = {}
        self.count = 0
        for index, atoms in enumerate(iread(path)):
            self.count = index + 1
            if index in wanted:
                self.frames[index] = atoms
        missing = wanted.difference(self.frames)
        if missing:
            raise IndexError(f'Selected trajectory indices are missing: {sorted(missing)[:10]}')

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return self.frames[int(index)]
