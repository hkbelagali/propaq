from ._pauli_string import PauliString

class PauliTermStreamer:
    @staticmethod
    def from_file(path: str) -> PauliTermStreamer:
        """
        Open a gzip-compressed binary file for lazy term streaming.

        Arguments:
            path: Path to a file written by PauliTermSum.save().
        """
        ...

    def __iter__(self) -> PauliTermStreamer: ...
    def __next__(self) -> tuple[PauliString, complex]: ...
