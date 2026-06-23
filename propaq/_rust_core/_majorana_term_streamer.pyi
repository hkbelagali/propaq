from ._majorana_monomial import MajoranaMonomial

class MajoranaTermStreamer:
    @staticmethod
    def from_file(path: str) -> MajoranaTermStreamer:
        """
        Open a gzip-compressed binary file for lazy term streaming.

        Arguments:
            path: Path to a file written by MajoranaTermSum.save().
        """
        ...

    def __iter__(self) -> MajoranaTermStreamer: ...
    def __next__(self) -> tuple[MajoranaMonomial, complex]: ...
