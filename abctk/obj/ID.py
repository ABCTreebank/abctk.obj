from typing import List, Optional, Tuple, Callable

from dataclasses import dataclass
from abc import ABCMeta, abstractmethod

class RecordID(metaclass = ABCMeta):
    def __new__(cls, *args, **kwargs):
        dataclass(cls)
        return super().__new__(cls)

    @classmethod
    @abstractmethod
    def from_string(cls, ID: str):
        """
        Try parsing an ID.

        Arguments
        ---------
        ID : str
            The ID to parse.
        """
        ...

@dataclass(frozen = True)
class SimpleRecordID(RecordID):
    orig: Optional[str] = None

    @classmethod
    def from_string(cls, ID: str):
        """
        Try parsing an ID.

        Arguments
        ---------
        ID : str
            The ID to parse.
        """
        return cls(orig = ID)


class RecordIDParser:
    _parser: List[
        Tuple[int, Callable[[str], RecordID]]
    ] = [(0, SimpleRecordID.from_string)]

    def register_parser(self, parser, weight: int = 0) -> None:
        self._parser.append((weight, parser))
        self._parser.sort(
            key = lambda i: i[0],
            reverse = True,
        )

    def parse(self, ID: str) -> "RecordID":
        return next(
            (
                result for result 
                in (parse(ID) for _, parse in self._parser)
                if result is not None
            )
        )