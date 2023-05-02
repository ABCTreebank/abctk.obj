import typing
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
    orig: typing.Optional[str] = None

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
    
