from typing import Literal, Optional
import re
from dataclasses import dataclass

from abctk.obj.ID import RecordID

_parser_ABCTCompBCCWJ_ID = re.compile(
    r"^ABCT-COMP-BCCWJ;(?P<comp_type>yori|kurabe);(?P<sampleID>[^,]+),(?P<start_pos>[0-9]+)$"
)

@dataclass(frozen = True)
class ABCTComp_BCCWJ_ID(RecordID):
    """
    The internal structure of a BCCWJ comparative dataset ID.
    """

    comp_type: Literal["yori", "kurabe"]

    sampleID: str

    start_pos: int

    @classmethod
    def from_string(cls, ID: str) -> Optional["ABCTComp_BCCWJ_ID"]:
        """
        Parse an ID.

        Arguments
        ---------
        ID : str
            The ID to parse.

        Examples
        --------
        >>> k = ABCTComp_BCCWJ_ID.from_string("ABCT-COMP-BCCWJ;yori;LBn3_00147,24370")
        >>> k.comp_type
        yori
        >>> k.sampleID
        LBn3_00147
        >>> k.start_pos
        24370
        """

        match = _parser_ABCTCompBCCWJ_ID.match(ID)

        if match:
            d = match.groupdict()
            comp_type: str = d["comp_type"]
            if comp_type not in ("yori", "kurabe"):
                raise ValueError(f"{comp_type} is an invalid comparative type")
            
            return cls(
                comp_type = comp_type, # type: ignore
                sampleID = d["sampleID"],
                start_pos = int(d["start_pos"]),
            )
        else:
            return None

    def __str__(self):
        return f"ABCT-COMP-BCCWJ;{self.comp_type};{self.sampleID},{self.start_pos}"
