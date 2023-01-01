import re
from typing import NamedTuple

_parser_Keyaki_ID = re.compile(
    r"^(?P<number>[0-9]+)_(?P<name>[^;]+)(;(?P<suffix>.*))?$"
)

_counter_default: int = 0

class Keyaki_ID(NamedTuple):
    """
    The internal structure of a Keyaki tree ID.
    """

    name: str
    """
    The name of the file which the tree with the ID should belong.
    """

    number: int
    """
    The number of the tree.
    """

    suffix: str
    """
    Additional annotations.
    """

    orig: str = ""

    @classmethod
    def new(cls) -> "Keyaki_ID":
        global _counter_default
        _counter_default += 1

        return cls(
            name = "",
            number = _counter_default,
            suffix = "",
            orig = "",
        )

    @classmethod
    def from_string(cls, ID: str) -> "Keyaki_ID":
        """
        Parse a Keyaki tree ID.

        Notes
        -----
        If `name` is empty, it means that the parsing has failed.
        The original ID is stored in `orig`.

        Examples
        --------
        >>> k = Keyaki_ID.from_string("1433_something")
        >>> k.name
        'something'
        >>> k.number
        1433
        >>> k.suffix
        ''

        >>> i = Keyaki_ID.from_string("1_spoken-closed-CSJ_04_S00F0066_CU;CUStartTime=0.223425_CUEndTime=1.914907_IPUID=0001;JP")
        >>> i.name
        'spoken-closed-CSJ_04_S00F0066_CU'
        >>> i.number
        1
        >>> i.suffix
        'CUStartTime=0.223425_CUEndTime=1.914907_IPUID=0001;JP'
        """

        global _counter_default

        match = _parser_Keyaki_ID.match(ID)

        if match:
            d = match.groupdict()
            return cls(
                name = d["name"],
                number = int(d["number"]),
                suffix = d["suffix"] or "",
                orig = ID,
            )
        elif ID.isnumeric():
            return cls(
                name = "",
                number = int(ID),
                suffix = "",
                orig = ID
            )
        else:
            _counter_default += 1

            return cls(
                name = "",
                number = _counter_default,
                suffix = "",
                orig = ID,
            )

    def __str__(self):
        suffix = self.suffix
        if suffix:
            suffix = ";" + suffix

        return f"{self.number}_{self.name}{suffix}"

    def tell_path(
        self,
        fmt: str = "{is_closed}/{name}.psd"
    ) -> str:
        """
        Tell the (relative) path to the file to which the tree should belong.

        Arguments
        ---------
        fmt
            The format of the path.
            Available variables: is_closed, name

        Examples
        --------
        >>> k = Keyaki_ID.from_string("2_aozora_Chiri-1956;JP")
        >>> k.tell_path("prefix/prefixes/{name}-b2psg.psd")
        'prefix/prefixes/aozora_Chiri-1956-b2psg.psd'

        >>> i = Keyaki_ID.from_string("1_spoken-closed-CSJ_04_S00F0066_CU;CUStartTime=0.223425_CUEndTime=1.914907_IPUID=0001;JP")
        >>> i.tell_path()
        'closed/spoken-closed-CSJ_04_S00F0066_CU.psd'
        """

        # TODO: property cache

        name = self.name or "UNKNOWN"
        
        is_closed_str: str
        if name.find("closed") > 0:
            is_closed_str = "closed"
        elif name:
            is_closed_str = "treebank"
        else:
            is_closed_str = ""

        return fmt.format(
            is_closed = is_closed_str,
            name = name
        )