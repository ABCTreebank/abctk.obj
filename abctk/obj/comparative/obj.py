from collections import defaultdict, deque
from typing import Iterable, TextIO, Optional, Sequence, NamedTuple, List, Match
import dataclasses
from dataclasses import dataclass
import re

import numpy as np

class _CompFeatBracket(NamedTuple):
    """
    Represents a span bracket in comparative annotations.
    """
    label: str
    """
    The label of this bracket.
    """

    is_start: bool
    """
    `True` if this is an opening bracket. `False` if it is a closed one.
    """

@dataclass
class CompSpan:
    """
    Represents a comparative feature span.
    """
    start: int
    end: int
    label: str

    def __str__(self):
        return f'({self.start}-{self.end}){self.label}'

_LABEL_WEIGHT = defaultdict(lambda: 0)
_LABEL_WEIGHT["root"] = 100

def _mod_token(
    token: str,
    feats: Iterable[_CompFeatBracket]
) -> str:
    for label, is_start in feats:
        if is_start:
            token = f"[{token}"
        else:
            token = f"{token}]{label}"
    return token

def linearize_annotations(
    tokens: Iterable[str],
    comp: Optional[Iterable[CompSpan]],
) -> str:
    """
    linearlize a comparative NER annotation.

    Examples
    --------
    >>> linearlize_annotation(
    ...     tokens = ["太郎", "花子", "より", "賢い"],
    ...     comp = [
                {"start": 0, "end": 4, "label": "root"},
                {"start": 1, "end": 2, "label": "prej"},
            ],
    ... )
    "[太郎 [花子 より]prej 賢い]root"
    """
    feats_pos: defaultdict[int, deque[_CompFeatBracket]] = defaultdict(deque)

    if comp:
        for feat in sorted(
            comp,
            key = lambda x: _LABEL_WEIGHT[x.label]
        ):
            feats_pos[feat.start].append(
                _CompFeatBracket(feat.label, True)
            )
            feats_pos[feat.end - 1].append( 
                _CompFeatBracket(feat.label, False)
            )

    return ' '.join(
        _mod_token(t, feats_pos[idx])
        for idx, t in enumerate(tokens)
    )

_RE_COMMENT = re.compile(r"^//\s*(?P<comment>.*)")
_RE_TOKEN_BR_CLOSE = re.compile(r"^(?P<token>[^\]]+)\](?P<feat>[a-z0-9]+)(?P<rem>.*)$")

@dataclass
class CompRecord:
    ID: str
    tokens: Sequence[str]
    comp: Sequence[CompSpan]
    comments: Sequence[str] = dataclasses.field(default_factory=list)
    ID_v1: Optional[str] = None
    
    def to_brackets(self) -> str:
        return linearize_annotations(self.tokens, self.comp)
    
    def to_brackets_full(self) -> str:
        comments_printed = "\n".join(self.comments)
        return f"{self.ID} {self.to_brackets()}\n{comments_printed}"
    
    @classmethod
    def from_brackets(
        cls, 
        line: str, 
        ID: str = "<NOT GIVEN>", 
        comments: Optional[Sequence[str]] = None,
        ID_v1: Optional[str] = None,
    ):
        """
        Parse a linearized comparative NER annotation.

        Examples
        --------
        >>> CompRecord.from_brackets(
        ...     "[太郎 [花子 より]prej 賢い]root",
        ...     ID = "test_11",
        ... )
        { 
            "ID": "test_11",
            "tokens": ["太郎", "花子", "より", "賢い"],
            "comp": {
                "start": 1, "end": 2, "label": "prej"}
                "start": 0, "end": 4, "label": "root"}
            }
        }
        """
        tokens = line.split(" ")

        res_token_list = []
        comp_span_list: list[CompSpan] = []
        stack_br_open: list[int] = []

        for i, token in enumerate(tokens):
            while token.startswith("["):
                stack_br_open.append(i)
                token = token[1:]

            while (match := _RE_TOKEN_BR_CLOSE.search(token)):
                start = stack_br_open.pop()
                comp_span_list.append(
                    CompSpan(start = start, end = i + 1, label = match.group("feat"))
                )
                token = match.group("token") + match.group("rem")

            res_token_list.append(token)

        return cls(
            ID = ID,
            tokens = res_token_list,
            comp = comp_span_list,
            comments = comments or [],
            ID_v1 = ID_v1,
        )

    @classmethod
    def from_brackets_with_ID(
        cls, 
        line: str, 
        comments: Optional[Sequence[str]] = None,
        ID_v1: Optional[str] = None
    ):
        """
        
        Examples
        --------
        >>> delinearize_ID_annotations(
        ...     "test_11 [太郎 [花子 より]prej 賢い]root",
        ... )
        { 
            "ID": "test_11",
            "tokens": ["太郎", "花子", "より", "賢い"],
            "comp": {
                "start": 1, "end": 2, "label": "prej"}
                "start": 0, "end": 4, "label": "root"}
            }
        }
        """
        line_split = line.split(" ", 1)
        ID, text = line_split[0], line_split[1]
        return cls.from_brackets(
            text, 
            ID = ID, 
            comments = comments, 
            ID_v1 = ID_v1
        )
    
    @classmethod
    def read_bracket_annotation_file(cls, stream: TextIO):
        comment_reservoir: List[str] = []
        record_reservoir: Optional[cls] = None

        for line in map(str.strip, stream):
            match_comment: Match[str] | None = _RE_COMMENT.match(line)

            if not line:
                continue
            elif match_comment:
                comment_reservoir.append(match_comment.group("comment"))
            else:
                # generate the previous record
                if record_reservoir:
                    if comment_reservoir:
                        record_reservoir.comments = [com for com in comment_reservoir]
                        comment_reservoir.clear()

                    yield record_reservoir
                
                record_reservoir = cls.from_brackets_with_ID(line)

        if record_reservoir:
            if comment_reservoir:
                record_reservoir.comments = [com for com in comment_reservoir]
                comment_reservoir.clear()

            yield record_reservoir

    @classmethod
    def chomp(
        cls,
        tokens_subworeded: Sequence[str],
        comp: Sequence[CompSpan],
        ID: str = "<NOT GIVEN>",
        comments: Optional[Sequence[str]] = None, 
        ID_v1: Optional[str] = None
    ):
        token_end_index = np.zeros(
            (len(tokens_subworeded), ),
            dtype = np.int_
        )

        cls_offset: int = 0
        pos_word: int = -1
        for pos_subword, token_subworded in enumerate(tokens_subworeded):
            if token_subworded in ("[SEP]", "[PAD]") :
                # reach the end
                # register the end of the last word
                token_end_index[pos_word] = pos_subword

                # end the loop
                break
            elif token_subworded == "[CLS]":
                cls_offset = pos_subword + 1
                continue
            elif token_subworded.startswith("##"):
                continue
            else:
                # register the end of the previous word
                if pos_word >= 0:
                    token_end_index[pos_word] = pos_subword

                # incr the word pointer
                pos_word += 1
        
        comp_realigned = [
            CompSpan(
                start = (
                    token_end_index[span.start - 1]
                    if span.start > 0 else cls_offset
                ),
                end = token_end_index[span.end - 1],
                label = span.label,
            ) for span in comp
        ]

        return cls(
            ID = ID,
            tokens = tokens_subworeded,
            comp = comp_realigned,
            comments = comments or [],
            ID_v1 = ID_v1,
        )

    def dice(self):
        char_end_index = np.zeros( 
            (len(self.tokens), ),
            dtype= np.int_
        )

        tokens_diced: List[str] = []

        for i, token in enumerate(self.tokens):
            token = token.strip("##")
            tokens_diced.extend(token)

            char_end_index[i] = len(token)
        char_end_index = np.cumsum(char_end_index)

        comp_realigned = [
            CompSpan(
                start = (
                    char_end_index[span.start - 1]
                    if span.start > 0 else 0
                ),
                end = char_end_index[span.end - 1],
                label = span.label,
            ) for span in self.comp
        ]

        return self.__class__(
            ID = self.ID,
            tokens = tokens_diced,
            comp = comp_realigned,
            comments = self.comments,
            ID_v1 = self.ID_v1
        )
