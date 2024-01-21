from collections import defaultdict, deque
from typing import ClassVar, Iterable, TextIO, Optional, Sequence, NamedTuple, List, Match
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

_RE_COMMENT = re.compile(r"^#\s*(?P<comment>.*)")
_RE_ID_V1 = re.compile(r"^ID_v1:\s*(?P<idv1>.*)")
_RE_TOKEN_BR_CLOSE = re.compile(r"^(?P<token>[^\]]+)\](?P<feat>[a-z0-9]+)(?P<rem>.*)$")

@dataclass
class CompRecord:
    """
    Represents a record of comparative annotations.

    Annotations can be serialized in two formats: 
    * The dictionary format: words and span annotations are separated as different attributes.

      Example: 
      ```
      { 
            "ID": "test_11",
            "tokens": ["太郎", "花子", "より", "賢い"],
            "comp": {
                "start": 1, "end": 2, "label": "prej"}
                "start": 0, "end": 4, "label": "root"}
      }
      ```

      This is the primary format in the sense that it is isomorphic to this dataclass.

    * The bracket format: words and span annotations are mingled in a single line. Words are separated by spaces and spans are brackets surrounding words within.

      Example:
      ```
      [太郎 [花子 より]prej 賢い]root
      ```

    Annotations in different formats can be converted into one another's.
    * From the dictionary to the bracket format:

      * `linearize_annotations` (class method)
      * `to_brackets`, `to_brackets_full` (instance methods)

    * From the bracket to the dictionary format:
    
      * `from_brackets`, `from_brackets_with_ID` (class methods)
      * `read_bracket_annotation_file` (class method, targeting streams)
    """
    
    ID: str
    """
    The record ID.
    """

    tokens: Sequence[str]
    """
    List of words.
    """

    comp: Sequence[CompSpan]
    """
    Comparative annotations.
    """
    
    comments: Sequence[str] = dataclasses.field(default_factory=list)
    """
    List of annotation comments.
    """
    
    ID_v1: Optional[str] = None
    """
    The previous ID.
    """

    yaml_tag: ClassVar[str] = "!CompRecord"
    
    @classmethod
    def to_yaml(cls, representer, node):
        # https://stackoverflow.com/a/66477701
        from ruamel.yaml.comments import CommentedSeq, CommentedMap

        mp = dataclasses.asdict(node)

        seq_tokens = CommentedSeq(mp["tokens"])
        seq_tokens.fa.set_flow_style()
        mp["tokens"] = seq_tokens

        def _wrap_comp_span(d: dict):
            res = CommentedMap(**d)
            res.fa.set_flow_style()
            return res
        
        mp["comp"] = CommentedSeq(
            _wrap_comp_span(span)
            for span in mp["comp"]
        )

        return representer.represent_mapping(
            tag = cls.yaml_tag,
            mapping = mp,
        )

    @classmethod
    def from_yaml(cls, constructor, node):
        return cls(**node)

    @classmethod
    def linearize_annotations(
        cls,
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

    def to_brackets(self) -> str:
        return self.linearize_annotations(self.tokens, self.comp)
    
    def to_brackets_with_ID(self) -> str:
        comments_printed = "\n".join(self.comments)
        return f"{self.ID} {self.to_brackets()}\n{comments_printed}"
    def dump_as_txt_bracket(
        self, stream: TextIO,
        show_comments: bool = True,
        show_ID_v1: int = 1,
    ) -> None:
        """
        Dump the record as a bracketed comparative annotation in the TXT format.
        """
        _ = stream.write(f"{self.ID} {self.to_brackets()}\n")
        if show_comments:
            _ = stream.writelines(
                f"# {c}\n" for c in self.comments
            )
        # else:
        #     pass

        if show_ID_v1 == 0 and self.ID_v1:
            _ = stream.write(f"ID_v1: {self.ID_v1}\n")
        elif show_ID_v1 > 0:
            _ = stream.write(f"ID_v1: {self.ID_v1 or ''}\n")
        # else:
        #     pass

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
        >>> r = CompRecord.from_brackets(
        ...     "[太郎 [花子 より]prej 賢い]root",
        ...     ID = "test_11",
        ... )
        >>> r._asdict()
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
        Parse a linearized comparative NER annotation with an ID attached to the beginning of the line.
        
        Examples
        --------
        >>> r = CompRecord.from_brackets_with_ID(
        ...     "test_11 [太郎 [花子 より]prej 賢い]root",
        ... )
        >>> r._asdict()
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
        """
        Parse a text stream of bracketed comparative annotations.
        """
        comment_reservoir: List[str] = []
        ID_v1_reservoir: Optional[str] = None
        record_reservoir: Optional[cls] = None

        for line in map(str.strip, stream):
            match_ID_v1: Match[str] | None = _RE_ID_V1.match(line)
            match_comment: Match[str] | None = _RE_COMMENT.match(line)

            if not line:
                continue
            elif match_ID_v1:
                ID_v1_reservoir = match_ID_v1.group("idv1")
            elif match_comment:
                comment_reservoir.append(match_comment.group("comment"))
            else:
                # generate the previous record
                if record_reservoir:
                    if comment_reservoir:
                        record_reservoir.comments = list(comment_reservoir)
                        comment_reservoir.clear()

                    record_reservoir.ID_v1 = ID_v1_reservoir
                    ID_v1_reservoir = None

                    yield record_reservoir

                record_reservoir = cls.from_brackets_with_ID(
                    line,
                )

        # generate the last record
        if record_reservoir:
            if comment_reservoir:
                record_reservoir.comments = list(comment_reservoir)
                comment_reservoir.clear()

            record_reservoir.ID_v1 = ID_v1_reservoir
            ID_v1_reservoir = None

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
        """
        Apply comparative spans to a sentence tokenized by WordPiece [1]_ to create a `CompRecord`.

        Arguments
        ---------
        tokens_subworded
            A sentence tokenized by tokenizers based on WordPiece.
            Special tokens such as `[CLS]`, `[SEP]`, and `[PAD]` are admitted.

            This function will stop at `[SEP]`, and `[PAD]`.
        
        comp
            A collection of comparative spans. The indices therein do not take subwords into consideration.

        References
        ----------
        .. [1] https://github.com/macmillancontentscience/wordpiece
        """

        # An array to keep records of the correspondence between the indices of the subworded tokens and those of the virtual (non-subworded) tokens used in `comp`.
        # For a non-subworded token whose index is `p_token`, `token_end_index[p_token]` is the next index of the last subworded token included by the `p_token`-th token.
        #
        # Example:
        #   subworded tokens: [0"[CLS]" 1"太郎" 2"花" 3"##子" 4"に"　５"[SEP]"]
        #   non-subworded tokens: [0"太郎" 1"花子" 2"に"]
        #   correspondence: 
        #       0"太郎" ↦ [1, 2)
        #       1"花子" ↦ [2, 4)
        #       2"に" ↦ [4, 5)
        #   token_end_index: [2, 4, 5, ...]

        # Note: 
        # For a non-subworded token whose index is `p_token`,
        # The beginning subworded index is equivalent to the last subworded index of the previous non-subworded token.
        # That is, `token_begin_index[p_token] = token_end_index[p_token - 1]`.
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
        
        # Realign the indices in the comparative spans to the subworded ones.
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
        """
        Dice the tokens into single characters. 
        The indices in the comparative spans will be adjusted.

        Notes
        -----
        Instance recreated.
        """
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
