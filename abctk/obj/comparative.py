from collections import defaultdict, deque, Counter
import typing
from typing import Iterable, TextIO, TypedDict, Optional, Sequence, Tuple, NamedTuple, DefaultDict, List, Dict, Match
from enum import IntEnum, auto
import re

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

class _CompFeatBracket(NamedTuple):
    label: str
    is_start: bool

class CompSpan(TypedDict):
    start: int
    end: int
    label: str

class CompRecord(TypedDict, total = False):
    # TODO: when Python 3.10 is in EOL: 
    # Use NotRequired field instead of total = False
    ID: str
    tokens: Sequence[str]
    comp: Sequence[CompSpan]
    comments: Sequence[str]

_LABEL_WEIGHT = defaultdict(lambda: 0)
_LABEL_WEIGHT["root"] = -100

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
            key = lambda x: _LABEL_WEIGHT[x["label"]]
        ):
            feats_pos[feat["start"]].append(
                _CompFeatBracket(feat["label"], True)
            )
            feats_pos[feat["end"] - 1].append( 
                _CompFeatBracket(feat["label"], False)
            )

    return ' '.join(
        _mod_token(t, feats_pos[idx])
        for idx, t in enumerate(tokens)
    )

def dict_to_bracket(datum: CompRecord) -> str:
    token_bred = linearize_annotations(
        datum["tokens"],
        datum["comp"],
    )
    return f"{datum['ID']} {token_bred}\n"


_RE_TOKEN_BR_CLOSE = re.compile(r"^(?P<token>[^\]]+)\](?P<feat>[a-z0-9]+)(?P<rem>.*)$")
def delinearize_annotations(
    line: str,
    ID: str = "<NOT GIVEN>",
) -> CompRecord:
    """
    Parse a linearized comparative NER annotation.

    Examples
    --------
    >>> delinearize_annotations(
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
    comp_dict_list = []
    stack_br_open: list[int] = []

    for i, token in enumerate(tokens):
        while token.startswith("["):
            stack_br_open.append(i)
            token = token[1:]

        while (match := _RE_TOKEN_BR_CLOSE.search(token)):
            start = stack_br_open.pop()
            comp_dict_list.append(
                {
                    "start": start,
                    "end": i + 1,
                    "label": match.group("feat")
                }
            )
            token = match.group("token") + match.group("rem")

        res_token_list.append(token)
    return {
        "ID": ID,
        "tokens": res_token_list,
        "comp": comp_dict_list,
    }

def delinearize_ID_annotations(line: str) -> CompRecord:
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
    return delinearize_annotations(text, ID = ID)

_RE_COMMENT = re.compile(r"^//\s*(?P<comment>.*)")

def read_bracket_annotation_file(stream: TextIO):
    comment_reservoir: List[str] = []
    record_reservoir: Optional[CompRecord] = None

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
                    record_reservoir["comments"] = [com for com in comment_reservoir]
                    comment_reservoir.clear()

                yield record_reservoir
            
            record_reservoir = delinearize_ID_annotations(line)

    if record_reservoir:
        if comment_reservoir:
            record_reservoir["comments"] = [com for com in comment_reservoir]
            comment_reservoir.clear()

        yield record_reservoir


class MatchSpanResult(IntEnum):
    CORRECT = auto()
    SPURIOUS = auto()
    MISSING = auto()
    WRONG_LABEL = auto()
    WRONG_SPAN = auto()
    WRONG_LABEL_SPAN = auto()

def match_CompSpan(
    prediction: CompSpan,
    reference: CompSpan,
):
    eq_start = reference["start"] == prediction["start"]
    eq_end = reference["end"] == prediction["end"]
    eq_label = reference["label"] == prediction["label"]
    results = (eq_start, eq_end, eq_label)

    if results == (True, True, True):
        return MatchSpanResult.CORRECT
    elif results == (True, True, False):
        return MatchSpanResult.WRONG_LABEL
    elif eq_label:
        return MatchSpanResult.WRONG_SPAN
    else:
        return MatchSpanResult.WRONG_LABEL_SPAN

PENALTY = {
    MatchSpanResult.CORRECT: 0,
    MatchSpanResult.SPURIOUS: 2,
    MatchSpanResult.MISSING: 2,
    MatchSpanResult.WRONG_LABEL: 2,
    MatchSpanResult.WRONG_SPAN: 1,
    MatchSpanResult.WRONG_LABEL_SPAN: 2,
}
class AlignResult(NamedTuple):
    map_pred_to_ref: Tuple[Tuple[int, MatchSpanResult]]
    map_ref_to_pred: Tuple[Tuple[int, MatchSpanResult]]

def align_comp_annotations(
    predictions: Sequence[CompSpan],
    references: Sequence[CompSpan],
) -> AlignResult:
    size_pred: int = len(predictions)
    size_ref: int = len(references)

    if size_pred == 0 and size_ref == 0:
        return AlignResult( 
            map_pred_to_ref = tuple(), 
            map_ref_to_pred = tuple(),
        )
    
    judgments = [
        [
            match_CompSpan(
                prediction = predictions[p],
                reference = references[r],
            )
            for r in range(size_ref)
        ]
        for p in range(size_pred)
    ]
    costs_orig = np.vectorize(
        lambda j: PENALTY[j],
        otypes = [np.int_],
    )(judgments).reshape( (size_pred, size_ref) )

    padding_pred_idle = np.full(
        (size_pred, size_pred),
        MatchSpanResult.SPURIOUS.value,
        dtype = np.int_
    )
    padding_idle_ref = np.full(
        (size_ref, size_ref),
        MatchSpanResult.MISSING.value,
        dtype = np.int_
    )
    padding_idle_idle = np.full(
        (size_ref, size_pred),
        0,
        dtype = np.int_
    )
    costs = np.block(
        [
            [costs_orig,       padding_pred_idle],
            [padding_idle_ref, padding_idle_idle],
        ]
    )

    # -------------
    # Minimize penalties
    # -------------
    opt_pred: NDArray[np.int_]
    opt_ref: NDArray[np.int_]
    opt_pred, opt_ref = linear_sum_assignment(costs)

    map_p2r = tuple(
        (
            (r, judgments[p][r])
            if r < size_ref
            else (-1, MatchSpanResult.SPURIOUS)
        )
        for p, r in zip(opt_pred, opt_ref)
        if p < size_pred
    )

    map_r2p = tuple(
        (
            (p, judgments[p][r])
            if p < size_pred
            else (-1, MatchSpanResult.MISSING)
        )
        for p, r in zip(opt_pred, opt_ref)
        if r < size_ref
    )

    return AlignResult(
        map_pred_to_ref = map_p2r,
        map_ref_to_pred = map_r2p,
    )

class Metrics(TypedDict):
    scores_spanwise: Dict[str, Dict[str, float]]
    F1_strict_average: float
    F1_partial_average: float
    alignments: List[AlignResult]

def calc_prediction_metrics(
    predictions: Iterable[Sequence[CompSpan]],
    references: Iterable[Sequence[CompSpan]],
) -> Metrics:
    """
    Calculate precision and recall scores of 
        comparative model predictions.

    Notes
    -----
    The implementation is basically based on MUC-5 Evaluation Metrics [1]_.
    The review blog article by David S. Batista [2]_ was also helpful.

    .. [1] Nancy Chinchor and Beth Sundheim. 1993. `MUC-5 Evaluation Metrics. <https://aclanthology.org/M93-1007/>`_
        In Fifth Message Understanding Conference (MUC-5): 
            Proceedings of a Conference 
            Held in Baltimore, Maryland, August 25-27, 1993.
    .. [2] David S. Batista. 2018. `Named-Entity evaluation metrics based on entity-level. <https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/>`_ Blog Article.

    """
    result_bin: DefaultDict[
        str, typing.Counter[MatchSpanResult]
    ] = defaultdict(
       lambda: Counter() 
    )
    result_alignments: List[AlignResult] = []

    for pred, ref in zip(predictions, references):
        align_res = align_comp_annotations(pred, ref) 
        result_alignments.append(align_res)

        for p, pred_span in enumerate(pred):
            _, pred_ref_jud = align_res.map_pred_to_ref[p]
            result_bin[pred_span["label"]][pred_ref_jud] += 1

        for r, ref_span in enumerate(ref):
            pred_index, ref_pred_jud = align_res.map_ref_to_pred[r]
            if pred_index < 0:
                result_bin[ref_span["label"]][ref_pred_jud] += 1

    # ------
    # Calc spanwise scores
    # ------
    res_per_label: dict[str, dict[str, float]] = {}
    for label, ct in result_bin.items():
        possible_entries = (
            ct[MatchSpanResult.CORRECT]
            + ct[MatchSpanResult.WRONG_SPAN]
            + ct[MatchSpanResult.WRONG_LABEL]
            + ct[MatchSpanResult.WRONG_LABEL_SPAN]
            + ct[MatchSpanResult.MISSING]
        )

        actual_entries = (
            possible_entries
            - ct[MatchSpanResult.MISSING]
            + ct[MatchSpanResult.SPURIOUS]
        )

        precision_strict = ct[MatchSpanResult.CORRECT] / actual_entries
        recall_strict = ct[MatchSpanResult.CORRECT] / possible_entries
        F1_strict = (
            2 * precision_strict * recall_strict
            / (precision_strict + recall_strict)
        )

        correct_with_partial = (
            ct[MatchSpanResult.CORRECT]
            + 0.5 * ct[MatchSpanResult.WRONG_SPAN]
        )
        precision_partial = correct_with_partial / actual_entries
        recall_partial = correct_with_partial / possible_entries
        F1_partial = (
            2 * precision_partial * recall_partial
            / (precision_partial + recall_partial)
        )

        res_per_label[label] = {
            key.name: value
            for key, value in ct.items()
        }
        res_per_label[label]["possible_entries"] = possible_entries
        res_per_label[label]["actual_entries"] = actual_entries
        
        res_per_label[label]["precision_strict"] = precision_strict
        res_per_label[label]["recall_strict"] = recall_strict
        res_per_label[label]["F1_strict"] = F1_strict

        res_per_label[label]["precision_partial"] = precision_partial
        res_per_label[label]["recall_partial"] = recall_partial
        res_per_label[label]["F1_partial"] = F1_partial
    # === END FOR result_bin
    F1_strict_list = tuple(
        res["F1_strict"]
        for res in res_per_label.values()
    )
    F1_partial_list = tuple(
        res["F1_partial"]
        for res in res_per_label.values()
    )
    return {
        "scores_spanwise": res_per_label,
        "F1_strict_average": sum(F1_strict_list) / len(F1_strict_list),
        "F1_partial_average": sum(F1_partial_list) / len(F1_partial_list),
        "alignments": result_alignments,
    }
