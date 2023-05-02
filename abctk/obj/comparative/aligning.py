from collections import defaultdict, Counter
import typing
from typing import Iterable, TypedDict, Sequence, Tuple, NamedTuple, DefaultDict, List, Dict
from collections.abc import Mapping
from enum import IntEnum, auto

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment

from abctk.obj.comparative.obj import *

class MatchSpanResult(IntEnum):
    CORRECT = auto()
    SPURIOUS = auto()
    MISSING = auto()
    WRONG_SPAN = auto()
    NON_MATCHING = auto()
    DIFFERENT_STRATA = auto()

def match_CompSpan(
    prediction: CompSpan,
    reference: CompSpan,
    strata: Mapping[str, int] = defaultdict(lambda: 0, root = 1),
):
    eq_span = reference.start == prediction.start and reference.end == prediction.end

    crossing_span = (
        (reference.start <= prediction.start < reference.end)
        or (prediction.start <= reference.start < prediction.end)
    )

    eq_label = reference.label == prediction.label
    eq_strata = strata[reference.label] == strata[prediction.label]

    results = (eq_span, crossing_span, eq_label, eq_strata)

    if eq_strata:
        if eq_span and eq_label:
            return MatchSpanResult.CORRECT
        elif crossing_span and eq_label:
            return MatchSpanResult.WRONG_SPAN
        else:
            return MatchSpanResult.NON_MATCHING
    else:
        return MatchSpanResult.DIFFERENT_STRATA

PENALTY = {
    MatchSpanResult.CORRECT: 0,
    MatchSpanResult.SPURIOUS: 2,
    MatchSpanResult.MISSING: 2,
    MatchSpanResult.WRONG_SPAN: 1,
    MatchSpanResult.NON_MATCHING: 65536,
    MatchSpanResult.DIFFERENT_STRATA: 65536,
}
class AlignResult(NamedTuple):
    map_pred_to_ref: Tuple[Tuple[int, MatchSpanResult]]
    map_ref_to_pred: Tuple[Tuple[int, MatchSpanResult]]

def print_AlignResult(
    prediction: Sequence[CompSpan],
    reference: Sequence[CompSpan],
    alignment: AlignResult
) -> List[str]:
    printed = []
    for p, pred_span in enumerate(prediction):
        ref, match_result = alignment.map_pred_to_ref[p]
        if match_result == MatchSpanResult.CORRECT:
            printed.append(
                f"ref: {reference[ref]}     ↔ pred: {pred_span}      ✓"
            )
        elif match_result == MatchSpanResult.SPURIOUS:
            printed.append(
                f"ref: NONE      ↔ pred: {pred_span}"
            )
        else:
            printed.append(
                f"ref: {reference[ref]}      ↔ pred: {pred_span}      {match_result.name}"
            )
    for r, ref_span in enumerate(reference):
        pred, match_result = alignment.map_ref_to_pred[r]
        if match_result == MatchSpanResult.MISSING:
            printed.append(
                f"ref: {ref_span}      ↔ pred: None"
            )

    return printed

def align_comp_annotations(
    predictions: Sequence[CompSpan],
    references: Sequence[CompSpan],
    strata: Mapping[str, int] = defaultdict(lambda: 0, root = 1),
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
                strata = strata,
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

    map_r2p: List[Tuple[int, MatchSpanResult]] = [(-1, MatchSpanResult.MISSING)] * size_ref
    for p, r in zip(opt_pred, opt_ref):
        if r < size_ref and p < size_pred:
            map_r2p[r] = (p, judgments[p][r])
        # else:
            # MISSING in prediction
            # use the filled value

    return AlignResult(
        map_pred_to_ref = map_p2r,
        map_ref_to_pred = tuple(map_r2p),
    )

class Metrics(TypedDict):
    scores_spanwise: Dict[str, Dict[str, float]]
    F1_strict_average: float
    F1_partial_average: float
    alignments: List[AlignResult]

def calc_prediction_metrics(
    predictions: Iterable[Sequence[CompSpan]],
    references: Iterable[Sequence[CompSpan]],
    strata: Mapping[str, int] = defaultdict(lambda: 0, root = 1),
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
        align_res = align_comp_annotations(
            pred, ref,
            strata = strata,
        ) 
        result_alignments.append(align_res)

        for p, pred_span in enumerate(pred):
            _, pred_ref_jud = align_res.map_pred_to_ref[p]
            result_bin[pred_span.label][pred_ref_jud] += 1

        for r, ref_span in enumerate(ref):
            pred_index, ref_pred_jud = align_res.map_ref_to_pred[r]
            if pred_index < 0:
                # count SPURIOUS
                result_bin[ref_span.label][ref_pred_jud] += 1

    # ------
    # Calc spanwise scores
    # ------
    res_per_label: dict[str, dict[str, float]] = {}
    for label, ct in result_bin.items():
        possible_entries = (
            ct[MatchSpanResult.CORRECT]
            + ct[MatchSpanResult.WRONG_SPAN]
            + ct[MatchSpanResult.MISSING]
        )

        actual_entries = (
            ct[MatchSpanResult.CORRECT]
            + ct[MatchSpanResult.WRONG_SPAN]
            + ct[MatchSpanResult.SPURIOUS]
        )

        precision_strict = (
            ct[MatchSpanResult.CORRECT] / actual_entries
        ) if actual_entries else np.inf
        recall_strict = (
            ct[MatchSpanResult.CORRECT] / possible_entries
        ) if possible_entries else np.inf
        F1_strict = (
            2 * precision_strict * recall_strict
            / (precision_strict + recall_strict)
        ) if (precision_strict + recall_strict) else np.inf

        correct_with_partial = (
            ct[MatchSpanResult.CORRECT]
            + 0.5 * ct[MatchSpanResult.WRONG_SPAN]
        )
        precision_partial = (
            correct_with_partial / actual_entries
        ) if actual_entries else np.inf
        recall_partial = (
            correct_with_partial / possible_entries
        ) if possible_entries else np.inf
        F1_partial = (
            2 * precision_partial * recall_partial
            / (precision_partial + recall_partial)
        ) if (precision_partial + recall_partial) else np.inf

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
        if not np.isnan(res["F1_strict"])
    )
    F1_partial_list = tuple(
        res["F1_partial"]
        for res in res_per_label.values()
        if not np.isnan(res["F1_partial"])
    )
    return {
        "scores_spanwise": res_per_label,
        "F1_strict_average": sum(F1_strict_list) / len(F1_strict_list),
        "F1_partial_average": sum(F1_partial_list) / len(F1_partial_list),
        "alignments": result_alignments,
    }
