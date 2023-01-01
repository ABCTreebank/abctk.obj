from abc import abstractmethod
from collections import deque
import dataclasses as d
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import re
from typing import FrozenSet, Generic, Dict, Optional, Tuple, Union, Iterator, Deque, Set, Sequence, Any, Callable, TypeVar

import lark

_CACHE_SIZE = 512

class DepMk(Enum):
    """
    An inventory of dependency markings used in the Keyaki-to-ABC conversion. 
    """

    NONE = "none"
    """
    The default, empty marking.
    """

    HEAD = "h"
    """
    Stands for a phrase head.
    """

    COMPLEMENT = "c"
    """
    Stands for a complement in a phrase.
    """

    ADJUNCT = "a"
    """
    Stands for an adjunct of a phrase.

    Notes
    -----
    The conversion process will
    try to make adjuncts marked by this
    modify the smallest core of the head.
    As a result, the adjuncts are made predicated of the 
    head without affecting its complements.
    This is intended to be an approximation to the raising effect.
    """

    ADJUNCT_CONTROL = "ac"
    """
    The adjunct-control marking.
    Nearly same as `ADJUNCT`, except that 
    the head will eventually get modified along with its subject(s).
    The "ac"-marked adjuncts will end up controlling this subject / these subjects.  
    """
# === END CLASS ===

PlainCat = str
KeyakiCat = str

X_co = TypeVar("X_co", covariant = True)

_annot_cat_basic_matcher = re.compile(r"^(?P<cat>[^#]*)(?P<feats>#.*)?$")
_annot_feat_matcher = re.compile(r"#(?P<key>[^=]+)=(?P<val>[^#]*)")

@dataclass(
    frozen = True,
)
class Annot(Generic[X_co]):
    cat: X_co
    feats: Dict[str, Any] = d.field(default_factory = dict)
    pprinter_cat: Callable[[X_co], str] = str

    # __slots__ = ["cat", "feats", "pprinter_cat"]

    def pprint(
        self, 
        pprinter_cat: Optional[
            Callable[[Any], str]
        ] = None
    ):
        """
        Prettyprint an ABC Treebank feature bundle.

        Examples
        --------
        >>> label = Annot.parse("<NP/NP>#role=h#deriv=leave")
        >>> label.annot_feat_pprint()
        '<NP/NP>#role=h#deriv=leave'

        """
        if "role" in self.feats:
            role = f"#role={self.feats['role'].value}"
        else:
            role = ""
        # === END IF===

        others = "".join(
            f"#{k}={v}"
            for k, v in self.feats.items()
            if k not in ["role"]
        )
        
        if pprinter_cat:
            cat = pprinter_cat(self.cat)
        else:
            cat = self.pprinter_cat(self.cat)

        return f"{cat}{role}{others}"

    @classmethod
    def parse(
        cls,
        source: str,
        parser_cat: Callable[[str], Any] = (lambda x: x),
        pprinter_cat: Callable[[Any], str] = str,
    ):
        """
        Parse a tree node label with ABC annotations.

        Arguments
        ---------
        source : str
            A tree node label.
        parser_cat
        pprinter_cat

        Examples
        --------
        >>> c = parse_annot("<NP/NP>#role=h#deriv=leave")
        >>> c.cat
        '<NP/NP>'
        >>> isinstance(c.cat, str)
        True
        >>> c.feats
        {'role': <DepMk.HEAD: 'h'>, 'deriv': 'leave'})
        """
        match = _annot_cat_basic_matcher.match(source)

        if match is None: raise ValueError

        feats = {
            m.group("key"):m.group("val")
            for m in _annot_feat_matcher.finditer(match.group("feats") or "")
        }

        if "role" in feats:
            feats["role"] = DepMk(feats["role"])
        else:
            feats["role"] = DepMk.NONE
        # === END IF ==

        return cls(
            parser_cat(match.group("cat")), 
            feats,
            pprinter_cat,
        )

    def __str__(self):
        return self.pprint()

# ============
# Classes for ABC categories
# ============

class ABCCatFunctorMode(Enum):
    """
    The enumeration of the types in ABC categories.
    """

    LEFT = "L"
    """
    The left functor `\\`.
    """

    RIGHT = "R"
    """
    The right functor '/'.
    """

    VERT = "V"
    """
    The vertical functor '|' of TLCG.
    """

    def __invert__(self):
        if self == self.LEFT:
            return self.RIGHT
        elif self == self.RIGHT:
            return self.LEFT
        else:
            return self

# === END CLASS ===

class ABCCatReprMode(Enum):
    """
    Styles of representation of categories.
    """

    TRADITIONAL = 0
    """
    The traditional way of representing categories.
    Adopted by Combinatorial Categorial Grammar.

    Examples
    --------
    - `<S/NP>` stands for an `S` wanting an `NP` to its right.
    - `<S\\NP>` is an `S` whose `NP` argument to its left is missing.
    """
    
    TLCG = 1
    """
    An iconic way of representing categories.
    Adopted by Type-logical Categorial Grammar and the ABC Treebank.
    This is the default representation of this package.
    
    Examples
    --------
    - `<S/NP>` stands for an `S` wanting an `NP` to its right.
    - `<NP\\S>` is an `S` whose `NP` argument to its left is missing.
    - `<Sm\\NP>` is a predicate which bears an `m` feature. The categorial features are not marked in a special way.
    """
    
    DEPCCG = 2
    """
    The style that can be read by depccg.
    Parentheses are used instead of angle brackets.

    Examples
    --------
    - `(S/NP)` stands for an `S` wanting an `NP` to its right.
    - `(S\\NP)` is an `S` whose `NP` argument to its left is missing.
    - `(S[m]\\NP)` is a predicate which bears an `m` feature.
    """

    CCG2LAMBDA = 3
    """
    The style compatible with ccg2lambda.

    Examples
    --------
    - `<S/NP>` stands for an `S` wanting an `NP` to its right.
    - `<S\\NP>` is an `S` whose `NP` argument to its left is missing.
    - `<S[m=true]\\NP>` is a predicate which bears an `m` feature.
    """

_re_elimtype = re.compile(
    r"^(?P<dir>[<>|])(B(?P<level>[0-9]+))?$"
)

@dataclass(
    frozen = True,
)
class ElimType:
    """
    Representing details of simplification of ABC categories.

    The string representation `__str__` is compatible with the Jigg-ccg2lambda format. 
    """

    func_mode: ABCCatFunctorMode
    """
    The used rule in the simplification.
    """

    level: int
    """
    The depth of functional compoisition.
    """

    __slots__ = ["func_mode", "level"]

    @classmethod
    def is_compatible_repr(cls, input: str) -> bool:
        return (
            input in ("", "none")
            or bool(_re_elimtype.match(input))
        )

    @classmethod
    def is_repr(cls, input: str) -> bool:
        return bool(_re_elimtype.match(input))

    @classmethod
    def maybe_parse(cls, input: str):
        match = _re_elimtype.match(input)

        if match:
            d = match.groupdict()

            level = int(d["level"]) if "level" in d else 0

            dir_raw = d["dir"]
            
            if dir_raw == "<":
                direct = ABCCatFunctorMode.LEFT
            elif dir_raw == ">":
                direct = ABCCatFunctorMode.RIGHT
            elif dir_raw == "|":
                direct = ABCCatFunctorMode.VERT
            else:
                raise ValueError
            
            return cls(func_mode = direct, level = level)
        else:
            return input

    def __str__(self):
        level_str = f"B{self.level}" if self.level > 0 else ""

        direct = self.func_mode

        if direct == ABCCatFunctorMode.LEFT:
            return f"<{level_str}"
        elif direct == ABCCatFunctorMode.RIGHT:
            return f">{level_str}"
        elif direct == ABCCatFunctorMode.VERT:
            return f"|{level_str}"
        else:
            raise ValueError

class ABCCatParseError(Exception):
    """
    The exception class for ABC category parsing.
    """

    input: str
    """
    The original input that one tried to parse.
    """

    def __init__(self, input: str):
        self.input = input
        super().__init__(f'"{input}" cannot be parsed')

ABCSimplifyRes = Tuple["ABCCat", ElimType]

ABCCatReady = Union[str, "ABCCat"]
"""
The union of the types that is interpretable as ABC categories.
"""

class ABCCat():
    """
    The abstract base class representing ABC categories.

    Notes
    -----
    All instances of all subclasses of this are expected to be identifiable by `==`, hashable, and frozen (unchangeable; all operations create new instances).

    All category-related arguments of the methods are expected to accept both `ABCCat` and any other types that can be interpretable as `ABCCat`, instances of the latter types begin implicitly parsed.
    This feature will enable a readable and succint coding and manupilation of `ABCCat` objects.
    """

    @lru_cache(maxsize = _CACHE_SIZE)
    def adjunct(self, func_mode: ABCCatFunctorMode) -> "ABCCatFunctor":
        """
        Make a self adjunction from a category.

        Examples
        --------
        >>> ABCCat.p("NP").adjunct(ABCCatFunctorMode.RIGHT).pprint()
        '<NP/NP>'
        
        >>> ABCCat.p("NP").adjunct(ABCCatFunctorMode.LEFT).pprint()
        '<NP\\\\NP>'

        >>> ABCCat.p("NP").adjunct(ABCCatFunctorMode.VERT).pprint()
        '<NP|NP>'
        """
        return ABCCatFunctor(
            func_mode = func_mode,
            ant = self,
            conseq = self,
        )

    def adj_l(self):
        """
        Make a self left adjunction from a category.
        This is no more than an alias of `ABCCat.adjunct(cat, ABCCatFunctorMode.LEFT)`.
        """

        return self.adjunct(ABCCatFunctorMode.LEFT)

    def adj_r(self):
        """
        Make a self right adjunction from a category.
        This is no more than an alias of `ABCCat.adjunct(cat, ABCCatFunctorMode.RIGHT)`.
        """
        return self.adjunct(ABCCatFunctorMode.RIGHT)

    def adj_v(self):
        """
        Make a self vertical adjunction from a category.
        This is no more than an alias of `ABCCat.adjunct(cat, ABCCatFunctorMode.VERT)`.
        """
        return self.adjunct(ABCCatFunctorMode.VERT)

    @lru_cache(maxsize = _CACHE_SIZE)
    def v(self, ant: ABCCatReady) -> "ABCCatFunctor":
        """
        An iconic method to create a vertical functor cateogry.
        To the argument comes the antecedent (viz. the `B` in `A|B`).

        Notes
        -----
        The (antecedent) argument can be of any types in `ABCCatReady`.
        It is not necessary to convert an `str` antecedent into an `ABCCat` beforehand.

        The `|` (bitwise OR) operator is also available as an alias of this function.

        See also
        --------
        ABCCat.l:
            The same method for left functors.

        Examples
        --------
        >>> ABCCat.p("NP").v("Scomp").pprint()
        '<NP|Scomp>'

        >>> (ABCCat.p("NP") | "Scomp").pprint()
        '<NP|Scomp>'
        """
        return ABCCatFunctor(
            func_mode = ABCCatFunctorMode.VERT,
            ant = self.p(ant),
            conseq = self,
        )

    def __or__(self, other: ABCCatReady):
        return self.v(other)

    @lru_cache(maxsize = _CACHE_SIZE)
    def r(self, ant: ABCCatReady) -> "ABCCatFunctor":
        """
        An iconic method to create a right functor cateogry.
        To the argument comes the antecedent (viz. the `B` in `A/B`).

        Notes
        -----
        The (antecedent) argument can be of any types in `ABCCatReady`.
        It is not necessary to convert an `str` antecedent into an `ABCCat` beforehand.

        The `/` (true division) operator is also available as an alias of this function.

        See also
        --------
        ABCCat.l:
            The same method for left functors.

        Examples
        --------
        >>> ABCCat.p("NP").r("Scomp").pprint()
        '<NP/Scomp>'

        >>> (ABCCat.p("NP") / "Scomp").pprint()
        '<NP/Scomp>'
        """
        return ABCCatFunctor(
            func_mode = ABCCatFunctorMode.RIGHT,
            ant = self.p(ant),
            conseq = self,
        )
    
    def __truediv__(self, others):
        return self.r(others)

    @lru_cache(maxsize = _CACHE_SIZE)
    def l(self, conseq: ABCCatReady) -> "ABCCatFunctor":
        """
        An iconic method to create a left functor cateogry.
        To the argument comes the consequence (viz. the `B` in `B\\A`).

        Notes
        -----
        The (antecedent) argument can be of any types in `ABCCatReady`.
        It is not necessary to convert an `str` antecedent into an `ABCCat` beforehand.

        For left functors, which has no baskslash counterpart that is eligible for an Python binary operator (like the `/` for right functors),
            a workaround would be combining `/` with the direction inversion `~`.

        Examples
        --------
        >>> ABCCat.p("NP").l("S").pprint()
        '<NP\\\\S>'

        >>> (~(ABCCat.p("S") / "NP")).pprint()
        '<NP\\\\S>'
        """
        return ABCCatFunctor(
            func_mode = ABCCatFunctorMode.LEFT,
            ant = self,
            conseq = self.p(conseq),
        )

    @abstractmethod
    def invert_dir(self) -> "ABCCat": ...

    def __invert__(self):
        return self.invert_dir()

    @classmethod
    def simplify(
        cls,
        left: ABCCatReady,
        right: ABCCatReady,
    ) -> Iterator[ABCSimplifyRes]:
        """
        Simplify a pair of ABC cateogires, using functor elimination rules
            and functor composition rules.
        All possible results are iterated, duplication not being eliminated.

        Arguments
        ---------
        left: ABCCatReady
        right: ABCCatReady

        Notes
        -----
        It yields nothing if no viable simplification is found.

        Yields
        -------
        cat: ABCCat
            The resulting category.
        res: ElimType
            The details of the simplification process.
        """
        
        left_parsed = ABCCat.p(left)
        right_parsed = ABCCat.p(right)

        queue: Deque[
            Tuple[
                "ABCCatFunctor",
                ABCCat,
                bool, # ant_left
                Callable[[ABCCat, ElimType], ABCSimplifyRes]
                ]
        ] = deque()
        if (
            isinstance(left_parsed, ABCCatFunctor)
            and left_parsed.func_mode in (
                ABCCatFunctorMode.RIGHT,
                ABCCatFunctorMode.VERT,
            )
        ):
            queue.append(
                (left_parsed, right_parsed, False, lambda x, res: (x, res))
            )
        # === END IF ===

        if (
            isinstance(right_parsed, ABCCatFunctor)
            and right_parsed.func_mode in (
                ABCCatFunctorMode.LEFT,
                ABCCatFunctorMode.VERT,
            )
        ):
            queue.append(
                (right_parsed, left_parsed, True, lambda x, res: (x, res))
                # swapped
            )
        # === END IF ===

        while queue:
            f, v, ant_left, decor = queue.popleft()

            cat_maybe = f.reduce_with(v, ant_left)
            if cat_maybe is None:
                # failed
                # try func comp
                if (
                    isinstance(v, ABCCatFunctor)
                    and v.func_mode == f.func_mode 
                    # NOTE: for crossed-composition this condition will be relaxed.
                ):
                    queue.append(
                        (
                            f,
                            v.conseq,
                            ant_left,
                            lambda x, res, _decor = decor, _f = f, _v = v: _decor(
                                ABCCatFunctor(
                                    func_mode = _v.func_mode,
                                    ant = _v.ant,
                                    conseq = x,
                                ),
                                d.replace(res, level = res.level + 1)
                            )
                        )
                    )
                else:
                    # no remedy
                    pass
            else:
                etype = ElimType(
                    func_mode = f.func_mode,
                    level = 0,
                )
                yield decor(cat_maybe, etype)
            # === END IF ===
        # === END WHILE queue ===
    # === END ===

    @classmethod
    @lru_cache(maxsize = _CACHE_SIZE)
    def simplify_exh(
        cls,
        left: ABCCatReady,
        right: ABCCatReady,
    ) -> Set[ABCSimplifyRes]:
        """
        Return all possible ways of functor elimination 
            of a pair of ABC cateogires, using functor elimination rules
            and functor composition rules.
        Duplicating results are eliminted in the same way as a set does.


        Arguments
        ---------
        left: ABCCatReady
        right: ABCCatReady

        Notes
        -----
        `*` is an alias operator that returns
            the result that is first found
            with simplification details omitted.
        Exceptions arise when it fails. 
        
        Yields
        -------
        cat: ABCCat
            The resulting category.
        res: ElimType
            The details of the simplification process.

        Examples
        --------
        >>> results = list(ABCCat.simplify_exh("A/B", "B/C"))
        >>> cat, details = results[0]
        >>> cat.pprint()
        '<A/C>'
        >>> str(details)
        '>'
        """
        
        return set(cls.simplify(left, right))

    def __mul__(self, others):
        # NOTE: This operator must hinge on `simplify_exh` rather than `simplify` for the proper exploitation of cache, which is not available for the latter.
        try:
            cat, _ = next(iter(ABCCat.simplify_exh(self, others)))
        except StopIteration as e:
            raise ValueError("No application can be applied to the two categories.")
        return cat

    @abstractmethod
    def pprint(
        self, 
        mode: ABCCatReprMode = ABCCatReprMode.TLCG
    ) -> str:
        if isinstance(self, ABCCat):
            return self.pprint(mode)
        else:
            return str(self)

    def __repr__(self) -> str:
        return f"<{self.__class__}: {self.pprint()} >"

    @classmethod
    @lru_cache(maxsize = _CACHE_SIZE)
    def parse(
        cls, 
        source: ABCCatReady,
        mode: ABCCatReprMode = ABCCatReprMode.TLCG
    ) -> "ABCCat":
        """ 
        Parse an ABC category.

        Arguments
        ---------
        source: ABCCatReady
            The thing to be parsed. If it is already an `ABCCat`, nothing happens and the method just returns this thing.
        mode: ABCCatReprMode, default: ABCCatReprMode.TLCG
            The linear order of functor categories.

        Returns
        -------
        parsed: ABCCat

        Raises
        ------
        ABCCatParseException
            When parsing fails.

        Examples
        --------
        >>> np: ABCCatBase = ABCCat.parse("NP")
        >>> np.name
        'NP'

        >>> pred: ABCCatFunctor = ABCCat.parse("NP\\\\S")
        >>> pred.pprint()
        '<NP\\\\S>'

        >>> ABCCat.parse(pred) == pred
        True
        """

        if isinstance(source, str):
            parser = _init_parser(mode)
            try:
                return parser.parse(source) # type: ignore
            except Exception as e:
                raise ABCCatParseError(source) from e

        elif isinstance(source, ABCCat):
            return source
        else:
            raise TypeError

    @classmethod
    def p(
        cls,
        source: ABCCatReady,
        mode: ABCCatReprMode = ABCCatReprMode.TLCG,
    ) -> "ABCCat":
        """
        An alias of `ABCCat.parse`.
        """
        return cls.parse(source, mode)

    @classmethod
    @lru_cache(maxsize = _CACHE_SIZE)
    def _lexer(
        cls,
        source: str,
        symbols: str = "/\\<>⊥⊤"
    ) -> Iterator[str]:
        pt_begin = 0
        pt_end = 0
    
        while pt_end < len(source):
            curr_char = source[pt_end]
            
            if curr_char in symbols:
                # yield previous char
                if pt_begin < pt_end:
                    yield source[pt_begin:pt_end]
                    
                yield curr_char
                
                pt_begin = pt_end + 1
                pt_end += 1
            else:
                # make pt_begin trailing
                pt_end += 1
            
        if pt_begin < pt_end:
            yield source[pt_begin:pt_end]
    # === END ===

    @abstractmethod
    def equiv_to(self, other, ignore_feature: bool = False) -> bool:
        ...

    @abstractmethod
    def unify(self, other) -> Optional["ABCCat"]:
        ...

    def __eq__(self, other):
        self.equiv_to(other, ignore_feature = False)

class ABCCatTop(ABCCat, Enum):
    """
    Represents the bottom type in the ABC Treebank.
    """
    TOP = "⊤"

    def pprint(
        self, 
        mode: ABCCatReprMode = ABCCatReprMode.TLCG
    ) -> str:
        return self.value

    def invert_dir(self):
        return self

    def equiv_to(self, other, ignore_feature: bool = False):
        return isinstance(other, ABCCatTop)

    def __eq__(self, other):
        self.equiv_to(other, ignore_feature = False)

    def unify(self, other):
        if self == other:
            return self
        else:
            return None
            
class ABCCatBot(ABCCat, Enum):
    """
    Represents the bottom type in the ABC Treebank.
    """
    BOT = "⊥"

    def pprint(
        self, 
        mode: ABCCatReprMode = ABCCatReprMode.TLCG
    ) -> str:
        return self.value

    def invert_dir(self):
        return self

    def equiv_to(self, other, ignore_feature: bool = False):
        return isinstance(other, ABCCatBot)

    def unify(self, other):
        if self == other:
            return self
        else:
            return None

    def __eq__(self, other):
        return self.equiv_to(other, ignore_feature = False)

    def __hash__(self):
        return hash(self.BOT.value)

_re_ABCCat_feature = re.compile(r"(?P<cat>.*?)(?<=[A-Z])(?P<feat>[a-z][a-z0-9]*)")

@dataclass(frozen = True)
class ABCCatBase(ABCCat):
    """
    Representing atomic ABC categories.
    """

    name: str
    """
    The letter of the atom.
    """

    feats: FrozenSet[Tuple[str, Any]] = d.field(
        default_factory = frozenset
    )

    # __slots__ = ["name", "feats"]

    @lru_cache(maxsize = _CACHE_SIZE)
    def pprint(
        self, 
        mode: ABCCatReprMode = ABCCatReprMode.TLCG
    ) -> str:
        if mode == ABCCatReprMode.TLCG:
            self_feats = self.feats
            self_feats_len = len(self_feats)

            if self_feats_len > 1:
                raise ValueError("TLCG mode does not support more than one subcategorization features")
            elif self_feats_len == 1:
                feat, val = next(iter(self_feats))
                if val != True:
                    raise ValueError("TLCG mode does not supportsubcategorization feature values other than `True`")
                
                return f"{self.name}{feat}"
            else:
                return self.name
        elif mode == ABCCatReprMode.CCG2LAMBDA:
            if self.feats:
                return "{name}[{feats}]".format(
                    name = self.name,
                    feats = ",".join(
                        (feat if val == True else f"{feat}={str(val).lower()}")
                        for feat, val 
                        in self.feats
                    )
                )
            else:
                return self.name
        elif mode == ABCCatReprMode.DEPCCG:
            if self.feats:
                return "{name}[{feats}]".format(
                    name = self.name,
                    feats = ",".join(
                        (feat if val == True else f"{feat}={str(val).lower()}")
                        for feat, val 
                        in self.feats
                        if val == True
                    )
                )
            else:
                return self.name
        else:
            return self.name

    def invert_dir(self):
        return self

    def equiv_to(self, other, ignore_feature: bool = False):
        if isinstance(other, ABCCatBase):
            return (
                self.name == other.name
                and (ignore_feature or self.feats == other.feats)
            )
        elif isinstance(other, ABCCat):
            return False
        else:
            return NotImplemented

    def unify(self, other):
        other = ABCCat.p(other)
        if (
            isinstance(other, ABCCatBase)
            and self.name == other.name
            and (feats := ABCCatBase.unify_feats(self.feats, other.feats))
        ):
            return ABCCatBase(self.name, feats)
        else:
            return None

    @staticmethod
    def unify_feats(
        feat_1: FrozenSet[Tuple[str, str]],
        feat_2: FrozenSet[Tuple[str, str]],
    ):
        feat_union = feat_1.union(feat_2)
        res_dict = {}
        for feat, val in feat_union:
            val_prev = res_dict.get(feat, None)
            if val_prev is None:
                res_dict[feat] = val
            elif val == val_prev:
                pass
            else:
                return None

        return frozenset(res_dict.items())
            
@dataclass(
    frozen = True,
)
class ABCCatFunctor(ABCCat):
    """
    Representing functor categories.
    """

    func_mode: ABCCatFunctorMode
    """
    The mode, or direction, of the functor.
    """

    ant: "ABCCat"
    """
    The antecedent.
    """

    conseq: "ABCCat"
    """
    The consequence.
    """

    __slots__ = ["func_mode", "ant", "conseq"]

    @lru_cache(maxsize = _CACHE_SIZE)
    def pprint(
        self, 
        mode: ABCCatReprMode = ABCCatReprMode.TLCG
    ) -> str:
        if mode == ABCCatReprMode.DEPCCG:
            if self.func_mode == ABCCatFunctorMode.LEFT:
                return f"({self.conseq.pprint(mode)}\\{self.ant.pprint(mode)})"
            elif self.func_mode == ABCCatFunctorMode.RIGHT:
                return f"({self.conseq.pprint(mode)}/{self.ant.pprint(mode)})"
            elif self.func_mode == ABCCatFunctorMode.VERT:
                return f"({self.conseq.pprint(mode)}|{self.ant.pprint(mode)})"
            else:
                raise ValueError
        elif mode == ABCCatReprMode.CCG2LAMBDA:
            if self.func_mode == ABCCatFunctorMode.LEFT:
                return f"({self.ant.pprint(mode)}\\{self.conseq.pprint(mode)})"
            elif self.func_mode == ABCCatFunctorMode.RIGHT:
                return f"({self.conseq.pprint(mode)}/{self.ant.pprint(mode)})"
            elif self.func_mode == ABCCatFunctorMode.VERT:
                return f"({self.conseq.pprint(mode)}|{self.ant.pprint(mode)})"
            else:
                raise ValueError
        else:
            if self.func_mode == ABCCatFunctorMode.LEFT:
                if mode == ABCCatReprMode.TLCG:
                    return f"<{self.ant.pprint(mode)}\\{self.conseq.pprint(mode)}>"
                else:
                    return f"<{self.conseq.pprint(mode)}\\{self.ant.pprint(mode)}>"
            elif self.func_mode == ABCCatFunctorMode.RIGHT:
                return f"<{self.conseq.pprint(mode)}/{self.ant.pprint(mode)}>"
            elif self.func_mode == ABCCatFunctorMode.VERT:
                return f"<{self.conseq.pprint(mode)}|{self.ant.pprint(mode)}>"
            else:
                raise ValueError
    # === END ===

    def invert_dir(self):
        """
        Invert the direction of the functor.

        Notes
        -----
        This method always returns a new instance.

        Examples
        --------
        >>> cat = ABCCat.p("NP\\\\S")
        >>> cat.invert_dir().pprint()
        '<S/NP>'

        >>> cat.invert_dir() == ~cat
        True
        """

        fm_new = ~self.func_mode

        return d.replace(self, func_mode = fm_new)

    def equiv_to(self, other, ignore_feature: bool = False):
        if isinstance(other, ABCCatFunctor):
            return (
                self.func_mode == other.func_mode
                and self.ant.equiv_to(other.ant, ignore_feature)
                and self.conseq.equiv_to(other.conseq, ignore_feature)
            )
        elif isinstance(other, ABCCat):
            return False
        else:
            return NotImplemented

    def __eq__(self, other):
        return self.equiv_to(other, ignore_feature = False)

    def unify(self, other):
        other = ABCCat.p(other)
        if isinstance(other, ABCCatFunctor):
            if (
                self.func_mode == other.func_mode
                and (ant_uni := self.ant.unify(other.ant))
                and (conseq_uni := self.conseq.unify(other.conseq))
            ):
                return ABCCatFunctor(
                    func_mode = self.func_mode,
                    ant = ant_uni,
                    conseq = conseq_uni,
                )
            else:
                return None

    def reduce_with(self, ant: ABCCatReady, ant_left: bool = False) -> Optional[ABCCat]:
        """
        Eliminate the functor with a given antecedent.
        
        Notes
        -----
        Function composition rules are not invoked here.

        Arguments
        ---------
        ant: ABCCatReady
            An antecedent.
        ant_left: bool, default: False
            The position of the ancedecent.
            `True` when it is on the left to the functor. 
        
        Returns
        -------
        conseq: ABCCat or None
            The resulting category. `None` on failure.
        """

        ant_parsed = ABCCat.p(ant)

        if self.ant.equiv_to(ant_parsed, ignore_feature = True):
            if self.func_mode == ABCCatFunctorMode.LEFT and ant_left:
                return self.conseq
            elif self.func_mode == ABCCatFunctorMode.RIGHT and not ant_left:
                return self.conseq
            elif self.func_mode == ABCCatFunctorMode.VERT:
                return self.conseq
            else:
                return None
        else:
            return None

# === END CLASS ===

_cat_grammar_TLCG = r"""
cat: (cat_simple | cat_group)
cat_group: "<" cat_simple ">"
cat_simple: func_left | func_right | func_vert | cat_singleton
cat_singleton: bot | top | atom
func_left: cat "\\" cat
func_right: cat "/" cat
func_vert: cat "|" cat
bot: "⊥"
top: "⊤"
atom: ATOM feats?
feats: featval
featval: FEAT
ATOM: /([A-Z0-9-]+|[^\/\\|<>⊥⊤#\s]+)/
FEAT: /[a-z0-9][a-zA-Z0-9-]*/
"""

_cat_grammar_DEPCCG = r"""
cat: (cat_simple | cat_group)
cat_group: "(" cat_simple ")"
cat_simple: func_left | func_right | func_vert | cat_singleton
cat_singleton: bot | top | atom
func_left: cat "\\" cat
func_right: cat "/" cat
func_vert: cat "|" cat
bot: "⊥"
top: "⊤"
atom: ATOM
ATOM: /[^\/\\|()⊥⊤#\s]+/
"""

_cat_grammar_CCG2LAMBDA = r"""
cat: (cat_simple | cat_group)
cat_group: "(" cat_simple ")"
cat_simple: func_left | func_right | func_vert | cat_singleton
cat_singleton: bot | top | atom
func_left: cat "\\" cat
func_right: cat "/" cat
func_vert: cat "|" cat
bot: "⊥"
top: "⊤"
atom: ATOM feats?
feats: "[" featval ("," featval)* ","? "]"
featval: FEAT ("=" VAL)?
ATOM: /[^\/\\\[\]|()⊥⊤#\s]+/
FEAT: /[^=\]]+/
VAL: /[^\],]+/
"""

class CatParserTransformer(lark.Transformer):
    def __init__(self, mode: ABCCatReprMode = ABCCatReprMode.TLCG):
        self.mode = mode

    def atom(self, args):
        cat: lark.Token
        feats: FrozenSet[Tuple[str, Any]]

        if len(args) == 2:
            cat, feats = args
        elif len(args) == 1:
            cat, = args
            feats = frozenset()
        else:
            raise ValueError

        return ABCCatBase(
            cat.value,
            feats,
        )

    def top(self, args):
        return ABCCatTop.TOP

    def bot(self, args):
        return ABCCatBot.BOT

    def func_vert(self, args):
        return ABCCatFunctor(
            ABCCatFunctorMode.VERT,
            args[1],
            args[0],
        )

    def func_right(self, args):
        return ABCCatFunctor(
            ABCCatFunctorMode.RIGHT,
            args[1],
            args[0],
        )

    def func_left(self, args):
        if self.mode in (ABCCatReprMode.TLCG, ABCCatReprMode.CCG2LAMBDA):
            return ABCCatFunctor(
                ABCCatFunctorMode.LEFT,
                args[0],
                args[1],
            )
        else:
            return ABCCatFunctor(
                ABCCatFunctorMode.LEFT,
                args[1],
                args[0]
            )

    def cat_simple(self, args):
        return args[0]

    def cat_singleton(self, args):
        return args[0]

    def cat_group(self, args):
        return args[0]

    def cat(self, args):
        return args[0]

    def feats(self, args):
        return frozenset(args)

    def featval(self, args: Sequence[lark.Token]):
        if len(args) > 1:
            feat, val = args
            val = val.value
        else:
            feat = args[0]
            val = "true"

        val_lower = val.lower()

        if val_lower == "true":
            val = True
        elif val_lower == "false":
            val = False
        elif val_lower == "none":
            val = None
        
        return (feat.value, val)

_cat_parser: Dict[ABCCatReprMode, lark.Lark] = {}
def _init_parser(mode: ABCCatReprMode):
    global _cat_parser

    if mode not in _cat_parser:
        if mode == ABCCatReprMode.TLCG:
            _cat_parser[mode] = lark.Lark(
                _cat_grammar_TLCG,
                start = "cat",
                parser = "lalr",
                transformer = CatParserTransformer(mode)
            )
        elif mode == ABCCatReprMode.DEPCCG:
            _cat_parser[mode] = lark.Lark(
                _cat_grammar_DEPCCG,
                start = "cat",
                parser = "lalr",
                transformer = CatParserTransformer(mode)
            )
        elif mode == ABCCatReprMode.CCG2LAMBDA:
            _cat_parser[mode] = lark.Lark(
                _cat_grammar_CCG2LAMBDA,
                start = "cat",
                parser = "lalr",
                transformer = CatParserTransformer(mode)
            )
        else:
            raise NotImplementedError

    return _cat_parser[mode]