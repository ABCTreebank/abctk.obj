from typing import Tuple, cast
import pytest

from abctk.obj.ABCCat import *

class Test_DepMk:
    def test_parse(self):
        assert DepMk("none") == DepMk.NONE
        assert DepMk("h") == DepMk.HEAD
        assert DepMk("c") == DepMk.COMPLEMENT
        assert DepMk("a") == DepMk.ADJUNCT
        assert DepMk("ac") == DepMk.ADJUNCT_CONTROL

class Test_Annot():
    pprint_items = (
        ("a", "#role=c"),
        ("<a/b/c>", "#deriv=stref#comp-id=1"),
    )
    
    @pytest.mark.parametrize("cat, feat", pprint_items)
    def test_pprint(self, cat, feat):
        string = cat + feat
        parsed = Annot.parse(string)
        parsed_pprinted = parsed.pprint()
        parsed_pprinted_parsed = Annot.parse(parsed_pprinted)

        assert parsed == parsed_pprinted_parsed

class Test_ABCCat:
    items_parse_TLCG = (
        ("⊥", ABCCatBot.BOT),
        ("NP", ABCCatBase("NP")),
        ("<NP\\S>", ABCCatFunctor(
                func_mode = ABCCatFunctorMode.LEFT,
                ant = ABCCatBase("NP"),
                conseq = ABCCatBase("S"),
            )
        ),
        ("<S/NP>", ABCCatFunctor(
                func_mode = ABCCatFunctorMode.RIGHT,
                ant = ABCCatBase("NP"),
                conseq = ABCCatBase("S"),
            )
        ),
        ("S/NP", ABCCatFunctor(
                func_mode = ABCCatFunctorMode.RIGHT,
                ant = ABCCatBase("NP"),
                conseq = ABCCatBase("S"),
            )
        ),
        ("S|NP", ABCCatFunctor(
                func_mode = ABCCatFunctorMode.VERT,
                ant = ABCCatBase("NP"),
                conseq = ABCCatBase("S"),
            )
        ),
    )

    @pytest.mark.parametrize("input, answer", items_parse_TLCG)
    def test_parse_TLCG(self, input, answer):
        assert ABCCat.parse(input) == answer

    items_parse_CCG2LAMBDA = (
        ("NP[q=true]", ABCCatBase(
                name = "NP",
                feats = frozenset([("q", True)]),
            )
        ),
        ("NP[s=false]", ABCCatBase(
                name = "NP",
                feats = frozenset([("s", False)])
            )
        ),
        ("NP", ABCCatBase(
                name = "NP",
                feats = frozenset()
            )
        ),
        ("(NP[s=false]\\VP)", ABCCatFunctor(
                func_mode = ABCCatFunctorMode.LEFT,
                ant = ABCCatBase("NP", frozenset([("s", False)])),
                conseq = ABCCatBase("VP"),
            )
        ),
    )
    @pytest.mark.parametrize("input, answer", items_parse_CCG2LAMBDA)
    def test_parse_CCG2LAMBDA(self, input, answer):
        assert ABCCat.parse(input, ABCCatReprMode.CCG2LAMBDA) == answer

    items_pprint_TLCG = (
        "⊥", "NP", "NP\\S", "<NP\\S>", "<S/NP>",
        "<NP\\<NP\\NP>>",
        "<S/NP>\\<S/NP>",
        "S|NP",
        "S|NP|NP",
        "NPs", "NPs\\S", "<NP\\Srel>", "<Srel/NPx>",
        "<NPq\\<NPq\\NPq>>",
        "<Sq/NP>\\<S/NPq>",
        "S|NPq",
        "S|NPq|NP",
    )
    @pytest.mark.parametrize("input", items_pprint_TLCG)
    def test_parse_pprint_TLCG(self, input):
        parse_1 = ABCCat.parse(input)
        parse_2 = ABCCat.parse(parse_1.pprint())
        assert parse_1 == parse_2

    items_pprint_CCG2LAMBDA = (
        "⊥", "NP", "NP\\S", "(NP\\S)", "(S/NP)",
        "(NP\\(NP\\NP))",
        "(S/NP)\\(S/NP)",
        "S|NP",
        "S|NP|NP",
        "NP[s=true]", "NP[s=true]\\S", "(NP\\S[rel=true])", "(S[rel=true]/NP[x=true])",
        "(NP[q=false]\\(NP[q=true]\\NP[q=false]))",
        "(S[q=true]/NP)\\(S/NP[q=false])",
        "S|NP[q=true]",
        "S|NP[q=false]|NP",
    )
    @pytest.mark.parametrize("input", items_pprint_CCG2LAMBDA)
    def test_parse_pprint_CCG2LAMBDA(self, input):
        parse_1 = ABCCat.parse(input, mode = ABCCatReprMode.CCG2LAMBDA)
        parse_2 = ABCCat.parse(
            parse_1.pprint(mode = ABCCatReprMode.CCG2LAMBDA),
            mode = ABCCatReprMode.CCG2LAMBDA,
        )
        assert parse_1 == parse_2

    def test_r(self):
        assert ABCCat.p("S").r("NP") == ABCCat.p("S/NP")
        assert ABCCat.p("S") / "NP" == ABCCat.p("S/NP")
    
    def test_l(self):
        assert ABCCat.p("NP").l("S") == ABCCat.p("NP\\S")

    def test_adjunct(self):
        assert ABCCat.p("S/NP").adj_l() == ABCCat.p("<S/NP>\\<S/NP>")
        assert ABCCat.p("NP\\S").adj_r() == ABCCat.p("<NP\\S>/<NP\\S>")

    _lexer_items = [
        (
            "<ABC/DE>>F//G\\\\H<>JJK//L/MM/N",
            ("<", "ABC", "/", "DE", ">", ">", "F", "/", "/", "G", "\\", "\\", "H", "<", ">", "JJK", "/", "/", "L", "/", "MM", "/", "N"),
            None
        ),
        ("DE//<", ("DE", "/", "/", "<"), None),
        (
            "(ABC/DE))F//G\\\\H()JJK//L/MM/N",
            ("(", "ABC", "/", "DE", ")", ")", "F", "/", "/", "G", "\\", "\\", "H", "(", ")", "JJK", "/", "/", "L", "/", "MM", "/", "N"),
            "/\\()⊥⊤"
        ),
        ("DE//(", ("DE", "/", "/", "("), "/\\()⊥⊤"),
    ]

    @pytest.mark.parametrize("input, answer, symbols", _lexer_items)
    def test__lexer(self, input, answer, symbols):
        if symbols:
            res = tuple(ABCCat._lexer(input, symbols))
        else:
            res = tuple(ABCCat._lexer(input))

        assert res == answer

    simp_items = [
        (("S/NP", "NP"), ("S", ">")),
        (("S|NP", "NP"), ("S", "|")),
        (("PPs", "S|PPs"), ("S", "|")),
        (("NP", "NP\\S"), ("S", "<")),
        (("A\\B", "B\\C"), ("A\\C", "<B1")),
        (("C/B", "B/A"), ("C/A", ">B1")),
        (("C/<B\\A>", "B\\A"), ("C", ">")),
        (
            (
                "<<PPs\\Srel>/<PPs\\Srel>>",
                "<<<PPs\\Srel>/<PPs\\Srel>>\\<<PPs\\Srel>/<PPs\\Srel>>>"
            ),
            ("<<PPs\\Srel>/<PPs\\Srel>>", "<"),
        )
    ]

    @pytest.mark.parametrize("input, answer", simp_items)
    def test_simplify_exh(self, input, answer):
        left, right = input
        cat_exp, res_exp = answer
        
        res_set = ABCCat.simplify_exh(left, right)
        assert (ABCCat.p(cat_exp), res_exp) in {
            (cat, str(res)) for cat, res in res_set
        }

    @pytest.mark.parametrize("input, answer", simp_items)
    def test_mul(self, input, answer):
        left, right = input
        cat_exp, res_exp = answer

        res = ABCCat.p(left) * right
        assert res == ABCCat.p(cat_exp)

    def test_invert_dir(self):
        test_items = (
            ("⊥", "⊥"),
            ("NP", "NP"),
            ("NP\\S", "S/NP"),
            ("NP/S", "S\\NP"),
            ("S|NP", "S|NP"),
        )

        for item, res_exp in test_items:
            item_parsed = ABCCat.p(item)
            res_exp_parsed = ABCCat.p(res_exp)

            assert ~item_parsed == res_exp_parsed
            assert item_parsed == ~res_exp_parsed

        assert ~(ABCCat.p("S") / ABCCat.p("NP")) == ABCCat.p("NP\\S")

class Test_ABCCatBase:
    eq_ign_items = (
        (("NP", "NP"), True),
        (("NPq", "NP"), True),
        (("NPq", "VP"), False),
    )
    @pytest.mark.parametrize("input, answer", eq_ign_items)
    @pytest.mark.ign_feat
    def test_eq_ign_feat(self, input: Tuple[str, str], answer: bool):
        item1 = ABCCat.p(input[0])
        item2 = ABCCat.p(input[1])

        assert (item1.equiv_to(item2, ignore_feature = True)) == answer

    items_unify = (
        ("NP", "NP[s=true]", "NP[s=true]"),
        ("NP[q=true]", "NP", "NP[q=true]"),
        ("NP[q=true]", "NP[s=true]", "NP[s=true,q=true]"),
        ("NP[q=true]", "NP[q=false]", None),
    )
    @pytest.mark.parametrize("cat1, cat2, answer", items_unify)
    def test_unify(self, cat1, cat2, answer: ABCCat):
        cat1 = ABCCat.p(cat1, mode = ABCCatReprMode.CCG2LAMBDA)
        cat2 = ABCCat.p(cat2, mode = ABCCatReprMode.CCG2LAMBDA)
        assert cat1.unify(cat2) == (
            ABCCat.p(answer, ABCCatReprMode.CCG2LAMBDA)
            if answer 
            else None
        )

class Test_ABCCatFunctor:
    reduce_with_items = (
        (("S/NP", "NP", False), "S"),
        (("NP\\S", "NP", True), "S"),
        (("C/<B\\A>", "B\\A", False), "C"),
        (("C|<B\\A>", "B\\A", False), "C"),
    )
    @pytest.mark.parametrize("input, answer", reduce_with_items)
    def test_reduce_with(
        self,
        input: Tuple[str, str, bool],
        answer: str
    ):
        func, ant, ant_left = input
        f = cast(ABCCatFunctor, ABCCat.parse(func))
        res = f.reduce_with(ant, ant_left)
        assert res == ABCCat.parse(answer)

    reduce_with_ign_feat_items = (
        (("NPq\\S", "NP", True), "S"),
        (("C/<B\\Aq>", "B\\A", False), "C"),
    )
    @pytest.mark.ign_feat
    @pytest.mark.parametrize("input, answer", reduce_with_ign_feat_items)
    def test_reduce_with_ign_feat(
        self,
        input: Tuple[str, str, bool],
        answer: str
    ):
        self.test_reduce_with(input, answer)
    
    pprint_items = (
        ("C/<B\\A>", ABCCatReprMode.TLCG, "<C/<B\\A>>"),
        ("C/<B\\A>", ABCCatReprMode.TRADITIONAL, "<C/<A\\B>>"),
        ("C/<Bm3/A>", ABCCatReprMode.DEPCCG, "(C/(B[m3]/A))"),
    )
    @pytest.mark.parametrize("input, mode, answer", pprint_items)
    def test_pprint(self, input, mode, answer):
        assert ABCCat.p(input).pprint(mode) == answer