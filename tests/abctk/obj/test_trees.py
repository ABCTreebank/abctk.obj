import io
from typing import Tuple

import pytest
from abctk.obj.trees import *
from abctk.obj.ID import RecordIDParser

class TestTree:
    RAW_TREES = (
        (
            "((a b c) d (e (f g (h ))))",
            Tree("", (
                Tree("a", (Tree("b"), Tree("c"))),
                Tree("d"),
                Tree("e", (Tree("f", (Tree("g"), Tree("h"))),))
            ))
        ),
        (
            "adf",
            Tree("adf"),
        ),
        (
            "(aaa (b c d) (e f g) (h (i j)))",
            Tree("aaa", (
                Tree("b", (Tree("c"), Tree("d"))),
                Tree("e", (Tree("f"), Tree("g"))),
                Tree("h", (Tree("i", (Tree("j"), )), )),
            ))
        ),
        # "((a b c) d (e (f g (h )))) adf (a (b c d) (e f g) (h (i j)))",
    )

    @pytest.mark.parametrize("input, result", RAW_TREES)
    def test_lexer(self, input: str, result: Tree):
        # Pass if no errors
        tuple(Tree.lexer(io.StringIO(input)))

    @pytest.mark.parametrize("input, result", RAW_TREES)
    def test_parse_stream(self, input: str, result: Tree):
        parsed = next(Tree.parse_stream(io.StringIO(input))).solidify()

        assert parsed == result

        # Pass if no errors

    RAW_COMMENT_ROOTS = (
        ("(ROOT A) (COMMENT a1)", "(ROOT A (COMMENT a1))"),
        ("(COMMENT 0) (ROOT A) (COMMENT a1)", "(ROOT (COMMENT 0) A (COMMENT a1))"),
        ("(COMMENT 0) (COMMENT 01) (ROOT A) (COMMENT a1) (COMMENT a2) (ROOT B) (COMMENT b1)", "(ROOT (COMMENT 0) (COMMENT 01) A (COMMENT a1) (COMMENT a2)) (ROOT B (COMMENT b1))"),
    )

    @pytest.mark.parametrize("trees_raw, results_raw", RAW_COMMENT_ROOTS)
    def test_merge_comments(self, trees_raw: str, results_raw: str):
        trees = tuple(
            tree.solidify() 
            for tree in Tree.parse_stream(io.StringIO(trees_raw))
        )
        results = tuple(
            tree.solidify()
            for tree in Tree.parse_stream(io.StringIO(results_raw))
        )

        assert tuple(Tree.lower_comments(trees)) == results


    RAW_TREES_WITH_ID_COMMENTS = (
        (
            "( (IP-MAT (PP (IP-ADV (PP (NP (PP (NP (NPR 明治)) (P の)) (N 中頃) (PU 、)) (P といえば)) (NP-SBJ *) (PU 、) (NP-PRD (IP-EMB (PP (NP (N アイヌ)) (P が)) (NP-SBJ *が*) (ADVP (ADV まだ)) (IP-ADV (PP (NP (N アイヌ語)) (P を)) (NP-OB1 *を*) (VB 使っ) (P て)) (CONJ *) (VB 暮らし) (P て) (VB2 い) (AXD た)) (N 時代)) (AX な) (FN の) (AX で) (VB2 あり) (AX ます)) (P が)) (CONJ *) (PU 、) (IP-ADV (IP-ADV (PP (NP (PP (NP (PP (NP (PP (NP (NPR 北海道)) (P の)) (N 南)) (P の)) (N 方)) (P の)) (PU 、) (D とある) (N アイヌ部落)) (P に)) (PU 、) (PP (NP (IP-REL (NP-SBJ *T*) (NP-TMP (N 当時)) (ADVP (ADV まだ)) (IP-ADV (ADVP (ADJN 非常) (AX に)) (ADJI 若く)) (CONJ *) (PU 、) (NP-PRD (PP (NP (N 新進気鋭)) (P の)) (N 牧師)) (AX で) (VB2 あら) (VB2 れ) (AXD た)) (NPR バチラー博士)) (P が)) (NP-SBJ *が*) (VB あらわれ) (P て)) (CONJ *) (PU 、) (NP-SBJ *pro*) (PP (NP (PP (NP (N 部落)) (P の)) (N アイヌ)) (P を)) (NP-OB1 *を*) (VB 集め) (P て)) (CONJ *) (PU 、) (NP-SBJ *pro*) (PP (NP (N キリスト教)) (P について)) (PU 、) (PP (NP (N アイヌ語)) (P で)) (PP (NP (N 説教)) (P を)) (NP-OB1 *を*) (VB 致し) (AX まし) (AXD た) (PU 。)) (ID 4_aozora_Chiri-1956;JP))",
            "4_aozora_Chiri-1956;JP",
        ),
        (
            "( (COMMENT {probability=-18.01902198791504}) (Sm (PPs (NPq (N (<N/N> cat) (N dog))) (<NP\\PPs> table)) (<PPs\\Sm> (CPt (Ssub (<Ssub/Ssub> (<Ssub/Ssub> (NP (N (Ns car) (<Ns\\N> chair)))) (<<Ssub/Ssub>\\<Ssub/Ssub>> and)) (Ssub (<Ssub/Ssub> star) (Ssub (PPs (PPs (NP (N (NUM three)) (<N\\NP> book))) (<PPs\\PPs> and)) (<PPs\\Ssub> (<<PPs\\Ssub>/<PPs\\Ssub>> (<PPs\\Sa> (N computer))) (<PPs\\Ssub> (<<PPs\\Ssub>/<PPs\\Ssub>> (NPq (N apple)) (<NP\\<<PPs\\Ssub>/<PPs\\Ssub>>> pen)) (<PPs\\Ssub> (<PPs\\Ssub> (<PPs\\Ssub> water) (<Ssub\\Ssub> paper)) (<Ssub\\Ssub> television))))))) (<Ssub\\CPt> bird)) (<CPt\\<PPs\\Sm>> (<CPt\\<PPs\\Sm>> (<CPt\\<PPs\\Sm>> flower) (<Sm\\Sm> window)) (<Sm\\Sm> house)))) (ID ABCT-COMP-BCCWJ;yori;LBl5_00051,85650))",
            "ABCT-COMP-BCCWJ;yori;LBl5_00051,85650"
        )
    )

    @pytest.mark.parametrize("tree_raw, ID_raw", RAW_TREES_WITH_ID_COMMENTS)
    def test_split_ID_from_tree(
        self, 
        ID_parser: RecordIDParser, 
        tree_raw: str, 
        ID_raw: str
    ):
        tree_parsed = next(Tree.parse_stream(io.StringIO(tree_raw))).solidify()

        ID, _ = tree_parsed.split_ID_from_tree(ID_parser)

        assert ID_parser.parse(ID_raw) == ID

    RAW_TREES_WITH_BRANCHES = (
        (
            "(S (NP (PRP My) (NN daughter)) (VP (VBD broke) (NP (NP (DET the) (JJ red) (NN toy)) (PP (IN with) (NP (DET a) (NN hammer))))))",
            (
                ("S", "NP", "PRP", "My"),
                ("S", "NP", "NN", "daughter"),
                ("S", "VP", "VBD", "broke"),
                ("S", "VP", "NP", "NP", "DET", "the"),
                ("S", "VP", "NP", "NP", "JJ", "red"),
                ("S", "VP", "NP", "NP", "NN", "toy"),
                ("S", "VP", "NP", "PP", "IN", "with"),
                ("S", "VP", "NP", "PP", "NP", "DET", "a"),
                ("S", "VP", "NP", "PP", "NP", "NN", "hammer"),
            )
        ),
    )

    @pytest.mark.parametrize("tree_raw, result", RAW_TREES_WITH_BRANCHES)
    def test_iter_leaves_with_branches(self, tree_raw: str, result):
        tree_parsed = next(Tree.parse_stream(io.StringIO(tree_raw))).solidify()

        assert tuple(
            tuple(branch)
            for branch in tree_parsed.iter_leaves_with_branches()
        ) == result

class TestGRVCell:
    RAW_TREES_WITH_GDV = (
        (
            "(S (NP (PRP My) (NN daughter)) (VP (VBD broke) (NP (NP (DET the) (JJ red) (NN toy)) (PP (IN with) (NP (DET a) (NN hammer))))))",
            (
                GRVCell("My", "PRP", 2, "NP"), 
                GRVCell("daughter", "NN", -1, "S"), 
                GRVCell("broke", "VBD", 1, "VP"),
                GRVCell("the", "DET", 2, "NP"),
                GRVCell("red", "JJ", 0, "NP"), 
                GRVCell("toy", "NN", -1, "NP"), 
                GRVCell("with", "IN", 1, "PP"), 
                GRVCell("a", "DET", 1, "NP"),
                GRVCell("hammer", "NN", 0, ""),
            ),
        ),
        (
            "(IP-MAT (ADJI ありがとう) (VB2 ござい) (AX ます))",
            (
                GRVCell("ありがとう", "ADJI", 1, "IP-MAT"),
                GRVCell("ござい", "VB2", 0, "IP-MAT"),
                GRVCell("ます", "AX", 0, ""),
            )
        )
    )

    @pytest.mark.parametrize("tree_raw, result", RAW_TREES_WITH_GDV)
    def test_encode_GRV(self, tree_raw: str, result):
        tree_parsed = next(Tree.parse_stream(io.StringIO(tree_raw))).solidify()
        tree_encoded = tuple(GRVCell.encode(tree_parsed))

        print(tree_encoded)
        assert tree_encoded == result
