import pytest

from abctk.obj.ID import *
from abctk.obj.comparative import ABCTComp_BCCWJ_ID
from abctk.obj.Keyaki import Keyaki_ID

RAW_IDs = (
    ("asdf", SimpleRecordID("asdf")),
    ("asdfgg", SimpleRecordID("asdfgg")),
    ("ABCT-COMP-BCCWJ;yori;LBg2_00048,14780", ABCTComp_BCCWJ_ID("yori", "LBg2_00048", 14780)),
    ("2_aozora_Chiri-1956;JP", Keyaki_ID("aozora_Chiri-1956", 2, "JP")),
)

class Test_RecordIDParser:
    @pytest.mark.parametrize("input, answer", RAW_IDs)
    def test_parse(
        self,
        ID_parser: RecordIDParser,
        input: str,
        answer: RecordID
    ):
        assert ID_parser.parse(input) == answer