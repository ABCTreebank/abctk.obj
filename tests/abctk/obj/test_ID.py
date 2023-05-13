import pytest

from abctk.obj.ID import *

RAW_IDs = (
    ("asdf", SimpleRecordID("asdf")),
    ("asdfgg", SimpleRecordID("asdfgg")),
)

class Test_SimpleRecordID:
    @pytest.mark.parametrize("input, answer", RAW_IDs)
    def test_from_string(self, input: str, answer: SimpleRecordID):
        parsed = SimpleRecordID.from_string(input)
        assert parsed == answer
        assert parsed.orig == input
