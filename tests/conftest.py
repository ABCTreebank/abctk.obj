import pytest
from abctk.obj.ID import RecordIDParser
from abctk.obj.Keyaki import Keyaki_ID
from abctk.obj.comparative import ABCTComp_BCCWJ_ID

@pytest.fixture(scope="module")
def ID_parser():
    p = RecordIDParser()
    p.register_parser(ABCTComp_BCCWJ_ID.from_string, 100)
    p.register_parser(Keyaki_ID.from_string, 40)
    return p