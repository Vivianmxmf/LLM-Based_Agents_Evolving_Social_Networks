# tests/test_llm_parser.py
from agents.llm_agent import LLMPolicy
from agents.base import AgentView

class DummyClient:
    def __init__(self, msg: str): self.msg = msg
    def generate(self, prompt: str, max_tokens: int | None = None) -> str: return self.msg

def _obs():
    return AgentView(i=0, z_net=10.0, xi=0.6, p=2.0, tau_i=0.3, ybar_i=1.0, deg_i=3)

def test_parse_happy_path():
    pol = LLMPolicy(grid_K=5, client=DummyClient("INDEX=3"))
    y = pol.act(_obs())
    # grid = [0.0, 0.25, 0.5, 0.75, 1.0]; y_max = 10/2=5 -> y = 0.75*5=3.75
    assert abs(y - 3.75) < 1e-9

def test_parse_fallback_to_br():
    pol = LLMPolicy(grid_K=5, client=DummyClient("nonsense"))
    y = pol.act(_obs())
    # Should be finite and within [0, y_max]
    assert 0.0 <= y <= 5.0