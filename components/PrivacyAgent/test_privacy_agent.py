import pytest
from privacy_agent import PrivacyAgent

@pytest.fixture
def agent():
    """Setup the agent once for all tests."""
    return PrivacyAgent()

def test_preliminary_censor_removes_date(agent):
    text = "My meeting is on 05/20/2024."
    result, changed = agent.preliminary_censor(text)
    assert "05/20/2024" not in result
    assert changed == True

def test_preliminary_censor_removes_SSN(agent):
    text = "Elon Musk (SSIN: 400-80-9876) visited the Tesla factory in Berlin on 08-01-2004."
    result, changed = agent.preliminary_censor(text)
    assert "400-80-9876" not in result
    assert changed == True

def test_preliminary_censor_no_censor(agent):
    text = "The watermelon is green."
    result, changed = agent.preliminary_censor(text)
    assert result == text
    assert changed == False

def test_preliminary_censor_empty(agent):
    text = ""
    result, changed = agent.preliminary_censor(text)
    assert result == text
    assert changed == False

def test_in_depth_censor_removes_date(agent):
    text = "My meeting is on 05/20/2024."
    result, changed = agent.in_depth_censor(text)
    assert "05/20/2024" not in result
    assert changed == True

def test_in_depth_censor_removes_SSN(agent):
    text = "Elon Musk (SSIN: 400-80-9876) visited the Tesla factory in Berlin on 08-01-2004."
    result, changed = agent.in_depth_censor(text)
    assert "400-80-9876" not in result
    assert changed == True

def test_in_depth_censor_removes_name(agent):
    text = "Elon Musk (SSIN: 400-80-9876) visited the Tesla factory in Berlin on 08-01-2004."
    result, changed = agent.in_depth_censor(text)
    assert "ELon Musk" not in result
    assert changed == True

def test_in_depth_censor_removes_organization(agent):
    text = "Elon Musk (SSIN: 400-80-9876) visited the Tesla factory in Berlin on 08-01-2004."
    result, changed = agent.in_depth_censor(text)
    assert "Tesla" not in result
    assert changed == True

def test_in_depth_censor_no_censor(agent):
    text = "The watermelon is green."
    result, changed = agent.in_depth_censor(text)
    assert result == text
    assert changed == False

def test_in_depth_censor_empty(agent):
    text = ""
    result, changed = agent.in_depth_censor(text)
    assert result == text
    assert changed == False

def test_custom_regex_censor_replaces_words(agent):
    text = "Send Project phoenix files to ABC-9988 in zip 90210."
    result, changed = agent.custom_regex_censor(text,[r"Project Phoenix",r"\b[A-Z]{3}-\d{4}\b"],["[PROJECT_ALPHA]","[CASE_ID]"])
    assert "[PROJECT_ALPHA]" in result
    assert "[CASE_ID]" in result
    assert "Project Phoenix" not in result
    assert "ABC-9988" not in result
    assert changed == True

def test_custom_regex_censor_no_replacement(agent):
  text = "Send Project phoenix files to ABC-9988 in zip 90210."
  result, changed = agent.custom_regex_censor(text)
  assert result == text
  assert changed == False

def test_custom_regex_censor_empty(agent):
  text = ""
  result, changed = agent.custom_regex_censor(text)
  assert result == text
  assert changed == False

def test_custom_regex_mismatched_lists(agent):
    with pytest.raises(ValueError):
        agent.custom_regex_censor("Some text", ["bad"], ["too", "many"])
