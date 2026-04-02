from unittest.mock import patch

from src.translator import translate_content


@patch("src.translator._chat")
def test_chinese(mock_chat):
    mock_chat.side_effect = ["Chinese", "This is a Chinese message"]
    is_english, translated_content = translate_content("这是一条中文消息")
    assert is_english is False
    assert translated_content == "This is a Chinese message"
    assert mock_chat.call_count == 2


@patch("src.translator._chat")
def test_llm_normal_response(mock_chat):
    mock_chat.side_effect = ["English"]
    is_english, out = translate_content("Hello world")
    assert is_english is True
    assert out == "Hello world"
    assert mock_chat.call_count == 1


@patch("src.translator._chat")
def test_llm_gibberish_response(mock_chat):
    mock_chat.side_effect = ["I don't understand your request"]
    is_english, out = translate_content("Hier ist ein Test")
    assert is_english is True
    assert out == "Hier ist ein Test"
    assert mock_chat.call_count == 1
