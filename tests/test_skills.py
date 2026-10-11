from jobrec.skills import category_of, extract_skills, vocabulary
from jobrec.text import clean, tokens


def test_clean_keeps_skill_punctuation():
    assert clean("Senior C++/Python Dev (Remote)!") == "senior c++/python dev remote"
    assert tokens("C#, .NET and node.js") == ["c#", "net", "and", "node.js"]


def test_aliases_map_to_canonical_names():
    found = extract_skills("Experience with sklearn, PyTorch, Keras and golang")
    assert found == ["scikit-learn", "pytorch", "tensorflow", "golang"]


def test_whole_phrase_matching_avoids_substrings():
    # "c" must not fire inside words, "go" (the verb) is not the Go language.
    assert "c" not in extract_skills("I can code in Python")
    assert "golang" not in extract_skills("I go to work by bus")
    assert extract_skills("firmware in C and C++") == ["c", "c++"]


def test_order_is_first_appearance_without_duplicates():
    assert extract_skills("MQTT, then ESP32, then MQTT again") == ["mqtt", "esp32"]


def test_longer_phrase_wins_but_implied_skill_also_reported():
    found = extract_skills("deployed to AWS IoT Core")
    assert "aws iot core" in found and "aws" in found


def test_vocabulary_is_consistent():
    vocab = vocabulary()
    assert len(vocab) > 80
    for name, skill in vocab.items():
        assert skill.name == name
        assert category_of(name) == skill.category
        assert name == name.lower()
