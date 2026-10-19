from eval.scoring import keyword_recall, says_no, score

REFUSAL = "I could not find this information in the patient's record."


def test_refusal_is_required_for_unanswerable_questions():
    assert score({"type": "refuse"}, REFUSAL) == (True, 1.0)
    assert score({"type": "refuse"}, "The blood type is O positive.")[0] is False


def test_negative_probe_accepts_denials_and_refusals():
    for answer in ["No, there is no record of asthma.", "There is no mention of asthma.",
                   "The patient has not been diagnosed with asthma.", REFUSAL]:
        assert says_no(answer), answer
    assert not says_no("Yes, asthma was diagnosed in 2010.")


def test_answerable_question_refused_scores_zero():
    assert score({"type": "year", "year": "2016"}, REFUSAL) == (False, 0.0)


def test_keywords_give_partial_credit():
    check = {"type": "keywords", "keywords": ["atorvastatin", "clopidogrel"]}
    assert score(check, "Atorvastatin 40 mg and Clopidogrel 75 mg.") == (True, 1.0)
    assert score(check, "Atorvastatin only.") == (False, 0.5)
    assert keyword_recall("anything", []) == 0.0


def test_numbers_within_tolerance():
    check = {"type": "numbers", "numbers": [147, 90], "tol": 0.5}
    assert score(check, "The latest reading was 147/90 mmHg on 24 Mar 2023.")[0]
    assert not score(check, "The latest reading was 132/85 mmHg.")[0]


def test_year():
    assert score({"type": "year", "year": "2016"}, "It was diagnosed on 10 April 2016.")[0]
