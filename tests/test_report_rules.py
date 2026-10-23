from knee.report_rules import score_report


def s(text, label):
    return score_report(text)[label]


def test_english_positive_and_negative():
    assert s("Complete tear of the anterior cruciate ligament.", "ACL") == 1.0
    assert s("The ACL is intact.", "ACL") == 0.0
    assert s("No joint effusion.", "Effusion") == 0.0
    assert s("Large joint effusion.", "Effusion") == 1.0


def test_laterality():
    text = "Horizontal tear of the posterior horn of the medial meniscus. Lateral meniscus intact."
    assert s(text, "Medial Meniscus") == 1.0
    assert s(text, "Lateral Meniscus") == 0.0


def test_not_mentioned_is_low():
    assert s("Small Baker's cyst.", "Fracture") < 0.1


def test_non_displaced_fracture_is_positive():
    assert s("Non-displaced fracture of the lateral tibial plateau.", "Fracture") == 1.0


def test_uncertain():
    assert 0.0 < s("Possible partial tear of the MCL.", "MCL") < 1.0


def test_other_languages():
    assert s("Ruptur des vorderen Kreuzbandes. Kein Gelenkerguss.", "ACL") == 1.0
    assert s("Ruptur des vorderen Kreuzbandes. Kein Gelenkerguss.", "Effusion") == 0.0
    assert s("Rotura del ligamento cruzado anterior.", "ACL") == 1.0
    assert s("Geen gewrichtsvocht.", "Effusion") == 0.0
    assert s("Épanchement articulaire abondant.", "Effusion") == 1.0


def test_impression_wins():
    text = "Findings: possible bone bruise of the lateral femoral condyle. Impression: no contusion."
    assert s(text, "Contusion") == 0.0
