from kite.navigate import Place, Plan, choose, decode_polyline, destination_text


def test_destination_text_strips_the_ask_and_spots_nearest():
    assert destination_text("Take me to the nearest pharmacy.") == ("pharmacy", True)
    assert destination_text("take me to Doe Library") == ("Doe Library", False)
    assert destination_text("navigate to coffee near me") == ("coffee", True)


def test_decode_polyline_matches_googles_example():
    assert decode_polyline("_p~iF~ps|U_ulLnnqC_mqNvxq`@") == [(38.5, -120.2), (40.7, -120.95), (43.252, -126.453)]


def test_choose_rejects_an_out_of_range_answer():
    p = Plan("clarify", "x", "Which one?", (0.0, 0.0), choices=[Place("a", None, 0, 0)])
    assert choose(p, 2).status == "error"
