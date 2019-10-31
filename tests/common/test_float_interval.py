from synthesized.common.conditional.conditional import FloatInterval, Inclusive, Exclusive


def test_left_open_right_open_parsing():
    interval = FloatInterval.parse('(123.45, 67.89)')
    assert interval.left == Exclusive(123.45)
    assert interval.right == Exclusive(67.89)


def test_left_closed_right_closed_parsing():
    interval = FloatInterval.parse('[123.45, 67.89]')
    assert interval.left == Inclusive(123.45)
    assert interval.right == Inclusive(67.89)


def test_left_open_right_closed_parsing():
    interval = FloatInterval.parse('(123.45, 67.89]')
    assert interval.left == Exclusive(123.45)
    assert interval.right == Inclusive(67.89)


def test_left_closed_right_open_parsing():
    interval = FloatInterval.parse('[123.45, 67.89)')
    assert interval.left == Inclusive(123.45)
    assert interval.right == Exclusive(67.89)
