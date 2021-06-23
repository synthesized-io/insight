import matplotlib.pyplot as plt

from synthesized.util import axes_grid, get_all_subclasses


def test_get_all_subclasses():
    class Foo:
        pass

    class FooChild(Foo):
        pass

    class FooChild2(FooChild):
        pass

    class FooChild3(FooChild2):
        pass

    class FooChild4(FooChild3):
        pass

    class Foo2(Foo):
        pass

    class Foo2Child(Foo2):
        pass

    class Foo2Child2(Foo2Child):
        pass

    sc = get_all_subclasses(Foo)
    assert sc == {FooChild, FooChild2, FooChild3, FooChild4, Foo2, Foo2Child, Foo2Child2}


def test_axes_grid():
    fig, ax = plt.subplots()

    axes = axes_grid(ax, rows=3, cols=3, col_titles=['a', 'b', 'c'], row_titles=['1', '2', '3'])
    assert len(axes) == 3
    assert len(axes[0]) == 3
