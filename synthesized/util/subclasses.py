from typing import Set, Type, TypeVar

C = TypeVar("C")


def get_all_subclasses(cls: Type[C]) -> Set[Type[C]]:
    """Recursively get all subclasses"""
    subclasses = set(cls.__subclasses__())
    for sc in subclasses:
        subclasses = subclasses.union(get_all_subclasses(sc))

    return subclasses
