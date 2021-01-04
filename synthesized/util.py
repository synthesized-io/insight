def get_all_subclasses(cls):
    """Recursively get all subclasses"""
    subclasses = cls.__subclasses__()
    for sc in subclasses:
        subclasses.extend(get_all_subclasses(sc))

    return subclasses
