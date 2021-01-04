def get_all_subclasses(cls):
    """Recursively get all subclasses"""
    return cls.__subclasses__() + [s for c in cls.__subclasses__() for s in get_all_subclasses(c)]
