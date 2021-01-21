def pytest_runtest_setup(item):
    markers = [marker for marker in item.iter_markers()]
    if len(markers) == 0:
        raise ValueError(f"Test '{item.name}' is not marked. Please mark it as 'slow' or 'fast'.")
