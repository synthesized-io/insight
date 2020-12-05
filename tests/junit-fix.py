import xml.etree.ElementTree as ET
tree = ET.parse('test-results/junit.xml')
root = tree.getroot()

suite = root.getchildren()[0]

cases = suite.getchildren()

file_to_case = {}
for case in cases:
    for child in case.getchildren():
        if child.tag == 'properties':
            case.remove(child)
    if case.get('classname') not in file_to_case:
        file_to_case[case.get('classname')] = [case]
    else:
        file_to_case[case.get('classname')].append(case)


files = [k for k in file_to_case.keys()]

suites = []

for file in files:
    suite1 = ET.Element('testsuite', attrib={
        'name': (file or '').replace('.', '/')+'.py',
        'file': (file or '').replace('.', '/')+'.py',
        'errors': str(sum([1 if 'error' in [c.tag for c in case.getchildren()] else 0 for case in file_to_case[file]])),
        'failures': str(sum([1 if 'failure' in [c.tag for c in case.getchildren()] else 0 for case in file_to_case[file]])),
        'hostname': suite.get('hostname') or '',
        'skipped': str(sum([1 if 'skipped' in [c.tag for c in case.getchildren()] else 0 for case in file_to_case[file]])),
        'tests': str(len(file_to_case[file])),
        'time': str(sum([float(c.get('time') or 0.0) for c in file_to_case[file]]).__round__(3)),
        'timestamp': suite.get('timestamp') or ''
    })

    suite1.extend(file_to_case[file])
    # print(ET.tostring(suite1))
    suites.append((suite1))


root.remove(suite)
root.extend(suites)

open('test-results/junit.xml', 'wb').write(ET.tostring(root, encoding='utf-8'))
