import json
import os
import pytest

from synthesized import ScenarioSynthesizer


@pytest.mark.integration
def test_scenarios_quick():
    passed = True
    failed = list()

    for root, dirs, files in os.walk('scenarios'):
        for filename in files:
            if filename.startswith('_') or not filename.endswith('.json'):
                continue

            try:
                with open(os.path.join(root, filename), 'r') as filehandle:
                    scenario = json.load(fp=filehandle)
                assert len(scenario) == 2 and 'functionals' in scenario and 'values' in scenario
                with ScenarioSynthesizer(
                    values=scenario['values'], functionals=scenario['functionals'], capacity=8, depth=1, batch_size=32
                ) as synthesizer:
                    synthesizer.learn(num_iterations=10)
                    df_synthesized = synthesizer.synthesize(num_rows=10000)
                    assert len(df_synthesized) == 10000

            except Exception as exc:
                passed = False
                failed.append((os.path.join(root, filename), exc))

    assert passed, '\n\n' + '\n\n'.join('{}\n{}'.format(path, exc) for path, exc in failed) + '\n'


def test_unittest_scenario_quick():
    with open('scenarios/unittest.json', 'r') as filehandle:
        scenario = json.load(fp=filehandle)
    assert len(scenario) == 2 and 'functionals' in scenario and 'values' in scenario
    with ScenarioSynthesizer(
        values=scenario['values'], functionals=scenario['functionals'], capacity=8, depth=1, batch_size=32
    ) as synthesizer:
        synthesizer.learn(num_iterations=10)
        df_synthesized = synthesizer.synthesize(num_rows=10000)
        assert len(df_synthesized) == 10000
