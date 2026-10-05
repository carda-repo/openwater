"""The identity switch defaults on and reaches data preparation and both model paths."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_pipeline as pipeline
from modelling_pipeline import make_effective_config


def load_entrypoint(relative):
    spec = spec_from_file_location('identity_entry_' + Path(relative).stem, ROOT / relative)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('enabled', [True, False])
def test_pipeline_passes_same_switch_to_preparation_and_model(tmp_path, monkeypatch, enabled):
    data = tmp_path / 'raw'
    (data / 'Operator_a').mkdir(parents=True)
    seen = {}
    monkeypatch.setattr(pipeline, 'clean_directory', lambda **kw: data / 'Operator_a')
    monkeypatch.setattr(pipeline, 'sort_directory', lambda **kw: data / 'Operator_a')
    monkeypatch.setattr(pipeline, 'build_features', lambda **kw: None)
    monkeypatch.setattr(pipeline, 'merge_and_sample', lambda cfg: seen.update(prepare=cfg))
    monkeypatch.setattr(pipeline, 'run_optuna_search', lambda cfg, *a, **kw: seen.update(model=cfg))
    pipeline.run_full_pipeline(data_dir=data, out_dir=tmp_path / 'out',
        x_tijdspad='01012026:01022026', y_tijdspad='01022026:01032026',
        x_test_tijdspad='01022026:01032026', y_test_tijdspad='01032026:01042026',
        do_operator_stats=False, do_active_filter=False,
        Niels_Identity_Confounding_switch=enabled, random_state=71)
    for cfg in seen.values():
        assert cfg['Niels_Identity_Confounding_switch'] is enabled
        assert cfg['random_state'] == 71
    assert set(seen) == {'prepare', 'model'}


@pytest.mark.parametrize('flag,expected', [(None, True),
    ('--no-Niels_Identity_Confounding_switch', False),
    ('--no-niels-identity-confounding-switch', False)])
def test_pipeline_cli_switch_default_and_off(monkeypatch, flag, expected):
    seen = {}
    monkeypatch.setattr(pipeline, 'run_full_pipeline', lambda **kw: seen.update(kw))
    args = ['--data-dir', '/raw', '--out-dir', '/out']
    if flag:
        args.append(flag)
    assert pipeline.main(args) == 0
    assert seen['Niels_Identity_Confounding_switch'] is expected


@pytest.mark.parametrize('path', ['7_modelling/run_optuna.py', '7_modelling/run_grid.py'])
def test_modelling_cli_follows_config_unless_explicitly_overridden(path, tmp_path, monkeypatch):
    entry = load_entrypoint(path)
    cfg_file = tmp_path / 'cfg.yaml'
    cfg_file.write_text('Niels_Identity_Confounding_switch: false\nrandom_state: 71\n')
    seen = []
    if 'optuna' in path:
        monkeypatch.setattr(entry, 'run_optuna_search', lambda cfg, *a, **kw: seen.append(cfg.copy()) or tmp_path)
    else:
        monkeypatch.setattr(entry, 'run_grid_search', lambda cfg, *a, **kw: seen.append(cfg.copy()) or [])
    args = ['--config', str(cfg_file), '--out-dir', str(tmp_path)]
    entry.main(args)
    entry.main(args + ['--Niels_Identity_Confounding_switch'])
    assert seen[0]['Niels_Identity_Confounding_switch'] is False
    assert seen[1]['Niels_Identity_Confounding_switch'] is True
    assert seen[0]['random_state'] == 71  # Keep the prepared dataset's split seed.
    assert seen[1]['random_state'] == 71


def test_effective_config_defaults_on_and_preserves_explicit_off():
    assert make_effective_config({})['Niels_Identity_Confounding_switch'] is True
    assert make_effective_config({'Niels_Identity_Confounding_switch': False})['Niels_Identity_Confounding_switch'] is False


def test_merge_cli_can_override_yaml_identity_switch_and_seed(tmp_path, monkeypatch):
    entry = load_entrypoint('6_merge_sample/run_merge_sample.py')
    cfg_file = tmp_path / 'cfg.yaml'
    cfg_file.write_text('Niels_Identity_Confounding_switch: true\nrandom_state: 23\n')
    seen = []
    monkeypatch.setattr(entry, 'merge_and_sample', lambda cfg: seen.append(cfg.copy()) or tmp_path)
    entry.main(['--config', str(cfg_file), '--no-Niels_Identity_Confounding_switch',
                '--random-state', '71'])
    assert seen[0]['Niels_Identity_Confounding_switch'] is False
    assert seen[0]['random_state'] == 71
