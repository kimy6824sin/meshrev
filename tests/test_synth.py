"""pytest: parameter recovery on noisy synthetic scans (sigma 0.03 mm, spikes, random pose)."""
import pytest
from synth import run


@pytest.mark.parametrize('name', ['flange', 'gear', 'shaft'])
def test_recovery(name):
    s, t = run(name)
    assert s['deviation']['rms'] < 0.2
    F = s['features']
    holes = sorted(f['d'] for f in F if f['op'] == 'hole')
    if name == 'flange':
        pat = [f for f in F if f['op'] == 'holes']
        assert pat and pat[0]['n'] == 4 and pat[0]['d'] == 6.6 and abs(pat[0]['pcd'] - 46) < 0.1
        assert 20.0 in holes
    elif name == 'gear':
        g = [f for f in F if f['op'] == 'gear']
        assert g and g[0]['n'] == 30 and g[0]['module'] == 2
        assert 20.0 in holes
    else:
        assert 5.0 in holes
