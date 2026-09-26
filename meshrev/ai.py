"""Optional LLM design-intent review (DeepSeek, OpenAI-compatible API, text only).
The model sees a compact geometric summary and may only return whitelisted, validated actions."""
import json
import os
import re
import urllib.request
from pathlib import Path

SYSTEM = """你是资深机械设计工程师，熟悉活塞发动机零部件（活塞、连杆、曲轴、凸轮轴、缸体、缸盖、飞轮、齿轮、法兰、支架等）
和通用零件的设计、制造工艺（铸造、锻造、机加工）与标准（ISO 273 通孔、ISO 261 螺纹、轴承配合、齿轮模数、优先数）。
输入是扫描网格逆向得到的特征树摘要（尺寸单位 mm，已按标准做过初步归整，measured 为实测值）。
请判断零件类型与设计意图，只输出 JSON：
{"part_type": "...", "remarks": ["简短中文结论..."],
 "actions": [{"id": 特征编号, "op": "set_d" | "drop" | "tier" | "name", "value": ...}]}
规则：set_d 仅用于孔(hole/holes)，value 为更合理的标准直径，且与实测值相差不超过 10%；
drop 仅用于明显是扫描缺陷/刻字/毛刺的 C 级小特征；tier 取 "A"(配合/接触面) "B"(一般) "C"(铸造外观)；
name 为参数英文名(如 PIN_BORE_D, BOLT_HOLE_D)。没有把握就不要给出动作。"""


def _key():
    k = os.environ.get('DEEPSEEK_API_KEY')
    if not k and Path('.env').exists():
        for line in Path('.env').read_text().splitlines():
            if line.strip().startswith('DEEPSEEK_API_KEY'):
                k = line.split('=', 1)[1].strip().strip('"\'')
    return k


def chat(messages, model=None, timeout=60):
    key = _key()
    if not key:
        raise RuntimeError('DEEPSEEK_API_KEY not set (env or .env)')
    body = dict(model=model or os.environ.get('DEEPSEEK_MODEL', 'deepseek-chat'), messages=messages,
                temperature=0.0, response_format={'type': 'json_object'})
    req = urllib.request.Request(os.environ.get('DEEPSEEK_URL', 'https://api.deepseek.com/chat/completions'),
                                 json.dumps(body).encode(), {'Content-Type': 'application/json',
                                                             'Authorization': f'Bearer {key}'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(json.loads(r.read())['choices'][0]['message']['content'])


def digest(feats, info):
    """Compact summary (~1-2k tokens) of the reconstruction for the LLM."""
    fs = []
    for i, f in enumerate(feats):
        d = dict(id=i, op=f['op'], mode=f.get('mode', 'add'), tier=f.get('tier', 'A'))
        if f['op'] in ('hole', 'holes'):
            d.update(d=f['d'], measured=round(f.get('d_mean', f['d']), 3), depth=round(f['h'], 2), axis='XYZ'[f['k']],
                     rough=round(f.get('rough', 0), 3), n=f.get('n', 1), pcd=f.get('pcd'), note=f.get('note'))
        elif f['op'] == 'prism':
            d.update(axis='XYZ'[f['ax']], z=[round(f['s0'], 2), round(f['s1'], 2)], draft=f.get('taper', 0),
                     sketch=''.join(e[0] for e in f['outer']))
        elif f['op'] == 'revolve':
            d.update(profile=[[round(v, 2) for v in e[1]] for e in f['profile']][:40])
        elif f['op'] == 'gear':
            d.update(teeth=f['n'], module=f.get('module'), r_tip=round(f['r_tip'], 2), r_root=round(f['r_root'], 2))
        fs.append(d)
    return dict(extents=[round(v, 2) for v in info['extents']], mode=info['mode'], sigma=round(info['sigma'], 3),
                tolerance=info['tol'], notes=info['notes'], features=fs)


def apply(feats, info, resp, log=print):
    """Validate and apply LLM actions; everything else is ignored."""
    info['notes'].append(f"AI 判定零件类型: {resp.get('part_type', '?')}")
    info['notes'] += [f'AI: {r}' for r in resp.get('remarks', [])[:8]]
    drop = set()
    for a in resp.get('actions', []):
        i, op, v = a.get('id'), a.get('op'), a.get('value')
        if not isinstance(i, int) or not 0 <= i < len(feats):
            continue
        f = feats[i]
        if op == 'set_d' and f['op'] in ('hole', 'holes') and isinstance(v, (int, float)):
            ref = f.get('d_mean', f['d'])
            if abs(v - ref) <= 0.1 * ref:
                log(f"  AI: hole #{i} Ø{f['d']} -> Ø{v}")
                f['d'] = float(v)
        elif op == 'drop' and f.get('tier') == 'C':
            drop.add(i)
        elif op == 'tier' and v in ('A', 'B', 'C'):
            f['tier'] = v
        elif op == 'name' and isinstance(v, str) and re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,30}', v):
            f['name'] = v.upper()
    return [f for i, f in enumerate(feats) if i not in drop], info


def review(feats, info, log=print):
    try:
        resp = chat([{'role': 'system', 'content': SYSTEM},
                     {'role': 'user', 'content': json.dumps(digest(feats, info), ensure_ascii=False, default=float)}])
    except Exception as e:  # offline / no key: the geometric pipeline stands on its own
        log(f'AI review skipped: {e}')
        return feats, info
    return apply(feats, info, resp, log)
