"""Report: deviation color-map preview PNG + self-contained interactive HTML (three.js)."""
import base64
import json
import numpy as np


def preview(path, scan, model, pts, d, lim=0.5, views=((30, 45), (30, 225), (-60, 45))):
    """PNG: scan points colored by deviation (top row) and model shading (bottom row)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(4 * len(views), 8))
    c, r = scan.bounds.mean(0), scan.extents.max() / 2
    for i, (el, az) in enumerate(views):
        for row, what in enumerate(('dev', 'model')):
            ax = fig.add_subplot(2, len(views), row * len(views) + i + 1, projection='3d')
            if what == 'dev':
                sc = ax.scatter(*pts.T, c=np.clip(d, -lim, lim), cmap='coolwarm', s=0.3, vmin=-lim, vmax=lim)
            else:
                tri = model.triangles
                sh = np.clip(model.face_normals @ np.array([0.3, 0.5, 0.8]), 0.15, 1)
                ax.add_collection3d(Poly3DCollection(tri, facecolors=plt.cm.gray(sh), edgecolor='none'))
            ax.set_xlim(c[0] - r, c[0] + r)
            ax.set_ylim(c[1] - r, c[1] + r)
            ax.set_zlim(c[2] - r, c[2] + r)
            ax.view_init(el, az)
            ax.set_axis_off()
    fig.colorbar(sc, ax=fig.axes[:len(views)], shrink=0.6, label='deviation mm (+ = model lacks material)')
    plt.savefig(path, dpi=70, bbox_inches='tight')
    plt.close(fig)


def _b64(a, dt):
    return base64.b64encode(np.ascontiguousarray(a, dt).tobytes()).decode()


PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>meshrev report</title><style>
:root{--bg:#f6f7f9;--fg:#1d2330;--card:#fff;--mut:#667}
@media (prefers-color-scheme:dark){:root{--bg:#14161b;--fg:#e6e8ee;--card:#1d2027;--mut:#99a}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}
main{display:grid;grid-template-columns:minmax(0,1fr) 380px;height:100vh}
#v{position:relative}#v canvas{display:block;width:100%;height:100%}
aside{overflow:auto;padding:16px;background:var(--card)}h1{font-size:18px;margin:0 0 8px}
table{border-collapse:collapse;width:100%;font-size:12px}td,th{border-bottom:1px solid #8883;padding:3px 4px;text-align:left}
.k{color:var(--mut)}#bar{position:absolute;left:12px;bottom:12px;background:var(--card);padding:6px 10px;border-radius:6px;font-size:12px}
label{margin-right:10px}@media (max-width:800px){main{grid-template-columns:1fr;grid-template-rows:60vh auto;height:auto}}
</style></head><body><main><div id="v"><div id="bar"><label><input type="checkbox" id="cs" checked>扫描(偏差色)</label>
<label><input type="checkbox" id="cm" checked>重建模型</label> 色标 ±<span id="lim"></span> mm</div></div><aside id="side"></aside></main>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const D=__DATA__;const dec=(s,T)=>new T(Uint8Array.from(atob(s),c=>c.charCodeAt(0)).buffer);
const el=document.getElementById('v'),R=new THREE.WebGLRenderer({antialias:true});el.appendChild(R.domElement);
const S=new THREE.Scene(),C=new THREE.PerspectiveCamera(40,1,0.1,1e5);S.add(new THREE.HemisphereLight(0xffffff,0x444455,2.2));
const dl=new THREE.DirectionalLight(0xffffff,1.5);dl.position.set(1,2,3);S.add(dl);
function mesh(v,f,col,opts){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(dec(v,Float32Array),3));
g.setIndex(new THREE.BufferAttribute(dec(f,Uint32Array),1));if(col)g.setAttribute('color',new THREE.BufferAttribute(dec(col,Float32Array),3));
g.computeVertexNormals();return new THREE.Mesh(g,new THREE.MeshStandardMaterial(opts));}
const ms=mesh(D.sv,D.sf,D.sc,{vertexColors:true,side:2}),mm=mesh(D.mv,D.mf,null,{color:0x6d8fd6,transparent:true,opacity:.45,side:2});
S.add(ms,mm);const bb=new THREE.Box3().setFromObject(ms),c=bb.getCenter(new THREE.Vector3()),r=bb.getSize(new THREE.Vector3()).length();
C.position.copy(c).add(new THREE.Vector3(r,r*.8,r));const O=new OrbitControls(C,R.domElement);O.target.copy(c);
cs.onchange=()=>ms.visible=cs.checked;cm.onchange=()=>mm.visible=cm.checked;lim.textContent=D.lim;
function rs(){const w=el.clientWidth,h=el.clientHeight;R.setSize(w,h);C.aspect=w/h;C.updateProjectionMatrix();}
addEventListener('resize',rs);rs();R.setAnimationLoop(()=>{O.update();R.render(S,C)});
const s=D.summary,dv=s.deviation;let h=`<h1>${s.file}</h1><p class=k>建模方式: ${s.mode} · 噪声σ ${s.sigma.toFixed(3)} mm</p>
<table><tr><th>偏差</th><td>RMS ${dv.rms.toFixed(3)} · P95 ${dv.p95.toFixed(3)} · 最大 ${dv.max.toFixed(2)} mm · ±${s.tol.B}内 ${(dv.within*100).toFixed(1)}%</td></tr>
<tr><th>公差分级</th><td>A 配合面 ${s.tol.A.toFixed(2)} · B 一般 ${s.tol.B} · C 铸造 ${s.tol.C.toFixed(2)} mm</td></tr>`+
Object.entries(s.tiers).map(([k,v])=>`<tr><th>${k}级 (${(v.frac*100).toFixed(0)}%)</th><td>RMS ${v.rms.toFixed(3)} · ±${s.tol[k].toFixed(2)}内 ${(v.within*100).toFixed(1)}%</td></tr>`).join('')+`</table>`;
if(s.notes.length)h+='<h3>设计意图/知识校正</h3><ul>'+s.notes.map(n=>`<li>${n}</li>`).join('')+'</ul>';
h+='<h3>特征树</h3><table><tr><th>#</th><th>特征</th><th>参数</th></tr>'+s.features.map((f,i)=>{const {op,mode,tier,...p}=f;
return `<tr><td>${i}</td><td>${op} ${mode=='cut'?'(切除)':''} [${tier}]</td><td>${Object.entries(p).map(([k,v])=>k+'='+(typeof v=='number'?+v.toFixed(3):v)).join(' ')}</td></tr>`}).join('')+'</table>';
side.innerHTML=h;
</script></body></html>"""


def write(path, scan, model, pts, d, summary, lim=None, max_faces=30000):
    """Self-contained HTML viewer: scan colored by deviation + translucent model + feature tree."""
    lim = lim or max(0.1, 2 * summary['tol']['B'])
    s = scan
    if len(s.faces) > max_faces:
        s = s.simplify_quadric_decimation(face_count=max_faces)
    mdl = model
    if len(mdl.faces) > max_faces:
        mdl = mdl.simplify_quadric_decimation(face_count=max_faces)
    from .verify import distance
    dv = distance(model, s.vertices)
    t = np.clip((dv + lim) / (2 * lim), 0, 1)[:, None]
    col = (1 - t) * np.array([0.23, 0.30, 0.75]) + t * np.array([0.71, 0.02, 0.15])
    mid = 1 - np.abs(2 * t - 1)
    col = col * (1 - mid) + mid * np.array([0.87, 0.87, 0.87])
    data = dict(sv=_b64(s.vertices, '<f4'), sf=_b64(s.faces, '<u4'), sc=_b64(col, '<f4'),
                mv=_b64(mdl.vertices, '<f4'), mf=_b64(mdl.faces, '<u4'), lim=lim, summary=summary)
    path.write_text(PAGE.replace('__DATA__', json.dumps(data, default=float, ensure_ascii=False)), encoding='utf-8')
    preview(path.with_suffix('.png'), scan, model, pts, d, lim)
