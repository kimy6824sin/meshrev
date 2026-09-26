"""Command line: meshrev part.stl [-o out] [--ai]"""
import argparse
from .pipeline import run


def main():
    ap = argparse.ArgumentParser(description='Scan mesh (STL) -> parametric CAD (CadQuery script + STEP)')
    ap.add_argument('mesh', nargs='+')
    ap.add_argument('-o', '--out', default='out')
    ap.add_argument('--ai', action='store_true', help='DeepSeek review of design intent (needs DEEPSEEK_API_KEY)')
    a = ap.parse_args()
    for f in a.mesh:
        print(f'== {f}')
        run(f, a.out, a.ai)


if __name__ == '__main__':
    main()
