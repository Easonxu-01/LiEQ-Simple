#!/usr/bin/env python3
"""Rebuild docs/assets/data/paper-data.js from the three JSON files beside it.

The JSON stays the source of truth; this only re-emits it as a plain script so
the figures draw with no network. Run after replacing any of the JSON files.
"""
import io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'docs', 'assets', 'data')

FILES = [('fig3a', 'fig3a_band_cka_feature_planes.json'),
         ('fig3b', 'fig3b_band_cka_voxel_logits.json'),
         ('fig4',  'fig4_energy_real.json')]

HEAD = """/*! paper-data.js - the measurements behind the manuscript's Fig. 3 and Fig. 4,
 * embedded as a plain script so the figures draw with no network at all.
 *
 * charts-paper.js used to reach for these with fetch, which is right on a served
 * page and useless anywhere else: open index.html straight off disk and every
 * fetch is blocked by the file:// origin, so both figures fell back to a
 * "Figure data unavailable" card. A <script> tag has no such restriction, so
 * loading the same numbers this way makes the figures work from a local copy, a
 * zip a reviewer was sent, or an offline machine.
 *
 * GENERATED - do not hand-edit. Rebuild from the JSON with:
 *   python3 tools/build_paper_data.py
 * The JSON files remain the source of truth and are still shipped beside this.
 */
window.LIEQPaperData = {
"""

def main():
    parts = []
    for key, fn in FILES:
        path = os.path.join(DATA, fn)
        if not os.path.exists(path):
            sys.exit('missing: ' + path)
        with io.open(path, encoding='utf-8') as fh:
            d = json.load(fh)
        parts.append('  %s: %s' % (key, json.dumps(d, separators=(',', ':'), ensure_ascii=False)))
    out = HEAD + ',\n'.join(parts) + '\n};\n'
    dest = os.path.join(DATA, 'paper-data.js')
    with io.open(dest, 'w', encoding='utf-8') as fh:
        fh.write(out)
    print('wrote %s (%d bytes)' % (os.path.relpath(dest), len(out.encode('utf-8'))))

if __name__ == '__main__':
    main()
