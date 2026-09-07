/* ArcANE project page — shared data access.
   Loads data/catalog.json, the per-character bundles (gzip), and the per-probe
   response bundles (gzip). Exposes window.ArcData. Plain ES2020, no build step. */
(function () {
  'use strict';

  var BASE = (function () {
    // Resolve relative to the script location so both index.html and explore.html work.
    var s = document.currentScript && document.currentScript.src;
    if (!s) return 'data/';
    return s.replace(/static\/js\/[^/]*$/, '') + 'data/';
  })();

  var cache = {};

  function fetchBytes(url) {
    return fetch(url, { cache: 'force-cache' }).then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status + ' for ' + url);
      return r.arrayBuffer();
    });
  }

  function isGzip(buf) {
    var u = new Uint8Array(buf, 0, 2);
    return u.length === 2 && u[0] === 0x1f && u[1] === 0x8b;
  }

  function gunzip(buf) {
    if (typeof DecompressionStream === 'undefined') {
      return Promise.reject(new Error('This browser cannot decompress gzip data (DecompressionStream is unavailable). Please use a current version of Chrome, Firefox, Safari, or Edge.'));
    }
    var ds = new DecompressionStream('gzip');
    var stream = new Blob([buf]).stream().pipeThrough(ds);
    return new Response(stream).arrayBuffer();
  }

  function loadJSON(url) {
    if (cache[url]) return cache[url];
    cache[url] = fetchBytes(url).then(function (buf) {
      // GitHub Pages serves .gz as application/gzip (no transparent decoding); a
      // proxy that already decoded it would hand us plain JSON, so sniff the magic.
      var p = isGzip(buf) ? gunzip(buf) : Promise.resolve(buf);
      return p.then(function (plain) { return JSON.parse(new TextDecoder('utf-8').decode(plain)); });
    }).catch(function (e) { delete cache[url]; throw e; });
    return cache[url];
  }

  var api = {
    base: BASE,
    catalog: function () { return loadJSON(BASE + 'catalog.json'); },
    paper: function () { return loadJSON(BASE + 'paper.json'); },
    character: function (novelId, charId) { return loadJSON(BASE + 'characters/' + novelId + '__' + charId + '.json.gz'); },
    responses: function (novelId, charId, probeId) { return loadJSON(BASE + 'responses/' + novelId + '__' + charId + '/' + probeId + '.json.gz'); },

    /* ── vocab helpers (all take the catalog) ── */
    PTYPES: ['in_text', 'in_world', 'out_of_world'],
    METRICS: ['apf', 'rpf', 'rae', 'ptf'],
    MODES: ['vanilla', 'summary', 'rag', 'lifechoice', 'timechara', 'arc'],
    model: function (cat, id) { return cat.models.find(function (m) { return m.id === id; }); },
    mode: function (cat, id) { return cat.modes.find(function (m) { return m.id === id; }); },
    ptypeLabel: function (cat, id) { var p = cat.probe_types.find(function (x) { return x.id === id; }); return p ? p.label : id; },
    modeLabel: function (cat, id) { var m = api.mode(cat, id); return m ? m.label : id; },
    modelLabel: function (cat, id) { var m = api.model(cat, id); return m ? m.label : id; },
    novel: function (cat, id) { return cat.novels.find(function (n) { return n.id === id; }); },
    characterMeta: function (cat, novelId, charId) { var n = api.novel(cat, novelId); return n && n.characters.find(function (c) { return c.id === charId; }); },

    /* cell = table[model][mode] -> {ptype: [apf,rpf,rae,ptf]} */
    overall: function (cell) {
      if (!cell) return null;
      var vals = [];
      api.PTYPES.forEach(function (pt) { (cell[pt] || []).forEach(function (v) { if (v != null) vals.push(v); }); });
      return vals.length ? vals.reduce(function (a, b) { return a + b; }, 0) / vals.length : null;
    },
    /* mean of one metric (index 0..3) or 'all' across probe types */
    cellValue: function (cell, ptype, metric) {
      if (!cell) return null;
      var mi = metric === 'all' ? null : api.METRICS.indexOf(metric);
      var pts = ptype === 'all' ? api.PTYPES : [ptype];
      var vals = [];
      pts.forEach(function (pt) {
        var arr = cell[pt]; if (!arr) return;
        if (mi === null) arr.forEach(function (v) { if (v != null) vals.push(v); });
        else if (arr[mi] != null) vals.push(arr[mi]);
      });
      return vals.length ? vals.reduce(function (a, b) { return a + b; }, 0) / vals.length : null;
    },
    /* Arc minus the strongest non-Arc mode on a table row set */
    arcLift: function (byMode, ptype, metric) {
      var arc = api.cellValue(byMode && byMode.arc, ptype, metric);
      if (arc == null) return null;
      var best = null, bestMode = null;
      api.MODES.forEach(function (m) {
        if (m === 'arc') return;
        var v = api.cellValue(byMode[m], ptype, metric);
        if (v != null && (best == null || v > best)) { best = v; bestMode = m; }
      });
      return best == null ? null : { lift: arc - best, arc: arc, best: best, bestMode: bestMode };
    },
    /* probe-level: scores[model][mode] = {phases:{idx:[a,r,e]}, ptf:[al,di,sh]|null} */
    probePhaseAvg: function (cell) {
      if (!cell || !cell.phases) return null;
      var vals = [];
      Object.keys(cell.phases).forEach(function (k) { var s = cell.phases[k]; vals.push((s[0] + s[1] + s[2]) / 3); });
      return vals.length ? vals.reduce(function (a, b) { return a + b; }, 0) / vals.length : null;
    },
    probePTF: function (cell) {
      if (!cell || !cell.ptf) return null;
      var v = cell.ptf.filter(function (x) { return x != null; });
      return v.length ? v.reduce(function (a, b) { return a + b; }, 0) / v.length : null;
    },
    probeOverall: function (cell) {
      var a = api.probePhaseAvg(cell), p = api.probePTF(cell);
      if (a == null && p == null) return null;
      if (p == null) return a;
      if (a == null) return p;
      return (a * 3 + p) / 4;
    },
    fmt: function (v, d) { if (v == null || isNaN(v)) return '—'; return v.toFixed(d == null ? 1 : d); },
    fmtSigned: function (v, d) { if (v == null || isNaN(v)) return '—'; var s = v.toFixed(d == null ? 1 : d); return (v > 0 ? '+' : v < 0 ? '−' : '') + s.replace('-', ''); },
    probeTypeOf: function (probeId) {
      var m = /_(intext|inworld|outworld)_a(\d+)$/.exec(probeId || '');
      return m ? { intext: 'in_text', inworld: 'in_world', outworld: 'out_of_world' }[m[1]] : null;
    },
    esc: function (s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  };

  window.ArcData = api;
})();
