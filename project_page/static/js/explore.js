/* ArcANE results explorer — a hash router over the released judge outputs.
   Routes: #/  #/novels  #/models  #/n/<novel>  #/n/<novel>/c/<char>
           #/n/<novel>/c/<char>/a/<axis_id>  #/n/<novel>/c/<char>/p/<probe_id>
   View state (view/metric/group/model/…) lives in the hash query so links are shareable.
   Plain ES2020, no build step. Data access goes through window.ArcData (static/js/data.js). */
(function () {
  'use strict';

  var D = window.ArcData;
  var esc = D.esc;
  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

  /* ── constants ─────────────────────────────────────────────── */
  var VIEWS = [['all', 'Overall'], ['in_text', 'In-Scenario'], ['in_world', 'In-World'], ['out_of_world', 'Out-of-World']];
  var METRICS = [['all', 'Mean'], ['apf', 'APF'], ['rpf', 'RPF'], ['rae', 'RAE'], ['ptf', 'PTF']];
  var GROUPS = [['main', 'Main 6'], ['all', '+Added 5']];
  var SLICES = [['validated', 'Validated 4'], ['lowpop', 'Low-popularity'], ['training', 'Training-reference']];
  var SLICE_SHORT = { validated: 'Validated', lowpop: 'Low-popularity', training: 'Training ref.' };
  var PT_FILTER = [['all', 'All types'], ['in_text', 'In-Scenario'], ['in_world', 'In-World'], ['out_of_world', 'Out-of-World']];
  var PICK = [['a', 'A'], ['b', 'B']];
  var ERA = { pre_industrial_agrarian: 'Pre-industrial agrarian', mid_century: 'Mid-century', industrial_early_20th: 'Industrial / early 20th century', speculative_near_future: 'Speculative near future', modern_urban: 'Modern urban', pre_modern_imperial: 'Pre-modern imperial', diasporic_contemporary: 'Diasporic contemporary' };
  var SHORT_TITLE = { 'monte-cristo': 'Monte Cristo', 'benjamin-franklin': 'Benjamin Franklin' };
  var DEFAULTS = { slice: 'validated', view: 'all', metric: 'all', group: 'main', model: 'arcane-32b-dpo', pt: 'all', arc: 'all', q: '', a: 'arcane-32b-dpo:arc', b: 'arcane-32b-dpo:vanilla', pick: 'a' };
  var VALID = { slice: SLICES, view: VIEWS, metric: METRICS, group: GROUPS, pt: PT_FILTER, pick: PICK };   // query keys with a closed option set
  var CARRY = ['view', 'metric', 'group', 'model'];   // state carried across page links
  var SITE = 'ArcANE explorer';
  var CLAMP_CHARS = 1100;       // responses longer than this are clamped with "Show more"

  /* ── state ─────────────────────────────────────────────────── */
  var state = { path: '/', segs: [], q: {} };
  var cat = null;               // catalog.json once loaded
  var bundles = {};             // "novel/char" -> character bundle
  var respCache = {};           // "novel/char/probe" -> responses bundle (or Error)
  var renderId = 0;             // guards async renders against stale routes
  var partial = {};             // named partial updaters registered by the current view
  var sortState = {};           // table data-key -> {col, asc}
  var app = $('#app');

  /* ── hash helpers ──────────────────────────────────────────── */
  function parseHash(h) {
    h = (h == null ? location.hash : h).replace(/^#/, '');
    var qi = h.indexOf('?');
    var path = qi >= 0 ? h.slice(0, qi) : h;
    var query = qi >= 0 ? h.slice(qi + 1) : '';
    if (path.charAt(0) !== '/') path = '/' + path;
    path = path.replace(/\/+$/, '') || '/';
    var q = {};
    query.split('&').forEach(function (kv) {
      if (!kv) return;
      var i = kv.indexOf('=');
      var k = i >= 0 ? kv.slice(0, i) : kv, v = i >= 0 ? kv.slice(i + 1) : '';
      try { q[decodeURIComponent(k)] = decodeURIComponent(v.replace(/\+/g, ' ')); } catch (e) { q[k] = v; }
    });
    return { path: path, segs: path.split('/').filter(Boolean), q: sanitize(q) };
  }
  /* drop query values outside their option set, so a mistyped link falls back to the defaults instead of an empty table */
  function sanitize(q) {
    Object.keys(q).forEach(function (k) {
      var opts = VALID[k];
      if (opts && !opts.some(function (o) { return o[0] === q[k]; })) delete q[k];
      else if ((k === 'a' || k === 'b') && !validSel(q[k])) delete q[k];
    });
    return q;
  }
  function validSel(s) { if (!cat) return true; var p = parseSel(s); return !!modelOf(p.model) && D.MODES.indexOf(p.mode) >= 0; }
  function enc(s) { return encodeURIComponent(s).replace(/%3A/gi, ':').replace(/%2C/gi, ',').replace(/%20/g, '+'); }
  function buildHash(path, q) {
    var parts = [];
    Object.keys(q).forEach(function (k) {
      var v = q[k];
      if (v == null || v === '' || v === DEFAULTS[k]) return;
      parts.push(enc(k) + '=' + enc(v));
    });
    return '#' + path + (parts.length ? '?' + parts.join('&') : '');
  }
  function get(k) { var v = state.q[k]; return v == null || v === '' ? DEFAULTS[k] : v; }
  /* link to another page, carrying the shareable view state along */
  function href(path, patch) {
    var q = {};
    CARRY.forEach(function (k) { if (state.q[k]) q[k] = state.q[k]; });
    if (patch) Object.keys(patch).forEach(function (k) { q[k] = patch[k]; });
    return buildHash(path, q);
  }
  function setQuery(patch, partialName) {
    Object.keys(patch).forEach(function (k) { state.q[k] = patch[k]; });
    history.replaceState(null, '', buildHash(state.path, state.q));
    if (partialName && partial[partialName]) { partial[partialName](); return; }
    render(false);
  }

  /* ── loaders (lazy, cached) ────────────────────────────────── */
  function bundleUrl(nId, cId) { return D.base + 'characters/' + nId + '__' + cId + '.json.gz'; }
  function respUrl(nId, cId, pId) { return D.base + 'responses/' + nId + '__' + cId + '/' + pId + '.json.gz'; }
  function loadBundle(nId, cId) {
    var key = nId + '/' + cId;
    if (bundles[key]) return Promise.resolve(bundles[key]);
    return D.character(nId, cId).then(function (b) { bundles[key] = b; return b; });
  }
  function loadResponses(nId, cId, pId) {
    var key = nId + '/' + cId + '/' + pId;
    if (respCache[key]) return Promise.resolve(respCache[key]);
    return D.responses(nId, cId, pId).then(function (r) { respCache[key] = r; return r; });
  }
  function errorBox(url, e) {
    return '<div class="error">Could not load <code>' + esc(url) + '</code>: ' + esc(e && e.message ? e.message : String(e)) + '</div>';
  }

  /* ── small helpers ─────────────────────────────────────────── */
  function fmt(v) { return D.fmt(v, 1); }
  function fmtInt(n) { return n == null ? '—' : String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
  function liftClass(v) { return v == null ? '' : v > 0 ? 'pos' : v < 0 ? 'neg' : ''; }
  function modelOf(id) { return D.model(cat, id); }
  function modelLabel(id) { return D.modelLabel(cat, id); }
  function modeLabel(id) { return D.modeLabel(cat, id); }
  function ptLabel(id) { return id === 'all' ? 'Overall' : D.ptypeLabel(cat, id); }
  function metricLabel(id) { return id === 'all' ? 'Mean' : id.toUpperCase(); }
  function novelOf(id) { return D.novel(cat, id); }
  function modelsFor(group) { return cat.models.filter(function (m) { return group === 'all' || m.group === 'main'; }); }
  /* model shown in the per-model tables: the requested one if it was evaluated on this table, else the default (or the first
     evaluated model). The hash is rewritten so that the URL, the select and the links agree, and a note explains the switch. */
  function pickModel(table, what) {
    var want = get('model');
    if (table[want]) return { id: want, note: '' };
    var eff = table[DEFAULTS.model] ? DEFAULTS.model : (cat.models.filter(function (m) { return table[m.id]; })[0] || {}).id;
    if (!eff) return { id: want, note: '' };
    var wm = modelOf(want);
    var note = '<p class="tnote" role="status">' + (wm ? esc(wm.label) + ' was not evaluated on this ' + what : 'Unknown model “' + esc(want) + '”') + '; showing ' + esc(modelLabel(eff)) + '.</p>';
    state.q.model = eff;
    history.replaceState(null, '', buildHash(state.path, state.q));
    return { id: eff, note: note };
  }
  function shortAxis(id) { return String(id || '').replace(/^final_/, ''); }
  function shortProbe(pid, charId) { return pid.indexOf(charId + '_') === 0 ? pid.slice(charId.length + 1) : pid; }
  function humanize(s) { return String(s || '').replace(/_/g, ' '); }
  function truncate(s, n) { s = String(s || ''); return s.length > n ? s.slice(0, n - 1).replace(/\s+\S*$/, '') + '…' : s; }
  function sourceLabel(s) { return { both: 'event + state streams', psych_only: 'state stream only', event_only: 'event stream only' }[s] || humanize(s); }
  function critTagLabel(t) { return { valid_axis: 'valid axis', partially_valid_axis: 'partially valid axis', none_axis: 'no valid axis' }[t] || humanize(t); }
  function validationText(arc) { return arc.validation ? arc.validation.valid_votes + '/' + arc.validation.n_annotators + ' annotators' : 'not human-validated'; }
  function validationClass(arc) { if (!arc.validation) return ''; return arc.validation.valid_votes * 2 > arc.validation.n_annotators ? 'good' : 'bad'; }
  function criticsText(arc) { var c = arc.critics || []; return c.filter(function (x) { return x.verdict; }).length + ' of ' + c.length + ' critics'; }
  function maxChapter(bundle) { var m = 1; bundle.arcs.forEach(function (a) { a.phases.forEach(function (ph) { if (ph.chapters && ph.chapters[1] > m) m = ph.chapters[1]; }); }); return m; }
  function parseSel(s) { var i = String(s || '').indexOf(':'); return i > 0 ? { model: s.slice(0, i), mode: s.slice(i + 1) } : { model: s, mode: 'arc' }; }
  function scoreClass(v) { return v == null ? '' : v >= 70 ? 'hi' : v <= 35 ? 'lo' : ''; }
  function scoreChip(label, v) { return '<span class="score ' + scoreClass(v) + '">' + esc(label) + ' <b>' + D.fmt(v, 0) + '</b></span>'; }
  function novelTitle(n) { return SHORT_TITLE[n.id] || n.title; }
  function pluralize(n, one, many) { return fmtInt(n) + ' ' + (n === 1 ? one : (many || one + 's')); }

  /* ── components ────────────────────────────────────────────── */
  function seg(key, options, current, label, partialName) {
    return '<div class="seg" role="group" aria-label="' + esc(label) + '">' + options.map(function (o) {
      return '<button type="button" data-set="' + esc(key) + '" data-val="' + esc(o[0]) + '"' + (partialName ? ' data-partial="' + esc(partialName) + '"' : '') +
        ' aria-pressed="' + (o[0] === current ? 'true' : 'false') + '">' + esc(o[1]) + '</button>';
    }).join('') + '</div>';
  }
  function chips(key, options, current, label, partialName) {
    return '<div class="chips" role="group" aria-label="' + esc(label) + '">' + options.map(function (o) {
      return '<button type="button" data-set="' + esc(key) + '" data-val="' + esc(o[0]) + '"' + (partialName ? ' data-partial="' + esc(partialName) + '"' : '') +
        (o[2] ? ' class="' + esc(o[2]) + '"' : '') + ' aria-pressed="' + (o[0] === current ? 'true' : 'false') + '">' + esc(o[1]) + '</button>';
    }).join('') + '</div>';
  }
  function select(key, options, current, label, partialName) {
    return '<label class="selwrap"><span class="label">' + esc(label) + '</span><select class="sel" data-key="' + esc(key) + '"' + (partialName ? ' data-partial="' + esc(partialName) + '"' : '') + '>' +
      options.map(function (o) { return '<option value="' + esc(o[0]) + '"' + (o[0] === current ? ' selected' : '') + '>' + esc(o[1]) + '</option>'; }).join('') + '</select></label>';
  }
  function searchBox(key, current, placeholder, partialName) {
    return '<label class="selwrap"><span class="visually-hidden">' + esc(placeholder) + '</span><input class="inp" type="search" data-key="' + esc(key) + '" data-partial="' + esc(partialName) + '" value="' + esc(current) + '" placeholder="' + esc(placeholder) + '" aria-label="' + esc(placeholder) + '"></label>';
  }
  function badge(text, cls, title) { return '<span class="badge' + (cls ? ' ' + cls : '') + '"' + (title ? ' title="' + esc(title) + '"' : '') + '>' + esc(text) + '</span>'; }
  function typeBadge(pt) { return '<span class="badge type-' + esc(pt) + '">' + esc(D.ptypeLabel(cat, pt)) + '</span>'; }
  function sliceBadge(slice) { var cls = slice === 'validated' ? 'good' : slice === 'lowpop' ? 'base' : ''; return badge(SLICE_SHORT[slice] || slice, cls, cat.slices[slice] ? cat.slices[slice].label : ''); }
  function roleBadge(central) { return central ? badge('central', 'arc') : badge('supporting', ''); }
  function sourceBadge(src) { return src === 'final_validated' ? badge('human-validated arcs', 'good') : badge('critic-grounded arcs', 'base'); }
  function axisTypeBadge(a) { return a.axis_type === 'relational' ? badge('relational → ' + (a.target_character || '?'), 'base') : badge('intrapersonal', ''); }
  function pill(idx, label) { return '<span class="phase-pill p' + (idx % 6) + '"><span class="n">' + (idx + 1) + '</span>' + esc(label) + '</span>'; }
  function crumbs(items) {
    return '<nav class="crumbs" aria-label="Breadcrumb">' + items.map(function (it, i) {
      var last = i === items.length - 1;
      return (i ? '<span class="sep">/</span>' : '') + (last ? '<span class="cur">' + esc(it.text) + '</span>' : '<a href="' + it.href + '">' + esc(it.text) + '</a>');
    }).join('') + '</nav>';
  }
  function pagehead(title, sub, badges) {
    return '<div class="pagehead"><h1>' + title + '</h1>' + (sub ? '<p class="sub">' + sub + '</p>' : '') + (badges ? '<div class="badges">' + badges + '</div>' : '') + '</div>';
  }
  function stats(items) { return '<div class="stats">' + items.map(function (it) { return '<span><b>' + it[0] + '</b> ' + esc(it[1]) + '</span>'; }).join('') + '</div>'; }
  function th(label, opts) {
    opts = opts || {};
    var cls = [opts.cls || '', opts.sort ? 'sort' : ''].filter(Boolean).join(' ');
    var inner = opts.sort ? '<button type="button" class="sortbtn" title="Sort by ' + esc(label) + '">' + esc(label) + '</button>' : esc(label);
    return '<th scope="col"' + (cls ? ' class="' + cls + '"' : '') + (opts.sort ? ' data-type="' + (opts.num ? 'num' : 'text') + '"' : '') + (opts.title ? ' title="' + esc(opts.title) + '"' : '') + '>' + inner + '</th>';
  }
  function numTd(v, extraCls) { return '<td class="num' + (extraCls ? ' ' + extraCls : '') + '" data-v="' + (v == null ? '' : v) + '">' + fmt(v) + '</td>'; }
  function intTd(v) { return '<td class="num" data-v="' + (v == null ? '' : v) + '">' + fmtInt(v) + '</td>'; }

  /* score controls (view / metric / model group) — state keys view, metric, group */
  function scoreControls(opts) {
    opts = opts || {};
    var html = '<div class="toolbar">' + seg('view', VIEWS, get('view'), 'Probe type view') + seg('metric', METRICS, get('metric'), 'Metric');
    if (opts.group !== false) html += seg('group', GROUPS, get('group'), 'Model group');
    return html + '</div>';
  }
  function countTitle(cnt, pt) {
    if (!cnt) return '';
    var pts = pt === 'all' ? D.PTYPES : [pt], a = 0, b = 0, any = false;
    pts.forEach(function (p) { var c = cnt[p]; if (c) { a += c[0] || 0; b += c[1] || 0; any = true; } });
    return any ? 'n = ' + fmtInt(a) + ' phase records · ' + fmtInt(b) + ' trajectories' : '';
  }

  /* model × mode table (Table-1 layout) for any {model:{mode:{pt:[apf,rpf,rae,ptf]}}} table */
  function modeTable(table, counts) {
    var view = get('view'), metric = get('metric'), group = get('group');
    var models = modelsFor(group);
    var present = models.filter(function (m) { return table && table[m.id]; });
    var absent = models.filter(function (m) { return !table || !table[m.id]; });
    if (!present.length) return '<div class="empty">No scores in this table.</div>';
    var cols;
    if (view === 'all') cols = D.PTYPES.map(function (pt) { return { label: ptLabel(pt), pt: pt, metric: metric }; }).concat([{ label: 'Overall', pt: 'all', metric: metric }]);
    else cols = D.METRICS.map(function (mt) { return { label: mt.toUpperCase(), pt: view, metric: mt }; }).concat([{ label: 'Mean', pt: view, metric: 'all' }]);
    var liftPt = view === 'all' ? 'all' : view, liftMetric = metric;
    var selIdx = -1;
    cols.forEach(function (c, i) { if (c.pt === liftPt && c.metric === liftMetric) selIdx = i; });
    var suffix = view === 'all' && metric !== 'all' ? ' <span class="muted">' + esc(metricLabel(metric)) + '</span>' : '';
    var head = '<tr><th scope="col" class="l">Model</th><th scope="col" class="l">Context mode</th>' + cols.map(function (c, i) {
      return '<th scope="col"' + (i === selIdx ? ' class="selcol" title="Column used for the Arc-lift"' : '') + '>' + esc(c.label) + suffix + '</th>';
    }).join('') + '<th scope="col" title="Arc minus the strongest non-Arc mode on the highlighted column">Arc lift</th></tr>';
    var body = present.map(function (m) {
      var byMode = table[m.id], byCount = counts && counts[m.id];
      var best = cols.map(function (c) {
        var b = null;
        D.MODES.forEach(function (mode) { var v = D.cellValue(byMode[mode], c.pt, c.metric); if (v != null && (b == null || v > b)) b = v; });
        return b;
      });
      var lift = D.arcLift(byMode, liftPt, liftMetric);
      return D.MODES.map(function (mode, mi) {
        var cell = byMode[mode];
        var cls = [mode === 'arc' ? 'arc' : '', mi === 0 ? 'blockstart' : ''].filter(Boolean).join(' ');
        var tds = cols.map(function (c, i) {
          var v = D.cellValue(cell, c.pt, c.metric);
          var isBest = v != null && best[i] != null && Math.abs(v - best[i]) < 1e-9;
          var t = countTitle(byCount && byCount[mode], c.pt);
          return '<td class="num"' + (t ? ' title="' + esc(t) + '"' : '') + '>' + (isBest ? '<b>' + fmt(v) + '</b>' : fmt(v)) + '</td>';
        }).join('');
        var liftTd;
        if (mode === 'arc') {
          liftTd = '<td class="num lift ' + liftClass(lift && lift.lift) + '"' + (lift ? ' title="Arc ' + fmt(lift.arc) + ' − ' + esc(modeLabel(lift.bestMode)) + ' ' + fmt(lift.best) + '"' : '') + '>' +
            (lift ? D.fmtSigned(lift.lift) + '<span class="vs">vs ' + esc(modeLabel(lift.bestMode)) + '</span>' : '—') + '</td>';
        } else liftTd = '<td></td>';
        var modelTh = mi === 0 ? '<th scope="row" class="l model" rowspan="' + D.MODES.length + '">' + esc(m.label) + (m.group === 'added' ? badge('added', '') : '') + '<span class="fam">' + esc(m.family + ' · ' + m.size) + '</span></th>' : '';
        return '<tr' + (cls ? ' class="' + cls + '"' : '') + '>' + modelTh + '<th scope="row" class="l">' + esc(modeLabel(mode)) + '</th>' + tds + liftTd + '</tr>';
      }).join('');
    }).join('');
    var note = absent.length ? '<p class="tnote">Not evaluated on this table: ' + absent.map(function (m) { return esc(m.label); }).join(', ') + '.</p>' : '';
    return '<div class="tablewrap"><table class="data modetable"><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div>' + note;
  }
  function modeTablePanel(title, table, counts, foot, opts) {
    return '<div class="panel"><div class="panel-head"><p class="panel-title">' + title + '</p>' + (opts && opts.headControls ? '<div class="controls">' + opts.headControls + '</div>' : '') + '</div>' +
      '<div class="panel-body">' + scoreControls(opts) + modeTable(table, counts) + '</div>' + (foot ? '<div class="panel-foot">' + foot + '</div>' : '') + '</div>';
  }
  var TABLE_CAPTION = 'Overall = mean of the twelve (probe type × metric) cells; scores are pooled within a novel and novels are averaged with equal weight. Bold = best context mode per model in each column. Arc lift = Arc minus the strongest non-Arc mode on the highlighted column.';
  var HOVER_NOTE = ' Hover a cell for the number of judged records.';   // only where the table carries per-cell counts
  function tableCaption(counts) { return '<p class="caption">' + TABLE_CAPTION + (counts ? HOVER_NOTE : '') + '</p>'; }

  /* timeline strip: segment widths ∝ chapter ranges, relative to the character's max chapter */
  function timeline(a, maxCh) {
    var segs = [], cursor = 1;
    a.phases.forEach(function (ph, i) {
      var s = ph.chapters[0], e = ph.chapters[1];
      if (s > cursor) segs.push({ gap: true, w: s - cursor });
      var start = Math.max(s, cursor);
      segs.push({ i: i, w: Math.max(e - start + 1, 1), t: 'Phase ' + (i + 1) + ': ' + ph.label + ' (ch. ' + s + '–' + e + ')' });
      cursor = Math.max(cursor, e + 1);
    });
    if (cursor <= maxCh) segs.push({ gap: true, w: maxCh - cursor + 1 });
    var desc = a.phases.map(function (ph, i) { return (i + 1) + ' ' + ph.label + ' ch. ' + ph.chapters[0] + '–' + ph.chapters[1]; }).join('; ');
    return '<div class="timeline" role="img" aria-label="' + esc('Phase timeline over chapters 1–' + maxCh + ': ' + desc) + '">' + segs.map(function (sg) {
      return '<span class="' + (sg.gap ? 'gap' : 'p' + (sg.i % 6)) + '" style="flex:' + sg.w + ' 0 0"' + (sg.t ? ' title="' + esc(sg.t) + '"' : '') + '></span>';
    }).join('') + '</div>';
  }
  function arcCard(nId, cId, a, maxCh) {
    var url = href('/n/' + nId + '/c/' + cId + '/a/' + a.axis_id);
    var first = a.phases[0].chapters[0], last = a.phases[a.phases.length - 1].chapters[1];
    return '<div class="card arc-card"><p class="title"><a href="' + url + '">' + esc(a.axis_name) + '</a></p>' +
      '<p class="dim">' + esc(a.dimension_label) + ' · <code class="id">' + esc(shortAxis(a.axis_id)) + '</code></p>' +
      '<div class="tags">' + axisTypeBadge(a) + badge(a.arc_direction, '', 'Arc direction') + badge(validationText(a), validationClass(a), 'Human validity votes') + '</div>' +
      timeline(a, maxCh) + '<div class="tl-lab"><span>ch. ' + first + '</span><span>ch. ' + last + ' (of ' + maxCh + ')</span></div>' +
      '<div class="meta"><span><b>' + a.n_phases + '</b> phases</span><span><b>' + a.n_probes + '</b> probes</span><span>' + esc(criticsText(a)) + '</span></div></div>';
  }

  /* ── table helpers (sorting) ───────────────────────────────── */
  function sortTable(table, col, asc) {
    var tbody = table.tBodies[0]; if (!tbody || !table.tHead) return;
    var hth = table.tHead.rows[0].cells[col]; if (!hth) return;
    var numeric = hth.getAttribute('data-type') === 'num';
    var rows = $$('tr', tbody);
    rows.sort(function (ra, rb) {
      var ca = ra.cells[col], cb = rb.cells[col];
      if (numeric) {
        var va = ca && ca.getAttribute('data-v') !== '' && ca.hasAttribute('data-v') ? parseFloat(ca.getAttribute('data-v')) : NaN;
        var vb = cb && cb.getAttribute('data-v') !== '' && cb.hasAttribute('data-v') ? parseFloat(cb.getAttribute('data-v')) : NaN;
        if (isNaN(va) && isNaN(vb)) return 0;
        if (isNaN(va)) return 1;
        if (isNaN(vb)) return -1;
        return asc ? va - vb : vb - va;
      }
      var ta = ((ca && (ca.getAttribute('data-v') || ca.textContent)) || '').trim().toLowerCase();
      var tb = ((cb && (cb.getAttribute('data-v') || cb.textContent)) || '').trim().toLowerCase();
      return asc ? ta.localeCompare(tb) : tb.localeCompare(ta);
    });
    rows.forEach(function (r) { tbody.appendChild(r); });
    $$('th', table.tHead).forEach(function (h) { h.classList.remove('sorted', 'asc'); h.removeAttribute('aria-sort'); });
    hth.classList.add('sorted'); if (asc) hth.classList.add('asc');
    hth.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
  }
  function onSortClick(hth) {
    var table = hth.closest('table'); if (!table) return;
    var col = Array.prototype.indexOf.call(hth.parentNode.children, hth);
    var numeric = hth.getAttribute('data-type') === 'num';
    var key = table.getAttribute('data-key') || '';
    var cur = sortState[key];
    var asc = cur && cur.col === col ? !cur.asc : !numeric;
    sortState[key] = { col: col, asc: asc };
    sortTable(table, col, asc);
  }
  function applySorts(root) {
    $$('table[data-key]', root).forEach(function (t) { var s = sortState[t.getAttribute('data-key')]; if (s) sortTable(t, s.col, s.asc); });
  }

  /* ── views ─────────────────────────────────────────────────── */

  /* Overview: #/  #/novels  #/models */
  function viewOverview(err) {
    var slice = get('slice'), sl = cat.slices[slice], t = cat.totals;
    var judge = cat.models.filter(function (m) { return m.key === cat.judge; })[0];   // catalog.judge is the judge's API key
    var html = '';
    if (err) html += '<div class="error notice" role="alert">' + esc(err) + '</div>';
    html += pagehead('ArcANE results explorer', 'Per-novel, per-character, per-arc and per-probe judge scores of role-playing language agents under six context modes, with every archived response. Scores are on a 1–100 scale; the judge is ' + esc(judge ? judge.label : cat.judge) + '.');
    html += '<div class="callout neutral">All numbers on these pages are <b>computed from the released judge outputs</b> (4 of 5 validated novels; <i>Harry Potter</i> is not distributed), so the validated-slice aggregates differ slightly from Table 2 of the paper. Two low-popularity novels (memorization control) and two training-reference novels (in the SFT/DPO pool; reference only) are also included.</div>';
    html += stats([[fmtInt(t.novels), 'novels'], [fmtInt(t.characters), 'characters'], [fmtInt(t.arcs), 'Character Arcs'], [fmtInt(t.probes), 'probes'], [fmtInt(t.phase_slots), 'phase slots'], [fmtInt(t.judge_records), 'phase-level judge records'], [fmtInt(t.trajectory_records), 'trajectory (PTF) records'], [fmtInt(t.responses), 'archived responses'], [fmtInt(t.responses_missing), 'responses not archived']]);

    html += '<section class="section" id="results"><h2>Model × context mode</h2><p class="lede">Rows are the six main models (plus five added baselines on request) under the six context modes; the highlighted rows are Arc, the context mode proposed in the paper.</p>' +
      modeTablePanel(esc(sl.label), sl.table, null, esc(sl.desc), { headControls: seg('slice', SLICES, slice, 'Slice') }) + tableCaption(null) + '</section>';

    html += '<section class="section" id="novels"><h2>Novels</h2><p class="lede">Each card shows the Overall Arc lift (Arc minus the strongest non-Arc mode) for ArcANE-32B-DPO and DeepSeek-V4-Pro on that novel.</p>';
    ['validated', 'lowpop', 'training'].forEach(function (s) {
      var sdef = cat.slices[s];
      html += '<h3>' + esc(sdef.label) + '</h3><p class="tnote">' + esc(sdef.desc) + '</p><div class="cards novels">' + sdef.novels.map(function (nid) { var n = novelOf(nid); return n ? novelCard(n) : ''; }).join('') + '</div>';
    });
    html += '</section>';

    html += '<section class="section" id="models"><h2>Models</h2><p class="lede">The six main models were run under all six context modes on every probe and their responses are archived. The five added models (the paper\'s appendix table on added role-playing models) were scored on the validated and training-reference novels only; their responses are not archived.</p>' +
      '<div class="tablewrap"><table class="data" data-key="models"><thead><tr>' + th('Model', { cls: 'l', sort: true }) + th('Family', { cls: 'l', sort: true }) + th('Size', { cls: 'l', sort: true }) + th('Group', { cls: 'l', sort: true }) + th('Responses archived', { cls: 'l', sort: true }) + th('Checkpoint / API id', { cls: 'l' }) + '</tr></thead><tbody>' +
      cat.models.map(function (m) {
        return '<tr><th scope="row" class="l name">' + esc(m.label) + '</th><td class="l">' + esc(m.family) + '</td><td class="l">' + esc(m.size) + '</td><td class="l">' + (m.group === 'main' ? badge('main', 'arc') : badge('added', '')) + '</td><td class="l" data-v="' + (m.responses ? 1 : 0) + '">' + (m.responses ? 'yes' : 'no') + '</td><td class="l mono">' + esc(m.key) + '</td></tr>';
      }).join('') + '</tbody></table></div>' +
      '<p class="footnote">ArcANE-8B/32B-DPO are the paper\'s trained models (SFT → DPO from Qwen3-8B/32B). The ArcANE-32B-RLVR checkpoint was evaluated on a separate judge run and is not part of this release.</p></section>';
    return { title: SITE, html: html };
  }
  function novelCard(n) {
    var l1 = D.arcLift(n.table['arcane-32b-dpo'], 'all', 'all'), l2 = D.arcLift(n.table['ds-v4-pro'], 'all', 'all');
    function liftSpan(label, l) { return '<span>' + esc(label) + ' <b class="' + liftClass(l && l.lift) + '">' + D.fmtSigned(l && l.lift) + '</b></span>'; }
    return '<a class="card" href="' + href('/n/' + n.id) + '"><p class="title">' + esc(n.title) + sliceBadge(n.slice) + '</p><p class="sub">' + esc(n.author + ', ' + n.year) + '</p>' +
      '<div class="meta"><span><b>' + n.n_characters + '</b> characters</span><span><b>' + n.n_arcs + '</b> arcs</span><span><b>' + fmtInt(n.n_probes) + '</b> probes</span></div>' +
      '<div class="lifts"><span class="muted">Arc lift</span>' + liftSpan('ArcANE-32B-DPO', l1) + liftSpan('DS-V4-Pro', l2) + '</div></a>';
  }

  /* Novel: #/n/<novel> */
  function viewNovel(n) {
    var sl = cat.slices[n.slice];
    var modelsHere = cat.models.filter(function (m) { return n.table[m.id]; });
    var pm = pickModel(n.table, 'novel'), model = pm.id;
    var slots = 0, resp = 0;
    n.characters.forEach(function (c) { slots += c.n_phase_slots || 0; resp += c.responses || 0; });
    var html = crumbs([{ href: href('/'), text: 'Explorer' }, { text: n.title }]);
    html += pagehead(esc(n.title), esc(n.author + ', ' + n.year) + (n.gutenberg ? ' · <a href="https://www.gutenberg.org/ebooks/' + esc(n.gutenberg) + '">Project Gutenberg #' + esc(n.gutenberg) + '</a>' : ''),
      sliceBadge(n.slice) + '<span>' + esc(sl ? sl.desc : '') + '</span>');
    html += stats([[n.n_characters, 'characters'], [n.n_arcs, 'Character Arcs'], [fmtInt(n.n_probes), 'probes'], [fmtInt(slots), 'phase slots'], [fmtInt(resp), 'archived responses']]);

    html += '<section class="section" id="results"><h2>Model × context mode</h2>' + modeTablePanel(esc(n.title) + ' · all characters', n.table, n.counts, null) + tableCaption(n.counts) + '</section>';

    /* characters table for the selected model */
    var pt = get('view'), metric = get('metric');
    html += '<section class="section" id="characters"><h2>Characters</h2><p class="lede">Scores for one model under Vanilla, its strongest non-Arc mode and Arc, on the column selected above (' + esc(ptLabel(pt)) + ' · ' + esc(metricLabel(metric)) + ').</p>' +
      '<div class="toolbar">' + select('model', modelsHere.map(function (m) { return [m.id, m.label]; }), model, 'Model') + '</div>' + pm.note +
      '<div class="tablewrap"><table class="data" data-key="chars:' + esc(n.id) + '"><thead><tr>' + th('Character', { cls: 'l', sort: true }) + th('Role', { cls: 'l' }) + th('Arcs', { sort: true, num: true }) + th('Probes', { sort: true, num: true }) + th('Vanilla', { sort: true, num: true }) + th('Best non-Arc', { sort: true, num: true }) + th('Arc', { sort: true, num: true, cls: 'selcol' }) + th('Arc lift', { sort: true, num: true }) + '</tr></thead><tbody>' +
      n.characters.map(function (c) {
        var byMode = c.table[model] || {};
        var van = D.cellValue(byMode.vanilla, pt, metric), lift = D.arcLift(byMode, pt, metric);
        var url = href('/n/' + n.id + '/c/' + c.id);
        return '<tr class="link" data-href="' + url + '"><th scope="row" class="l name"><a href="' + url + '">' + esc(c.name) + '</a></th><td class="l">' + roleBadge(c.central) + '</td>' + intTd(c.n_arcs) + intTd(c.n_probes) + numTd(van) +
          '<td class="num" data-v="' + (lift ? lift.best : '') + '">' + (lift ? fmt(lift.best) + ' <span class="muted small">' + esc(modeLabel(lift.bestMode)) + '</span>' : '—') + '</td>' +
          '<td class="num" data-v="' + (lift ? lift.arc : '') + '"><b>' + (lift ? fmt(lift.arc) : fmt(D.cellValue(byMode.arc, pt, metric))) + '</b></td>' +
          '<td class="num lift ' + liftClass(lift && lift.lift) + '" data-v="' + (lift ? lift.lift : '') + '">' + D.fmtSigned(lift && lift.lift) + '</td></tr>';
      }).join('') + '</tbody></table></div></section>';

    /* arcs across characters, searchable */
    var arcs = [];
    n.characters.forEach(function (c) { c.arcs.forEach(function (a) { arcs.push({ c: c, a: a }); }); });
    html += '<section class="section" id="arcs"><h2>Character Arcs</h2><div class="toolbar">' + searchBox('q', get('q'), 'Search arcs (axis, dimension, character, type)', 'arcs') + '<span class="count" id="arcs-count"></span></div>' +
      '<div class="tablewrap tall"><table class="data" data-key="arcs:' + esc(n.id) + '"><thead><tr>' + th('Axis', { cls: 'l', sort: true }) + th('Character', { cls: 'l', sort: true }) + th('Type', { cls: 'l', sort: true }) + th('Direction', { cls: 'l', sort: true }) + th('Phases', { sort: true, num: true }) + th('Probes', { sort: true, num: true }) + '</tr></thead><tbody id="arcs-rows"></tbody></table></div></section>';

    function arcRows() {
      var q = get('q').trim().toLowerCase();
      var rows = arcs.filter(function (x) {
        if (!q) return true;
        var hay = [x.a.axis_name, x.a.dimension_label, x.c.name, x.a.axis_type, x.a.target_character || '', x.a.arc_direction, x.a.axis_id].join(' ').toLowerCase();
        return hay.indexOf(q) >= 0;
      });
      var body = rows.map(function (x) {
        var url = href('/n/' + n.id + '/c/' + x.c.id + '/a/' + x.a.axis_id);
        return '<tr class="link" data-href="' + url + '"><th scope="row" class="l axis" data-v="' + esc(x.a.axis_name) + '"><a href="' + url + '">' + esc(x.a.axis_name) + '</a><span class="sub">' + esc(x.a.dimension_label) + ' · ' + esc(shortAxis(x.a.axis_id)) + '</span></th>' +
          '<td class="l"><a href="' + href('/n/' + n.id + '/c/' + x.c.id) + '">' + esc(x.c.name) + '</a></td><td class="l" data-v="' + esc(x.a.axis_type) + '">' + axisTypeBadge(x.a) + '</td><td class="l">' + esc(x.a.arc_direction) + '</td>' + intTd(x.a.n_phases) + intTd(x.a.n_probes) + '</tr>';
      }).join('');
      return { body: body || '<tr><td colspan="6" class="l"><div class="empty inline">No arcs match “' + esc(q) + '”.</div></td></tr>', count: rows.length + ' of ' + arcs.length + ' arcs' };
    }
    partial.arcs = function () {
      var r = arcRows(); var tb = $('#arcs-rows'), cnt = $('#arcs-count');
      if (tb) { tb.innerHTML = r.body; applySorts(tb.closest('table').parentNode); }
      if (cnt) cnt.textContent = r.count;
      markScrollAll();
    };
    return { title: n.title + ' · ' + SITE, html: html, after: partial.arcs };
  }

  /* Character: #/n/<novel>/c/<char> */
  function viewCharacter(n, cm, b) {
    var cId = cm.id, maxCh = maxChapter(b);
    var modelsHere = cat.models.filter(function (m) { return b.table[m.id]; });
    var pm = pickModel(b.table, 'character'), model = pm.id;
    var html = crumbs([{ href: href('/'), text: 'Explorer' }, { href: href('/n/' + n.id), text: novelTitle(n) }, { text: cm.name }]);
    html += pagehead(esc(cm.name), '<a href="' + href('/n/' + n.id) + '">' + esc(n.title) + '</a> · ' + esc(n.author + ', ' + n.year),
      roleBadge(cm.central) + sourceBadge(b.character.arc_source) + sliceBadge(n.slice) + '<span>' + (b.character.arc_source === 'final_validated' ? 'Arcs passed 2-of-3 human validity; critics are LLM literary critics.' : 'Arcs were validated by three LLM literary critics only (no human validation on this slice).') + '</span>');
    html += stats([[b.arcs.length, 'Character Arcs'], [b.probes.length, 'probes'], [fmtInt(cm.n_phase_slots), 'phase slots'], [fmtInt(cm.responses), 'archived responses']]);

    html += '<section class="section" id="arcs"><h2>Character Arcs</h2><p class="lede">One arc per psychological axis. The strip shows where each phase sits in the novel (chapter ranges relative to this character\'s last chapter; grey = chapters outside any phase).</p><div class="arc-cards">' +
      b.arcs.map(function (a) { return arcCard(n.id, cId, a, maxCh); }).join('') + '</div></section>';

    html += '<section class="section" id="results"><h2>Model × context mode</h2>' + modeTablePanel(esc(cm.name) + ' · all ' + b.probes.length + ' probes', b.table, b.counts, null) + tableCaption(b.counts) + '</section>';

    /* probes list */
    var arcOpts = [['all', 'All arcs']].concat(b.arcs.map(function (a) { return [a.axis_id, shortAxis(a.axis_id) + ' · ' + truncate(a.axis_name, 44)]; }));
    var arcById = {}; b.arcs.forEach(function (a) { arcById[a.axis_id] = a; });
    html += '<section class="section" id="probes"><h2>Probes</h2><p class="lede">Every probe for this character with its per-probe Overall (mean of the per-phase APF/RPF/RAE average, weighted 3, and PTF, weighted 1) for one model under Arc and under Vanilla; Δ = Arc minus Vanilla.</p>' +
      '<div class="toolbar">' + chips('pt', PT_FILTER, get('pt'), 'Probe type', 'probes') + select('arc', arcOpts, get('arc'), 'Arc', 'probes') + select('model', modelsHere.map(function (m) { return [m.id, m.label]; }), model, 'Model', 'probes') + searchBox('q', get('q'), 'Search scenario / question / id', 'probes') + '<span class="count" id="probes-count"></span></div>' + pm.note +
      '<div class="tablewrap tall"><table class="data probes" data-key="probes:' + esc(cId) + '"><thead><tr>' + th('Probe', { cls: 'l', sort: true }) + th('Type', { cls: 'l', sort: true }) + th('Arc', { sort: true, num: true, cls: 'selcol' }) + th('Vanilla', { sort: true, num: true }) + th('Δ', { sort: true, num: true, title: 'Arc minus Vanilla' }) + th('Anchor phase', { cls: 'l', sort: true }) + th('Axis', { cls: 'l', sort: true }) + th('Scenario', { cls: 'l' }) + '</tr></thead><tbody id="probes-rows"></tbody></table></div></section>';

    function probeRows() {
      var q = get('q').trim().toLowerCase(), pt = get('pt'), arc = get('arc'), mdl = get('model');
      if (!b.table[mdl]) mdl = model;
      if (arc !== 'all' && !arcById[arc]) arc = 'all';   // unknown axis in the hash: the select already shows "All arcs"
      var rows = b.probes.filter(function (p) {
        if (pt !== 'all' && p.probe_type !== pt) return false;
        if (arc !== 'all' && p.axis_id !== arc) return false;
        if (q && [p.probe_id, p.scenario, p.question, p.anchor_phase_label].join(' ').toLowerCase().indexOf(q) < 0) return false;
        return true;
      });
      var body = rows.map(function (p) {
        var sc = p.scores[mdl] || {};
        var va = D.probeOverall(sc.arc), vv = D.probeOverall(sc.vanilla);
        var d = va != null && vv != null ? va - vv : null;
        var url = href('/n/' + n.id + '/c/' + cId + '/p/' + p.probe_id);
        var a = arcById[p.axis_id];
        return '<tr class="link" data-href="' + url + '"><th scope="row" class="l mono" data-v="' + esc(p.probe_id) + '"><a href="' + url + '">' + esc(shortProbe(p.probe_id, cId)) + '</a></th><td class="l" data-v="' + esc(p.probe_type) + '">' + typeBadge(p.probe_type) + '</td>' +
          numTd(va) + numTd(vv) + '<td class="num ' + liftClass(d) + '" data-v="' + (d == null ? '' : d) + '">' + D.fmtSigned(d) + '</td>' +
          '<td class="l anchor" data-v="' + p.anchor_phase_idx + '">' + pill(p.anchor_phase_idx, p.anchor_phase_label) + '</td>' +
          '<td class="l" data-v="' + esc(p.axis_id) + '"><a href="' + href('/n/' + n.id + '/c/' + cId + '/a/' + p.axis_id) + '" title="' + esc(a ? a.axis_name : p.axis_id) + '">' + esc(shortAxis(p.axis_id)) + '</a></td>' +
          '<td class="l snippet" title="' + esc(p.scenario) + '">' + esc(truncate(p.scenario, 120)) + '</td></tr>';
      }).join('');
      return { body: body || '<tr><td colspan="8" class="l"><div class="empty inline">No probes match the current filters.</div></td></tr>', count: rows.length + ' of ' + b.probes.length + ' probes' };
    }
    partial.probes = function () {
      var r = probeRows(); var tb = $('#probes-rows'), cnt = $('#probes-count');
      if (tb) { tb.innerHTML = r.body; applySorts(tb.closest('table').parentNode); }
      if (cnt) cnt.textContent = r.count;
      markScrollAll();
      $$('button[data-set="pt"]', app).forEach(function (bt) { bt.setAttribute('aria-pressed', bt.getAttribute('data-val') === get('pt') ? 'true' : 'false'); });
    };
    return { title: cm.name + ' · ' + SITE, html: html, after: partial.probes };
  }

  /* Arc: #/n/<novel>/c/<char>/a/<axis_id> */
  function viewArc(n, cm, b, a) {
    var cId = cm.id, maxCh = maxChapter(b);
    var base = '/n/' + n.id + '/c/' + cId;
    var html = crumbs([{ href: href('/'), text: 'Explorer' }, { href: href('/n/' + n.id), text: novelTitle(n) }, { href: href(base), text: cm.name }, { text: shortAxis(a.axis_id) }]);
    var nCrit = (a.critics || []).length, nYes = (a.critics || []).filter(function (c) { return c.verdict; }).length;
    html += pagehead(esc(a.axis_name), '<a href="' + href(base) + '">' + esc(cm.name) + '</a> · <a href="' + href('/n/' + n.id) + '">' + esc(n.title) + '</a> · <code class="id">' + esc(a.axis_id) + '</code>',
      axisTypeBadge(a) + badge(a.dimension_label, '', 'Psychological dimension') + badge('direction: ' + a.arc_direction) + badge('confidence: ' + a.confidence) + badge(sourceLabel(a.source), '', 'Which candidate stream(s) proposed the axis') +
      badge(validationText(a), validationClass(a), 'Human validity votes') + badge(nYes + ' of ' + nCrit + ' critics', nYes === nCrit ? 'good' : nYes ? '' : 'bad', 'LLM critic verdicts') + badge(critTagLabel(a.critic_tag), a.critic_tag === 'valid_axis' ? 'good' : a.critic_tag === 'none_axis' ? 'bad' : ''));
    html += '<div class="kv"><span class="k">Evidence</span><span class="v">' + esc(a.evidence_summary) + '</span>' + (a.abstract_axis ? '<span class="k">Abstract axis</span><span class="v">' + esc(a.abstract_axis) + '</span>' : '') + '</div>';

    html += '<section class="section"><h2>Poles</h2><div class="poles"><div class="pole"><div class="lab">Start</div>' + esc(a.pole_start) + '</div><div class="arrow" aria-hidden="true">→</div><div class="pole"><div class="lab">End</div>' + esc(a.pole_end) + '</div></div></section>';

    html += '<section class="section"><h2>Phases</h2>' + timeline(a, maxCh) + '<p class="tnote">Chapter positions relative to ' + esc(cm.name) + '\'s last chapter (' + maxCh + ').</p><div class="phases">' +
      a.phases.map(function (ph, i) {
        var stage = [ph.life_stage ? humanize(ph.life_stage) : '', ph.approx_age ? 'age ' + ph.approx_age : ''].filter(Boolean).join(' · ');
        return '<div class="phase p' + (i % 6) + '"><div class="head">' + pill(i, ph.label) + '<span class="ch">Chapters ' + ph.chapters[0] + '–' + ph.chapters[1] + (stage ? ' · ' + esc(stage) : '') + '</span></div><div><div>' + esc(ph.description) + '</div>' +
          (ph.key_moments && ph.key_moments.length ? '<ul>' + ph.key_moments.map(function (k) { return '<li>' + esc(k) + '</li>'; }).join('') + '</ul>' : '') + '</div></div>';
      }).join('') + '</div></section>';

    html += '<section class="section"><h2>Decision variable</h2><div class="callout">' + esc(a.decision_variable) + '</div>' +
      '<details class="fold"><summary>Phase contrasts and abstract phases' + badge((a.phase_contrasts || []).length + ' contrasts') + '</summary><div class="body">' +
      (a.phase_contrasts || []).map(function (pc) {
        var from = a.phases[pc.from_phase_idx], to = a.phases[pc.to_phase_idx];
        return '<div class="contrast"><span class="pills">' + pill(pc.from_phase_idx, from ? from.label : '') + '<span class="arrow">→</span>' + pill(pc.to_phase_idx, to ? to.label : '') + '</span><span>' + esc(pc.contrast) + '</span></div>';
      }).join('') +
      (a.abstract_phases && a.abstract_phases.length ? '<h3 class="inset">Abstract phases (used for Out-of-World transposition)</h3><ol class="plain">' + a.abstract_phases.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('') + '</ol>' : '') + '</div></details>';
    html += '<details class="fold"><summary>Critic panel' + badge(nYes + ' of ' + nCrit + ' critics accept', nYes === nCrit ? 'good' : nYes ? '' : 'bad') + '</summary><div class="body">' +
      (a.critics || []).map(function (c) {
        return '<div class="critic"><div class="who">' + esc(c.label) + (c.verdict ? badge('valid', 'good') : badge('not valid', 'bad')) + '</div><p>' + esc(c.reasoning) + '</p>' +
          (c.citations && c.citations.length ? '<ul>' + c.citations.map(function (ci) { return '<li>' + esc([ci.author, ci.title, ci.publication, ci.year].filter(function (x) { return x != null && x !== ''; }).join(', ')) + '</li>'; }).join('') + '</ul>' : '') + '</div>';
      }).join('') + '</div></details>';
    var wp = a.weak_pairs || [];
    if (wp.length) {
      var byProbe = {}, sim = 0, amb = 0;
      wp.forEach(function (w) { byProbe[w.probe_id] = 1; if (w.separation === 'similar') sim++; else amb++; });
      html += '<p class="tnote">Weakly separated phase pairs flagged by the discriminability check (Q-Discrim): ' + wp.length + ' pairs across ' + Object.keys(byProbe).length + ' probes (' + sim + ' “similar”, ' + amb + ' “ambiguous”). Such pairs are kept but note that neighbouring references may be hard to tell apart.</p>';
    } else html += '<p class="tnote">No weakly separated phase pairs were flagged by the discriminability check for this arc.</p>';
    html += '</section>';

    html += '<section class="section" id="results"><h2>Model × context mode</h2>' + modeTablePanel(esc(shortAxis(a.axis_id)) + ' · ' + a.n_probes + ' probes', a.table, null, null) + tableCaption(null) + '</section>';

    /* probe cards grouped by type */
    var probes = b.probes.filter(function (p) { return p.axis_id === a.axis_id; });
    var mainModels = cat.models.filter(function (m) { return m.group === 'main'; });
    html += '<section class="section" id="probes"><h2>Probes</h2><p class="lede">The same scenario and question are posed at every phase; each card shows the per-probe Overall of the six main models under Arc and under Vanilla.</p>';
    D.PTYPES.forEach(function (pt) {
      var ps = probes.filter(function (p) { return p.probe_type === pt; });
      if (!ps.length) return;
      html += '<h3>' + esc(D.ptypeLabel(cat, pt)) + ' <span class="muted small">' + esc(cat.probe_types.filter(function (x) { return x.id === pt; })[0].desc) + '</span></h3><div class="probe-cards">' +
        ps.map(function (p) {
          var url = href(base + '/p/' + p.probe_id);
          var strip = '<div class="strip" role="group" aria-label="Per-probe Overall by model"><span class="h">Model</span><span class="h arc">Arc</span><span></span><span class="h base">Vanilla</span><span></span>' + mainModels.map(function (m) {
            var sc = p.scores[m.id] || {};
            var va = D.probeOverall(sc.arc), vv = D.probeOverall(sc.vanilla);
            return '<span class="m">' + esc(m.short) + '</span><span class="bar arc"><i style="width:' + (va == null ? 0 : va) + '%"></i></span><span class="v">' + D.fmt(va, 0) + '</span><span class="bar"><i style="width:' + (vv == null ? 0 : vv) + '%"></i></span><span class="v">' + D.fmt(vv, 0) + '</span>';
          }).join('') + '</div>';
          return '<div class="card probe-card"><div class="title"><a href="' + url + '">' + esc(shortProbe(p.probe_id, cId)) + '</a>' + pill(p.anchor_phase_idx, p.anchor_phase_label) + (p.era_label ? badge(ERA[p.era_label] || humanize(p.era_label)) : '') + '</div>' +
            '<p class="scenario">' + esc(p.scenario) + '</p><p class="question">' + esc(p.question) + '</p>' + strip + '</div>';
        }).join('') + '</div>';
    });
    if (!probes.length) html += '<div class="empty">No probes were generated for this arc.</div>';
    html += '</section>';
    return { title: shortAxis(a.axis_id) + ' · ' + cm.name + ' · ' + SITE, html: html };
  }

  /* Probe: #/n/<novel>/c/<char>/p/<probe_id> */
  function viewProbe(n, cm, b, p, idx) {
    var cId = cm.id, base = '/n/' + n.id + '/c/' + cId;
    var a = b.arcs.filter(function (x) { return x.axis_id === p.axis_id; })[0];
    var html = crumbs([{ href: href('/'), text: 'Explorer' }, { href: href('/n/' + n.id), text: novelTitle(n) }, { href: href(base), text: cm.name }, { href: href(base + '/a/' + p.axis_id), text: shortAxis(p.axis_id) }, { text: shortProbe(p.probe_id, cId) }]);
    var check = p.check && (p.check.world || p.check.anchor), checkKind = p.check && p.check.world ? 'World check' : 'Anchor check';
    var badges = typeBadge(p.probe_type) + '<span>' + pill(p.anchor_phase_idx, p.anchor_phase_label) + ' anchor · query chapter ' + esc(p.anchor_query_chapter) + '</span>';
    if (p.era_label) badges += badge(ERA[p.era_label] || humanize(p.era_label), 'arc', p.era_description || '');
    if (check) badges += badge(checkKind + ': ' + check.verdict, check.verdict === 'pass' ? 'good' : 'bad', check.note || '');
    html += pagehead('<code class="id">' + esc(p.probe_id) + '</code>', 'Probe on <a href="' + href(base + '/a/' + p.axis_id) + '">' + esc(a ? a.axis_name : p.axis_id) + '</a> · <a href="' + href(base) + '">' + esc(cm.name) + '</a> · <a href="' + href('/n/' + n.id) + '">' + esc(n.title) + '</a>', badges);
    if (p.era_description) html += '<p class="checknote"><b>Era:</b> ' + esc(p.era_description) + '</p>';
    if (p.anchor_source) html += '<p class="checknote"><b>Anchor event (ch. ' + esc(p.anchor_source.chapter) + '):</b> ' + esc(p.anchor_source.event) + '</p>';
    if (check && check.note) html += '<p class="checknote"><b>' + esc(checkKind) + ':</b> ' + esc(check.note) + '</p>';
    var wps = (p.check && p.check.weak_pairs) || [];
    if (wps.length) html += '<p class="checknote"><b>Weakly separated pairs:</b> ' + wps.map(function (w) { return 'phase ' + (w.from + 1) + '→' + (w.to + 1) + ' (' + esc(w.separation) + ')'; }).join(', ') + '</p>';
    var prev = idx > 0 ? b.probes[idx - 1] : null, next = idx < b.probes.length - 1 ? b.probes[idx + 1] : null;
    html += '<div class="probe-nav"><span>' + (prev ? '<a href="' + href(base + '/p/' + prev.probe_id, { a: state.q.a, b: state.q.b, pick: state.q.pick }) + '" rel="prev">← ' + esc(shortProbe(prev.probe_id, cId)) + '</a>' : '') + '</span><span class="mid">' + (idx + 1) + ' / ' + b.probes.length + '</span><span>' + (next ? '<a href="' + href(base + '/p/' + next.probe_id, { a: state.q.a, b: state.q.b, pick: state.q.pick }) + '" rel="next">' + esc(shortProbe(next.probe_id, cId)) + ' →</a>' : '') + '</span></div>';

    html += '<div class="probe-box"><p class="lab">Scenario</p><p class="scenario">' + esc(p.scenario) + '</p><p class="lab">Question</p><p class="question">' + esc(p.question) + '</p></div>';

    /* phase references */
    html += '<section class="section" id="refs"><h2>Phase references</h2><p class="lede">One reference per phase: the character\'s action, speech and thought at that point of the arc. Badges are the probe-validation verdicts (Q-Voice, Q-PhaseFit) and the typicality of the reference; greyed phases are unavailable (the character is absent or the situation cannot arise).</p><div class="refs">' +
      p.refs.map(function (r) {
        var i = r.phase_idx;
        var bd = [
          badge(humanize(r.typicality || ''), r.typicality === 'plausible_tail' ? 'base' : '', 'Typicality of the reference behaviour'),
          badge('voice: ' + r.voice, r.voice === 'pass' ? 'good' : 'bad', 'Q-Voice verdict'),
          badge('phase fit: ' + humanize(r.phase_fit), r.phase_fit === 'pass' ? 'good' : r.phase_fit === 'off_phase' ? 'bad' : '', 'Q-PhaseFit verdict'),
          r.retry_count ? badge('retries: ' + r.retry_count) : '',
          r.unavailable ? badge('unavailable', 'bad') : ''
        ].join('');
        return '<div class="ref p' + (i % 6) + (r.unavailable ? ' unavailable' : '') + '"><div class="head">' + pill(i, r.phase_label) + '<span class="ch">query chapter ' + esc(r.query_chapter) + '</span><span class="badges">' + bd + '</span></div>' +
          '<div class="field"><span class="k">Action</span><span class="v">' + esc(r.action) + '</span></div>' +
          (r.speech ? '<div class="field"><span class="k">Speech</span><span class="v speech">' + esc(r.speech) + '</span></div>' : '') +
          '<div class="field thought"><span class="k">Thought</span><span class="v">' + esc(r.thought) + '</span></div></div>';
      }).join('') + '</div></section>';

    /* score matrix */
    html += '<section class="section" id="scores"><h2>Score matrix</h2><p class="lede">Per-probe Overall = (3 × mean of the per-phase APF/RPF/RAE averages + PTF) / 4. Hover for the two parts; click a cell to load that (model, context mode) into the response viewer below.</p>' +
      '<div class="toolbar"><span class="label">Click assigns to</span>' + seg('pick', PICK, get('pick'), 'Cell click assigns to viewer slot') + '</div>' + matrix(p) + '</section>';

    /* response viewer */
    var sa = parseSel(get('a')), sb = parseSel(get('b'));
    var modelOpts = cat.models.map(function (m) { return [m.id, m.label + (m.responses ? '' : ' (responses not archived)')]; });
    var modeOpts = D.MODES.map(function (m) { return [m, modeLabel(m)]; });
    html += '<section class="section" id="responses"><h2>Responses</h2><div class="panel"><div class="panel-head"><p class="panel-title">Compare two (model, context mode) cells phase by phase</p><div class="selectors">' +
      selectorRow('a', sa, modelOpts, modeOpts) + selectorRow('b', sb, modelOpts, modeOpts) + '</div></div><div class="panel-body" id="viewer"><p class="loading">Loading responses…</p></div></div></section>';

    var out = { title: shortProbe(p.probe_id, cId) + ' · ' + cm.name + ' · ' + SITE, html: html };
    out.after = function () { fillViewer(n, cm, p, sa, sb); };
    return out;
  }
  function selectorRow(slot, sel, modelOpts, modeOpts) {
    return '<div class="selector"><span class="who" aria-hidden="true">' + slot.toUpperCase() + '</span><label class="selwrap"><span class="visually-hidden">Model ' + slot.toUpperCase() + '</span><select class="sel" data-key="' + slot + '" data-part="model">' +
      modelOpts.map(function (o) { return '<option value="' + esc(o[0]) + '"' + (o[0] === sel.model ? ' selected' : '') + '>' + esc(o[1]) + '</option>'; }).join('') + '</select></label>' +
      '<label class="selwrap"><span class="visually-hidden">Context mode ' + slot.toUpperCase() + '</span><select class="sel" data-key="' + slot + '" data-part="mode">' +
      modeOpts.map(function (o) { return '<option value="' + esc(o[0]) + '"' + (o[0] === sel.mode ? ' selected' : '') + '>' + esc(o[1]) + '</option>'; }).join('') + '</select></label></div>';
  }
  function matrix(p) {
    var sa = parseSel(get('a')), sb = parseSel(get('b'));
    var models = cat.models.filter(function (m) { return p.scores && p.scores[m.id]; });
    var missing = cat.models.filter(function (m) { return !p.scores || !p.scores[m.id]; });
    if (!models.length) {
      var allUnavailable = p.refs.length && p.refs.every(function (r) { return r.unavailable; });
      return '<div class="empty">No judge scores are available for this probe' + (allUnavailable ? ': every phase reference is marked unavailable, so the probe was excluded from evaluation.' : '.') + '</div>';
    }
    var head = '<tr><th scope="col" class="l">Model</th>' + D.MODES.map(function (mode) { return '<th scope="col"' + (mode === 'arc' ? ' class="arccol"' : '') + '>' + esc(modeLabel(mode)) + '</th>'; }).join('') + '</tr>';
    var body = models.map(function (m) {
      return '<tr><th scope="row" class="l model' + (m.group === 'added' ? ' added' : '') + '">' + esc(m.label) + (m.group === 'added' ? ' ' + badge('added') : '') + '</th>' + D.MODES.map(function (mode) {
        var cell = p.scores[m.id][mode], v = D.probeOverall(cell);
        var arcCls = mode === 'arc' ? ' arccol' : '';
        if (v == null) return '<td class="na' + arcCls + '">—</td>';
        var h = Math.max(0, Math.min(1, (v - 20) / 80));
        var parts = 'per-phase avg ' + fmt(D.probePhaseAvg(cell)) + ' · PTF ' + fmt(D.probePTF(cell));
        var isA = sa.model === m.id && sa.mode === mode, isB = sb.model === m.id && sb.mode === mode;
        return '<td class="heat num' + arcCls + '" style="--h:' + h.toFixed(2) + '"><button type="button" class="cellbtn' + (isA ? ' selA' : '') + (isB ? ' selB' : '') + '" data-cell="' + esc(m.id + ':' + mode) + '" title="' + esc(parts) + '" aria-label="' + esc(m.label + ' under ' + modeLabel(mode) + ': ' + fmt(v) + ' (' + parts + ')') + '">' +
          (isA ? '<span class="tag a">A</span>' : '') + (isB ? '<span class="tag b">B</span>' : '') + fmt(v) + '</button></td>';
      }).join('') + '</tr>';
    }).join('');
    return '<div class="tablewrap"><table class="data matrix"><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div>' +
      (missing.length ? '<p class="matrix-help">Not scored on this probe: ' + missing.map(function (m) { return esc(m.label); }).join(', ') + '.</p>' : '');
  }
  function fillViewer(n, cm, p, sa, sb) {
    var box = $('#viewer'); if (!box) return;
    var id = renderId;
    function paint(resp, err) {
      if (id !== renderId) return;
      var el = $('#viewer'); if (!el) return;
      el.innerHTML = viewerHtml(p, sa, sb, resp, err);
    }
    var needs = [sa, sb].some(function (s) { var m = modelOf(s.model); return m && m.responses; });
    if (!p.has_responses) { paint(null, { message: 'No responses were archived for this probe.', soft: true }); return; }
    if (!needs) { paint(null, null); return; }
    var url = respUrl(n.id, cm.id, p.probe_id);
    loadResponses(n.id, cm.id, p.probe_id).then(function (r) { paint(r.responses || {}, null); }, function (e) { paint(null, { message: e && e.message ? e.message : String(e), url: url }); });
  }
  function viewerHtml(p, sa, sb, resp, err) {
    var html = '';
    if (err && !err.soft) html += errorBox(err.url || '', err);
    else if (err) html += '<div class="empty inline">' + esc(err.message) + '</div>';
    html += '<div class="ptfrow">' + ptfBox('A', sa, p) + ptfBox('B', sb, p) + '</div>';
    p.refs.forEach(function (r) {
      var i = r.phase_idx;
      html += '<div class="phaseblock"><div class="refsum p' + (i % 6) + (r.unavailable ? ' unavailable' : '') + '"><span>' + pill(i, r.phase_label) + '<br><span class="ch">query chapter ' + esc(r.query_chapter) + (r.unavailable ? ' · unavailable' : '') + '</span></span><span class="txt">' + esc(r.action) + (r.thought ? ' <i>' + esc(r.thought) + '</i>' : '') + '</span></div>';
      if (r.unavailable) html += '<p class="tnote">This phase is unavailable, so no response was generated and no score exists.</p></div>';
      else html += '<div class="compare">' + respBox('A', sa, p, resp, i, err && err.soft) + respBox('B', sb, p, resp, i, err && err.soft) + '</div></div>';
    });
    return html;
  }
  function ptfBox(who, sel, p) {
    var cell = p.scores && p.scores[sel.model] && p.scores[sel.model][sel.mode];
    var ptf = cell && cell.ptf, m = D.probePTF(cell), avg = D.probePhaseAvg(cell);
    var cls = sel.mode === 'arc' ? 'arc' : 'base';
    return '<div class="ptfbox"><span class="who ' + cls + '">' + who + ' · ' + esc(modelLabel(sel.model)) + ' · ' + esc(modeLabel(sel.mode)) + '</span>' +
      (cell ? scoreChip('PTF', m) + (ptf ? '<span class="muted">alignment ' + D.fmt(ptf[0], 0) + ' · direction ' + D.fmt(ptf[1], 0) + ' · shape ' + D.fmt(ptf[2], 0) + '</span>' : '<span class="muted">no trajectory score</span>') + scoreChip('per-phase', avg) + scoreChip('Overall', D.probeOverall(cell)) : '<span class="muted">not scored on this probe</span>') + '</div>';
  }
  function respBox(who, sel, p, resp, phaseIdx, softMissing) {
    var m = sel.model, mode = sel.mode;
    var sc = p.scores && p.scores[m] && p.scores[m][mode] && p.scores[m][mode].phases && p.scores[m][mode].phases[String(phaseIdx)];
    var chipsHtml = sc ? '<span class="scores">' + scoreChip('APF', sc[0]) + scoreChip('RPF', sc[1]) + scoreChip('RAE', sc[2]) + '</span>' : '<span class="muted small">no score</span>';
    var head = '<div class="rhead"><span class="who ' + (mode === 'arc' ? 'arc' : 'base') + '">' + who + ' · ' + esc(modelLabel(m)) + ' <small>· ' + esc(modeLabel(mode)) + '</small></span>' + chipsHtml + '</div>';
    var mm = modelOf(m);
    if (mm && !mm.responses) return '<div class="resp missing">' + head + '<div class="rbody">Responses for this model are not archived (scores only).</div></div>';
    if (!resp) return '<div class="resp missing">' + head + '<div class="rbody">' + (softMissing ? 'No responses archived for this probe.' : 'Response not archived.') + '</div></div>';
    var r = resp[m] && resp[m][mode] && resp[m][mode][String(phaseIdx)];
    if (!r || !r.text) return '<div class="resp missing">' + head + '<div class="rbody">Response not archived.</div></div>';
    var long = (r.chars || r.text.length) > CLAMP_CHARS;
    return '<div class="resp">' + head + '<div class="rbody' + (long ? ' clamp' : '') + '">' + esc(r.text) + (r.truncated ? '\n<span class="muted">[archived text truncated at ' + fmtInt(cat.response_cap_chars) + ' characters]</span>' : '') + '</div>' +
      (long ? '<button type="button" class="more" aria-expanded="false">Show more (' + fmtInt(r.chars || r.text.length) + ' characters)</button>' : '') +
      (r.think ? '<details class="think"><summary>Reasoning trace' + (r.think_truncated ? ' (truncated)' : '') + '</summary><pre>' + esc(r.think) + '</pre></details>' : '') + '</div>';
  }

  /* ── router ────────────────────────────────────────────────── */
  function focusKeyOf(el) {
    if (!el || el === document.body) return null;
    if (el.hasAttribute('data-set')) return '[data-set="' + el.getAttribute('data-set') + '"][data-val="' + el.getAttribute('data-val') + '"]';
    if (el.hasAttribute('data-cell')) return '[data-cell="' + el.getAttribute('data-cell') + '"]';
    if (el.hasAttribute('data-key')) return el.tagName.toLowerCase() + '[data-key="' + el.getAttribute('data-key') + '"]' + (el.hasAttribute('data-part') ? '[data-part="' + el.getAttribute('data-part') + '"]' : '');
    return null;
  }
  function show(out, opts) {
    opts = opts || {};
    app.innerHTML = out.html;
    document.title = out.title || SITE;
    if (out.after) out.after();
    applySorts(app);
    renderSubnav();
    if (opts.focusKey) { var el = $(opts.focusKey, app); if (el) { try { el.focus({ preventScroll: true }); } catch (e) { el.focus(); } } }
    if (opts.scroll) {
      var target = opts.scroll === true ? null : $(opts.scroll);
      if (target) target.scrollIntoView();
      else window.scrollTo(0, 0);
    }
    markScrollAll();
  }
  function loadingHtml(crumbHtml, what) { return crumbHtml + '<p class="loading">Loading ' + esc(what) + '…</p>'; }

  function render(pathChanged) {
    var id = ++renderId;
    partial = {};
    var focusKey = pathChanged ? null : focusKeyOf(document.activeElement);
    var segs = state.segs;
    var scroll = pathChanged;

    /* overview and its anchors */
    if (!segs.length || (segs.length === 1 && (segs[0] === 'novels' || segs[0] === 'models'))) {
      show(viewOverview(null), { scroll: pathChanged ? (segs.length ? '#' + segs[0] : true) : false, focusKey: focusKey });
      return;
    }
    if (segs[0] !== 'n' || !segs[1]) { show(viewOverview('Unknown route "#' + state.path + '". Showing the overview instead.'), { scroll: true }); return; }
    var n = novelOf(segs[1]);
    if (!n) { show(viewOverview('Unknown novel "' + segs[1] + '". Showing the overview instead.'), { scroll: true }); return; }
    if (segs.length === 2) { show(viewNovel(n), { scroll: scroll, focusKey: focusKey }); return; }
    if (segs[2] !== 'c' || !segs[3]) { show(viewOverview('Unknown route "#' + state.path + '". Showing the overview instead.'), { scroll: true }); return; }
    var cm = D.characterMeta(cat, n.id, segs[3]);
    if (!cm) { show(viewOverview('Unknown character "' + segs[3] + '" in ' + n.title + '. Showing the overview instead.'), { scroll: true }); return; }

    var crumbHtml = crumbs([{ href: href('/'), text: 'Explorer' }, { href: href('/n/' + n.id), text: novelTitle(n) }, { text: cm.name }]);
    function notFound(what, id) {
      return { title: 'Not found · ' + SITE, html: crumbHtml + pagehead('Not found') + '<div class="error notice" role="alert">Unknown ' + what + ' “' + esc(id) + '” for ' + esc(cm.name) + '.</div>' +
        '<p><a href="' + href('/n/' + n.id + '/c/' + cm.id) + '">Back to ' + esc(cm.name) + '</a></p>' };
    }
    function withBundle(fn) {
      var key = n.id + '/' + cm.id;
      if (bundles[key]) { show(fn(bundles[key]), { scroll: scroll, focusKey: focusKey }); return; }
      show({ title: cm.name + ' · ' + SITE, html: loadingHtml(crumbHtml, cm.name) }, { scroll: scroll });
      loadBundle(n.id, cm.id).then(function (b) {
        if (id !== renderId) return;
        show(fn(b), { scroll: false });
      }, function (e) {
        if (id !== renderId) return;
        show({ title: 'Error · ' + SITE, html: crumbHtml + errorBox(bundleUrl(n.id, cm.id), e) }, { scroll: false });
      });
    }
    if (segs.length === 4) { withBundle(function (b) { return viewCharacter(n, cm, b); }); return; }
    if (segs[4] === 'a' && segs[5] && segs.length === 6) {
      withBundle(function (b) {
        var a = b.arcs.filter(function (x) { return x.axis_id === segs[5]; })[0];
        if (!a) return notFound('arc', segs[5]);
        return viewArc(n, cm, b, a);
      });
      return;
    }
    if (segs[4] === 'p' && segs[5] && segs.length === 6) {
      withBundle(function (b) {
        var idx = -1;
        b.probes.forEach(function (x, i) { if (x.probe_id === segs[5]) idx = i; });
        if (idx < 0) return notFound('probe', segs[5]);
        return viewProbe(n, cm, b, b.probes[idx], idx);
      });
      return;
    }
    show(viewOverview('Unknown route "#' + state.path + '". Showing the overview instead.'), { scroll: true });
  }

  function renderSubnav() {
    var box = $('#subnav'); if (!box || !cat) return;
    var cur = state.segs[0] === 'n' ? state.segs[1] : null;
    var html = '<a href="#/"' + (!state.segs.length || state.segs[0] === 'novels' || state.segs[0] === 'models' ? ' class="on" aria-current="page"' : '') + '>Overview</a>';
    ['validated', 'lowpop', 'training'].forEach(function (s) {
      html += '<span class="grp"><span class="fam">' + esc(SLICE_SHORT[s]) + '</span>';   // a slice and its novels wrap as one unit
      cat.slices[s].novels.forEach(function (nid) {
        var n = novelOf(nid); if (!n) return;
        html += '<a href="' + href('/n/' + nid) + '"' + (cur === nid ? ' class="on" aria-current="page"' : '') + ' title="' + esc(n.title) + '">' + esc(novelTitle(n)) + '</a>';
      });
      html += '</span>';
    });
    box.innerHTML = html;
    setChrome();
  }

  /* ── chrome: sticky-bar height and scroll cues ─────────────── */
  /* --chrome = height of the sticky top bar + sub-nav, so anchor scrolling (scroll-margin-top) clears them at any width */
  function setChrome() {
    var h = 0; $$('.topbar, .subnav').forEach(function (el) { h += el.offsetHeight; });
    document.documentElement.style.setProperty('--chrome', h + 'px');
  }
  /* while more content lies beyond the right or bottom edge of a scroll container, fade that edge (.fade-r / .fade-b) */
  var SCROLLERS = '.tablewrap, .subnav .wrap, .seg';
  function markScroll(el) {
    el.classList.toggle('fade-r', el.scrollWidth > el.clientWidth + 1 && el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
    el.classList.toggle('fade-b', el.scrollHeight > el.clientHeight + 1 && el.scrollTop + el.clientHeight < el.scrollHeight - 1);
  }
  function markScrollAll() { $$(SCROLLERS).forEach(markScroll); }

  function onHashChange() {
    if (!cat) { boot(); return; }   // catalog still loading (the current hash is rendered when it lands) or failed (retry the download)
    var r = parseHash();
    var pathChanged = r.path !== state.path;
    state = r;
    canonicalizeHash();
    render(pathChanged);
  }
  function canonicalizeHash() {
    var canon = buildHash(state.path, state.q);
    if (canon !== location.hash) history.replaceState(null, '', canon);
  }

  /* ── events (delegated) ────────────────────────────────────── */
  app.addEventListener('click', function (e) {
    var t = e.target.closest('button[data-set], th.sort, button.more, button[data-cell], tr.link');
    if (!t) return;
    if (t.matches('button[data-set]')) {
      var patch = {}; patch[t.getAttribute('data-set')] = t.getAttribute('data-val');
      setQuery(patch, t.getAttribute('data-partial'));
      return;
    }
    if (t.matches('th.sort')) { onSortClick(t); return; }
    if (t.matches('button.more')) {
      var body = t.previousElementSibling;
      var open = !!body && body.classList.toggle('clamp') === false;
      t.setAttribute('aria-expanded', open ? 'true' : 'false');
      t.textContent = open ? 'Show less' : 'Show more';
      return;
    }
    if (t.matches('button[data-cell]')) {
      var pk = {}; pk[get('pick')] = t.getAttribute('data-cell');
      setQuery(pk);
      return;
    }
    if (t.matches('tr.link')) {
      if (e.target.closest('a, button, select, input')) return;
      var h = t.getAttribute('data-href');
      if (h) location.hash = h;
    }
  });
  app.addEventListener('change', function (e) {
    var s = e.target.closest('select[data-key]'); if (!s) return;
    var key = s.getAttribute('data-key'), patch = {};
    if (s.hasAttribute('data-part')) {           // response-viewer selectors: a / b = model:mode
      var cur = parseSel(get(key));
      if (s.getAttribute('data-part') === 'model') cur.model = s.value; else cur.mode = s.value;
      patch[key] = cur.model + ':' + cur.mode;
    } else patch[key] = s.value;
    setQuery(patch, s.getAttribute('data-partial'));
  });
  app.addEventListener('input', function (e) {
    var i = e.target.closest('input[data-key]'); if (!i) return;
    var patch = {}; patch[i.getAttribute('data-key')] = i.value;
    setQuery(patch, i.getAttribute('data-partial'));
  });
  window.addEventListener('hashchange', onHashChange);
  document.addEventListener('click', function (e) {
    var a = e.target.closest('a[href^="#"]'); if (!a || !cat) return;
    if (a.getAttribute('href') === location.hash) { e.preventDefault(); render(true); }
  });
  document.addEventListener('scroll', function (e) { var el = e.target; if (el && el.matches && el.matches(SCROLLERS)) markScroll(el); }, true);
  window.addEventListener('resize', function () { setChrome(); markScrollAll(); });
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(function () { setChrome(); markScrollAll(); });

  /* ── boot ──────────────────────────────────────────────────── */
  var booting = false;
  function boot() {
    if (cat || booting) return;
    booting = true;
    D.catalog().then(function (c) {
      booting = false;
      cat = c;
      state = parseHash();
      canonicalizeHash();
      render(true);
    }, function (e) {
      booting = false;
      app.innerHTML = errorBox(D.base + 'catalog.json', e) + '<p class="tnote">The explorer needs the released data bundle under <code>data/</code>. If you are viewing a local copy, serve the folder over HTTP rather than opening the file directly. Following any explorer link retries the download.</p>';
      var sb = $('#subnav'); if (sb) sb.innerHTML = '<span class="fam">Catalog unavailable</span>';
    });
  }
  boot();
})();
