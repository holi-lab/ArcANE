/* ArcANE project page — landing page behaviour.
   1. Hero probe panel: one probe from Anna Karenina loaded live from the
      per-character bundle (arc, probe, references, judge scores) and the
      per-probe response bundle (archived model responses).
   2. Interactive main-results table driven by data/paper.json.
   3. Explorer teaser cards from data/catalog.json.
   4. BibTeX copy button.
   Plain ES2020, no libraries. Depends on window.ArcData (static/js/data.js). */
(function () {
  'use strict';

  var D = window.ArcData;
  var esc = D ? D.esc : function (s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); };

  /* ?theme=dark|light forces the colour scheme (used for testing; the site
     otherwise follows prefers-color-scheme). */
  try {
    var th = new URLSearchParams(window.location.search).get('theme');
    if (th === 'dark' || th === 'light') document.documentElement.setAttribute('data-theme', th);
  } catch (e) { /* ignore */ }

  /* Initial state can be given in the query string so that a particular view
     is linkable: ?probe_model=ds-v4-pro&probe_mode=vanilla&probe_phase=3&view=in_world&metric=ptf */
  var QUERY = {};
  try { new URLSearchParams(window.location.search).forEach(function (v, k) { QUERY[k] = v; }); } catch (e) { /* ignore */ }
  function pressButton(group, attr, value) {
    var btn = group && group.querySelector('button[' + attr + '="' + value + '"]');
    if (btn) setPressed(group, btn);
    return !!btn;
  }

  function mean(arr) {
    var v = (arr || []).filter(function (x) { return x != null && !isNaN(x); });
    if (!v.length) return null;
    return v.reduce(function (a, b) { return a + b; }, 0) / v.length;
  }
  function fmt(v, d) { if (v == null || isNaN(v)) return '—'; return v.toFixed(d == null ? 1 : d); }
  function fmtSigned(v, d) {
    if (v == null || isNaN(v)) return '—';
    var s = Math.abs(v).toFixed(d == null ? 1 : d);
    return (v > 0 ? '+' : v < 0 ? '−' : '') + s;
  }
  function setPressed(group, active) {
    Array.prototype.forEach.call(group.querySelectorAll('button'), function (b) {
      b.setAttribute('aria-pressed', b === active ? 'true' : 'false');
    });
  }
  function scoreChip(label, v) {
    var cls = v == null ? '' : v >= 70 ? ' hi' : v <= 35 ? ' lo' : '';
    return '<span class="score' + cls + '" title="' + esc(label) + '">' + esc(label) + ' <b>' + (v == null ? '—' : Math.round(v)) + '</b></span>';
  }
  function errorBox(msg) { return '<div class="error">' + esc(msg) + '</div>'; }

  /* ════════════════════════════════════════════════════════════════
     1. Hero probe panel
     ════════════════════════════════════════════════════════════════ */
  var HERO = { novel: 'anna-karenina', character: 'anna_karenina', probe: 'anna_karenina_final_rel_02_outworld_a0' };
  /* grouped by family so each family label is printed once in the chips */
  var MODEL_ORDER = ['arcane-32b-dpo', 'arcane-8b-dpo', 'ds-v4-pro', 'ds-v4-flash', 'qwen3-32b', 'qwen3-8b'];
  /* era ids -> the paper's spelling of the seven-slot era menu (Appendix, probe generation) */
  var ERA_LABEL = { modern_urban: 'modern urban', industrial_early_20th: 'industrial early 20th century', mid_century: 'mid-century', pre_industrial_agrarian: 'pre-industrial agrarian', speculative_near_future: 'speculative near-future', pre_modern_imperial: 'pre-modern imperial', diasporic_contemporary: 'diasporic contemporary' };
  var CLAMP_CHARS = 1400;

  function heroPanel() {
    var panel = document.getElementById('heroPanel');
    if (!panel) return;
    var body = document.getElementById('heroBody');
    var foot = document.getElementById('heroFoot');
    var chips = document.getElementById('heroModelChips');
    var seg = document.getElementById('heroModeSeg');
    if (!D) { body.innerHTML = errorBox('The data loader (static/js/data.js) did not load, so the live probe cannot be shown.'); foot.hidden = true; return; }

    var state = { mode: 'both', model: null, phase: 0 };
    var cat, probe, arc, resp, models;

    Promise.all([D.catalog(), D.character(HERO.novel, HERO.character), D.responses(HERO.novel, HERO.character, HERO.probe)])
      .then(function (r) {
        cat = r[0];
        var ch = r[1];
        resp = r[2];
        probe = (ch.probes || []).find(function (p) { return p.probe_id === HERO.probe; });
        arc = probe && (ch.arcs || []).find(function (a) { return a.axis_id === probe.axis_id; });
        if (!probe || !arc) throw new Error('probe ' + HERO.probe + ' is not in the character bundle');
        models = MODEL_ORDER.filter(function (m) {
          var rm = resp.responses && resp.responses[m];
          return rm && (rm.arc || rm.vanilla);
        });
        if (!models.length) throw new Error('no archived responses for this probe');
        state.model = models.indexOf(QUERY.probe_model) >= 0 ? QUERY.probe_model : models[0];
        state.phase = probe.refs && probe.refs.length ? probe.refs[0].phase_idx : 0;
        var qp = parseInt(QUERY.probe_phase, 10);
        if (!isNaN(qp) && probe.refs.some(function (r) { return r.phase_idx === qp; })) state.phase = qp;
        if (['both', 'arc', 'vanilla'].indexOf(QUERY.probe_mode) >= 0 && pressButton(seg, 'data-mode', QUERY.probe_mode)) state.mode = QUERY.probe_mode;
        buildChips();
        buildStatic();
        render();
      })
      .catch(function (e) {
        body.innerHTML = errorBox('Could not load the probe data (' + (e && e.message ? e.message : e) + '). The caption below describes what the panel shows, and the same probe is available in the Explorer.');
        foot.hidden = true;
        chips.hidden = true;
        seg.hidden = true;
      });

    function buildChips() {
      chips.innerHTML = '';
      var lastFam = null;
      models.forEach(function (id) {
        var m = D.model(cat, id) || { label: id, family: '', size: id };
        if (m.family !== lastFam) {
          var f = document.createElement('span'); f.className = 'fam'; f.textContent = m.family; chips.appendChild(f); lastFam = m.family;
        }
        var b = document.createElement('button');
        b.type = 'button';
        b.textContent = m.size;
        b.title = m.label;
        b.setAttribute('aria-label', m.label);
        b.setAttribute('aria-pressed', id === state.model ? 'true' : 'false');
        b.addEventListener('click', function () {
          if (state.model === id) return;
          state.model = id;
          setPressed(chips, b);
          render();
        });
        chips.appendChild(b);
      });
    }

    seg.addEventListener('click', function (ev) {
      var b = ev.target.closest('button');
      if (!b || !seg.contains(b)) return;
      var mode = b.getAttribute('data-mode');
      if (mode === state.mode) return;
      state.mode = mode;
      setPressed(seg, b);
      render();
    });

    var grid, nav;
    function buildStatic() {
      var ptype = D.ptypeLabel(cat, probe.probe_type);
      var era = probe.era_label ? (ERA_LABEL[probe.era_label] || probe.era_label.replace(/_/g, ' ')) : null;
      var target = arc.target_character ? ' toward <b>' + esc(arc.target_character) + '</b>' : '';
      var votes = arc.validation && arc.validation.n_annotators ? ' · validated by ' + arc.validation.valid_votes + ' of ' + arc.validation.n_annotators + ' annotators' : '';
      var html = '';
      html += '<div class="hp-meta">';
      html += '<span class="badge type-' + esc(probe.probe_type) + '">' + esc(ptype) + '</span>';
      if (era) html += '<span>era: ' + esc(era) + '</span>';
      /* separators are drawn by .dot::before so they wrap together with their item */
      html += '<span class="dot"><span class="badge">' + esc(arc.axis_type) + ' axis</span>' + target + '</span>';
      html += '<span class="dot axis">' + esc(arc.axis_name) + '</span>';
      html += '<span class="muted">' + arc.n_phases + ' phases' + votes + '</span>';
      html += '</div>';
      html += '<div class="probe-box"><p class="lab">Scenario</p><p class="scenario">' + esc(probe.scenario) + '</p><p class="lab">Question</p><p class="question">' + esc(probe.question) + '</p></div>';
      html += '<div class="phase-nav" role="group" aria-label="Phase" id="heroPhaseNav"></div>';
      html += '<div class="hp-grid" id="heroGrid"></div>';
      body.innerHTML = html;
      nav = document.getElementById('heroPhaseNav');
      grid = document.getElementById('heroGrid');
      probe.refs.forEach(function (ref) {
        var k = ref.phase_idx;
        var b = document.createElement('button');
        b.type = 'button';
        b.className = 'p' + Math.min(k, 5);
        b.setAttribute('data-phase', String(k));
        b.setAttribute('aria-pressed', k === state.phase ? 'true' : 'false');
        b.innerHTML = '<span class="n">' + (k + 1) + '</span><span>' + esc(ref.phase_label) + '</span><span class="ch">ch. ' + esc(ref.query_chapter) + '</span>';
        b.addEventListener('click', function () {
          if (state.phase === k) return;
          state.phase = k;
          setPressed(nav, b);
          render();
        });
        nav.appendChild(b);
      });
    }

    function phaseMeta(k) { return (arc.phases || []).find(function (p) { return p.idx === k; }); }

    function refBlock(ref) {
      var k = ref.phase_idx, pm = phaseMeta(k);
      var span = pm && pm.chapters ? 'phase spans ch. ' + pm.chapters[0] + '–' + pm.chapters[1] : '';
      var html = '<div class="ref p' + Math.min(k, 5) + (ref.unavailable ? ' unavailable' : '') + '">';
      html += '<div class="head"><span class="phase-pill p' + Math.min(k, 5) + '"><span class="n">' + (k + 1) + '</span>' + esc(ref.phase_label) + '</span><span class="small muted">query ch. ' + esc(ref.query_chapter) + (span ? ' · ' + span : '') + '</span></div>';
      if (ref.unavailable) {
        html += '<p class="muted small" style="margin:0">Reference marked unavailable for this phase (dropped from scoring).</p>';
      } else {
        html += '<div class="field"><span class="k">Action</span><span class="v">' + esc(ref.action) + '</span></div>';
        if (ref.speech) html += '<div class="field"><span class="k">Speech</span><span class="v">“' + esc(ref.speech) + '”</span></div>';
        html += '<div class="field thought"><span class="k">Thought</span><span class="v">' + esc(ref.thought) + '</span></div>';
      }
      html += '</div>';
      if (pm && pm.description) html += '<p class="tnote"><b>Phase ' + (k + 1) + ' in the arc:</b> ' + esc(pm.description) + '</p>';
      return html;
    }

    function respBlock(mode) {
      var modeLabel = D.modeLabel(cat, mode);
      var who = '<span class="who ' + (mode === 'arc' ? 'arc' : 'base') + '">' + esc(modeLabel) + '</span>';
      var modelLabel = D.modelLabel(cat, state.model);
      var cell = resp.responses[state.model] && resp.responses[state.model][mode] && resp.responses[state.model][mode][String(state.phase)];
      var sc = probe.scores && probe.scores[state.model] && probe.scores[state.model][mode];
      var ph = sc && sc.phases && sc.phases[String(state.phase)];
      var chipsHtml = ph ? '<span class="scores">' + scoreChip('APF', ph[0]) + scoreChip('RPF', ph[1]) + scoreChip('RAE', ph[2]) + '</span>' : '<span class="scores"><span class="score">no judge record</span></span>';
      if (!cell) {
        return '<div class="resp missing"><div class="rhead">' + who + '<span class="model">' + esc(modelLabel) + '</span>' + chipsHtml + '</div><div class="rbody">Response not archived.</div></div>';
      }
      var text = cell.text || '';
      var clamp = text.length > CLAMP_CHARS;
      var html = '<div class="resp"><div class="rhead">' + who + '<span class="model">' + esc(modelLabel) + '</span>' + chipsHtml + '</div>';
      html += '<div class="rbody' + (clamp ? ' clamp' : '') + '">' + esc(text) + '</div>';
      if (clamp) html += '<button type="button" class="more">Show full response</button>';
      if (cell.truncated) html += '<div class="think">Archived response truncated at ' + esc(cell.chars) + ' characters.</div>';
      if (cell.think) html += '<details class="think"><summary>Model reasoning' + (cell.think_truncated ? ' (truncated)' : '') + '</summary><pre>' + esc(cell.think) + '</pre></details>';
      html += '</div>';
      return html;
    }

    function ptfStat(mode) {
      var sc = probe.scores && probe.scores[state.model] && probe.scores[state.model][mode];
      var label = D.modeLabel(cat, mode);
      if (!sc || !sc.ptf) return '<span class="stat"><span class="k">PTF under ' + esc(label) + '&nbsp;</span>—</span>';
      var p = sc.ptf;
      return '<span class="stat"><span class="k">PTF under ' + esc(label) + '&nbsp;</span><b>' + fmt(mean(p), 1) + '</b> <span class="sub">(alignment ' + fmt(p[0], 0) + ' · direction ' + fmt(p[1], 0) + ' · shape ' + fmt(p[2], 0) + ')</span></span>';
    }
    function phaseAvgStat(modes) {
      var parts = modes.map(function (mode) {
        var sc = probe.scores && probe.scores[state.model] && probe.scores[state.model][mode];
        return esc(D.modeLabel(cat, mode)) + ' ' + fmt(D.probePhaseAvg(sc), 1);
      });
      return '<span class="stat"><span class="k">Per-phase mean of APF/RPF/RAE over all phases&nbsp;</span>' + parts.join(' <span class="k">·</span> ') + '</span>';
    }

    function render() {
      var ref = probe.refs.find(function (r) { return r.phase_idx === state.phase; }) || probe.refs[0];
      var modes = state.mode === 'both' ? ['arc', 'vanilla'] : [state.mode];
      var html = '<div class="hp-ref"><p class="aside-lab">Reference at phase ' + (state.phase + 1) + '</p>' + refBlock(ref) + '</div>';
      html += '<div class="hp-resps"><p class="aside-lab">' + esc(D.modelLabel(cat, state.model)) + ' at phase ' + (state.phase + 1) + ', with judge scores</p>';
      html += '<div class="compare' + (modes.length === 1 ? ' single' : '') + '">' + modes.map(respBlock).join('') + '</div></div>';
      grid.innerHTML = html;
      Array.prototype.forEach.call(grid.querySelectorAll('.resp .more'), function (btn) {
        btn.addEventListener('click', function () {
          var bodyEl = btn.previousElementSibling;
          bodyEl.classList.remove('clamp');
          btn.remove();
        });
      });
      foot.hidden = false;
      foot.innerHTML = modes.map(ptfStat).join('') + phaseAvgStat(modes);
    }
  }

  /* ════════════════════════════════════════════════════════════════
     2. Main results table (data/paper.json → "main")
     ════════════════════════════════════════════════════════════════ */
  var PTYPE_LABEL = { in_text: 'In-Scenario', in_world: 'In-World', out_of_world: 'Out-of-World' };
  var METRIC_LABEL = { apf: 'APF', rpf: 'RPF', rae: 'RAE', ptf: 'PTF', all: 'Mean' };
  var METRIC_INDEX = { apf: 0, rpf: 1, rae: 2, ptf: 3 };

  function mainTable() {
    var panel = document.getElementById('mainPanel');
    if (!panel) return;
    var body = document.getElementById('mainBody');
    var foot = document.getElementById('mainFoot');
    var viewSeg = document.getElementById('mainViewSeg');
    var metricSeg = document.getElementById('mainMetricSeg');
    if (!D) { body.innerHTML = errorBox('The data loader did not load, so the results table cannot be rendered.'); return; }

    var state = { view: 'overall', metric: 'all' };
    var blocks = [];
    if (['overall', 'in_text', 'in_world', 'out_of_world'].indexOf(QUERY.view) >= 0 && pressButton(viewSeg, 'data-view', QUERY.view)) state.view = QUERY.view;
    if (['apf', 'rpf', 'rae', 'ptf', 'all'].indexOf(QUERY.metric) >= 0 && pressButton(metricSeg, 'data-metric', QUERY.metric)) state.metric = QUERY.metric;
    metricSeg.hidden = state.view === 'overall';

    D.paper().then(function (paper) {
      var map = {};
      (paper.main || []).forEach(function (r) {
        var key = r.family + '|' + r.size;
        if (!map[key]) { map[key] = { family: r.family, size: r.size, rows: [] }; blocks.push(map[key]); }
        map[key].rows.push(r);
      });
      if (!blocks.length) throw new Error('paper.json has no main table');
      render();
    }).catch(function (e) {
      body.innerHTML = errorBox('Could not load data/paper.json (' + (e && e.message ? e.message : e) + ').');
    });

    viewSeg.addEventListener('click', function (ev) {
      var b = ev.target.closest('button');
      if (!b || !viewSeg.contains(b)) return;
      var v = b.getAttribute('data-view');
      if (v === state.view) return;
      state.view = v;
      setPressed(viewSeg, b);
      metricSeg.hidden = v === 'overall';
      if (blocks.length) render();
    });
    metricSeg.addEventListener('click', function (ev) {
      var b = ev.target.closest('button');
      if (!b || !metricSeg.contains(b)) return;
      var m = b.getAttribute('data-metric');
      if (m === state.metric) return;
      state.metric = m;
      setPressed(metricSeg, b);
      if (blocks.length) render();
    });

    function columns() {
      if (state.view === 'overall') {
        return [
          { key: 'in_text', label: 'In-Scenario', value: function (r) { return mean(r.in_text); } },
          { key: 'in_world', label: 'In-World', value: function (r) { return mean(r.in_world); } },
          { key: 'out_of_world', label: 'Out-of-World', value: function (r) { return mean(r.out_of_world); } },
          { key: 'overall', label: 'Overall', value: function (r) { return r.overall; }, basis: true }
        ];
      }
      var pt = state.view;
      var cols = ['apf', 'rpf', 'rae', 'ptf'].map(function (m) {
        return { key: m, label: METRIC_LABEL[m], value: function (r) { return r[pt] ? r[pt][METRIC_INDEX[m]] : null; }, basis: state.metric === m };
      });
      cols.push({ key: 'all', label: 'Mean', value: function (r) { return mean(r[pt]); }, basis: state.metric === 'all' });
      return cols;
    }

    function render() {
      var cols = columns();
      var basis = cols.find(function (c) { return c.basis; }) || cols[cols.length - 1];
      var html = '<div class="tablewrap"><table class="data"><thead><tr><th scope="col" class="l">Family</th><th scope="col" class="l">Size</th><th scope="col" class="l">Mode</th>';
      cols.forEach(function (c) { html += '<th scope="col"' + (c.basis ? ' class="on"' : '') + '>' + esc(c.label) + '</th>'; });
      html += '<th scope="col" class="on" title="Arc minus the strongest non-Arc mode on the ' + esc(basis.label) + ' column">Arc lift</th></tr></thead><tbody>';
      var wins = 0, lifts = [];
      blocks.forEach(function (blk, bi) {
        var vals = blk.rows.map(function (r) { return cols.map(function (c) { return c.value(r); }); });
        var best = cols.map(function (c, ci) { return Math.max.apply(null, vals.map(function (v) { return v[ci] == null ? -Infinity : v[ci]; })); });
        var bidx = cols.indexOf(basis);
        var arcRow = blk.rows.findIndex(function (r) { return r.mode === 'Arc'; });
        var arcVal = arcRow >= 0 ? vals[arcRow][bidx] : null;
        var bestNon = -Infinity, bestNonMode = null;
        blk.rows.forEach(function (r, ri) {
          if (r.mode === 'Arc' || vals[ri][bidx] == null) return;
          if (vals[ri][bidx] > bestNon) { bestNon = vals[ri][bidx]; bestNonMode = r.mode; }
        });
        var lift = arcVal != null && isFinite(bestNon) ? arcVal - bestNon : null;
        if (lift != null) { lifts.push(lift); if (lift > 0) wins += 1; }
        blk.rows.forEach(function (r, ri) {
          var isArc = r.mode === 'Arc';
          html += '<tr class="' + (isArc ? 'arc' : '') + (ri === 0 && bi > 0 ? ' sep' : '') + '">';
          if (ri === 0) html += '<th scope="row" class="l fam" rowspan="' + blk.rows.length + '">' + esc(blk.family) + '</th><th scope="row" class="l fam" rowspan="' + blk.rows.length + '">' + esc(blk.size) + '</th>';
          html += '<th scope="row" class="l">' + esc(r.mode) + '</th>';
          cols.forEach(function (c, ci) {
            var v = vals[ri][ci];
            var s = fmt(v, 1);
            var isBest = v != null && Math.abs(v - best[ci]) < 1e-9;
            html += '<td class="num">' + (isBest ? '<b>' + s + '</b>' : s) + '</td>';
          });
          if (isArc) html += '<td class="num lift' + (lift == null ? '' : lift > 0 ? ' pos' : lift < 0 ? ' neg' : '') + '" title="Arc ' + fmt(arcVal, 1) + ' minus best non-Arc (' + esc(bestNonMode || '—') + ') ' + fmt(isFinite(bestNon) ? bestNon : null, 1) + '">' + fmtSigned(lift, 1) + '</td>';
          else html += '<td class="num"></td>';
          html += '</tr>';
        });
      });
      html += '</tbody></table></div>';
      body.innerHTML = html;
      var viewLabel = state.view === 'overall' ? 'Overall' : PTYPE_LABEL[state.view] + ' · ' + basis.label;
      var lo = Math.min.apply(null, lifts), hi = Math.max.apply(null, lifts);
      foot.innerHTML = '<span class="stat"><span class="k">Column&nbsp;</span>' + esc(viewLabel) + '</span>' +
        '<span class="stat"><span class="k">Arc is the best mode for&nbsp;</span>' + wins + ' of ' + blocks.length + ' models</span>' +
        '<span class="stat"><span class="k">Arc lift range&nbsp;</span>' + fmtSigned(lo, 1) + ' to ' + fmtSigned(hi, 1) + '</span>';
    }
  }

  /* ════════════════════════════════════════════════════════════════
     3. Explorer teaser cards (data/catalog.json)
     ════════════════════════════════════════════════════════════════ */
  function novelCards() {
    var box = document.getElementById('novelCards');
    if (!box || !D) return;
    D.catalog().then(function (cat) {
      var ids = [].concat(cat.slices.validated.novels || [], cat.slices.lowpop.novels || []);
      var html = '';
      ids.forEach(function (id) {
        var n = D.novel(cat, id);
        if (!n) return;
        var slice = cat.slices[n.slice] ? cat.slices[n.slice].label : n.slice;
        var badgeCls = n.slice === 'validated' ? 'good' : '';
        html += '<a class="card" href="explore.html#/n/' + encodeURIComponent(id) + '">' +
          '<p class="title">' + esc(n.title) + '</p>' +
          '<p class="sub">' + esc(n.author) + ', ' + esc(n.year) + '</p>' +
          '<p class="tags"><span class="badge ' + badgeCls + '">' + esc(slice.replace(' slice', '')) + '</span></p>' +
          '<div class="meta"><span><b class="num">' + n.n_characters + '</b> characters</span><span><b class="num">' + n.n_arcs + '</b> arcs</span><span><b class="num">' + n.n_probes + '</b> probes</span></div>' +
          '</a>';
      });
      box.innerHTML = html;
    }).catch(function () {
      box.innerHTML = '<a class="card" href="explore.html"><p class="title">Open the Explorer</p><p class="sub">The novel list could not be loaded here.</p></a>';
    });
  }

  /* ════════════════════════════════════════════════════════════════
     4. BibTeX copy
     ════════════════════════════════════════════════════════════════ */
  function copyBib() {
    var copy = document.getElementById('copyBib');
    var bib = document.getElementById('bib');
    var status = document.getElementById('copyStatus');
    if (!copy || !bib) return;
    copy.addEventListener('click', function () {
      var txt = bib.textContent;
      function done(ok) {
        copy.textContent = ok ? 'Copied' : 'Select and copy';
        if (status) { status.textContent = ''; setTimeout(function () { status.textContent = ok ? 'BibTeX copied to clipboard.' : 'Copy failed; select the text and copy it manually.'; }, 50); }
        setTimeout(function () { copy.textContent = 'Copy'; }, 1800);
      }
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(txt).then(function () { done(true); }, function () { done(false); });
      else done(false);
    });
  }

  heroPanel();
  mainTable();
  novelCards();
  copyBib();
})();
