// ── State ─────────────────────────────────────────────────────────────────
const state = {
  jobId: null,
  pollTimer: null,
  since: 0,
  totalIter: 30,
  totalSeeds: 3,
  // data[method] = { [seed]: [{iter, best, sel},...] }
  data: { A: {}, B: {}, C: {} },
  runSeed: 'all',
  gpMethod: 'A',
  gpSeed: 0,
  gpIter: 1,
  maxGpIter: 1,
};

const COLORS = { A: '#E8593C', B: '#3B8BD4', C: '#2CA02C' };
const ALPHA  = { A: 'rgba(232,89,60,0.15)', B: 'rgba(59,139,212,0.15)', C: 'rgba(44,160,44,0.15)' };
const NAMES  = { A: 'HM+UCB', B: 'CEI', C: 'HM+EI' };

const plotCfg = { displayModeBar: false, responsive: true };
const darkLayout = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(0,0,0,0)',
  font:   { color: '#8892aa', family: 'Inter, sans-serif', size: 11 },
  xaxis:  { gridcolor: '#2e3245', zerolinecolor: '#2e3245' },
  yaxis:  { gridcolor: '#2e3245', zerolinecolor: '#2e3245' },
  legend: { bgcolor: 'rgba(0,0,0,0)', bordercolor: '#2e3245', borderwidth: 1, font: { size: 10 } },
  margin: { l: 50, r: 20, t: 20, b: 50 },
};

// ── Helpers ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const param = id => parseFloat($(id).value);

function getConfig() {
  return {
    n1_layer:              param('p-n1-layer'),
    n2_layer:              param('p-n2-layer'),
    sigma:                 param('p-sigma'),
    n1_samples:            param('p-n1-samples'),
    pool_grid:             param('p-pool-grid'),
    n1_threshold:          param('p-threshold'),
    n_iterations:          param('p-n-iter'),
    n_seeds:               param('p-n-seeds'),
    ucb_beta:              param('p-ucb-beta'),
    constraint_confidence: param('p-conf'),
  };
}

// ── Tabs ──────────────────────────────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    $('tab-' + btn.dataset.tab).classList.add('active');
  });
});

// ── Landscape ─────────────────────────────────────────────────────────────
async function loadLandscape() {
  const cfg  = getConfig();
  const grid = 50;

  const [rN1, rN2, r3d] = await Promise.all([
    fetch(`/api/landscape?layer=${cfg.n1_layer}&sigma=${cfg.sigma}&grid=${grid}`).then(r => r.json()),
    fetch(`/api/landscape?layer=${cfg.n2_layer}&sigma=${cfg.sigma}&grid=${grid}`).then(r => r.json()),
    fetch(`/api/landscape3d?sigma=${cfg.sigma}&n1=${cfg.n1_layer}&n2=${cfg.n2_layer}`).then(r => r.json()),
  ]);

  const heatLayout = () => ({
    ...darkLayout,
    margin: { l: 50, r: 20, t: 10, b: 50 },
    xaxis: { ...darkLayout.xaxis, title: 'x' },
    yaxis: { ...darkLayout.yaxis, title: 'y' },
  });

  const mkHeat = (d, colorscale) => [{
    type: 'heatmap', x: d.x, y: d.y, z: d.z,
    colorscale, showscale: true,
    colorbar: { thickness: 12, len: 0.8, tickfont: { size: 9 } }
  }];

  Plotly.react('plot-land-n1', mkHeat(rN1, 'Plasma'),  heatLayout(), plotCfg);
  Plotly.react('plot-land-n2', mkHeat(rN2, 'Viridis'), heatLayout(), plotCfg);

  // Cross-section slice at each layer's peak y
  const peakY = n => (n + 1) / n;
  const nearestIdx = (arr, val) =>
    arr.reduce((best, v, i) => Math.abs(v - val) < Math.abs(arr[best] - val) ? i : best, 0);

  const sliceN1 = rN1.z[nearestIdx(rN1.y, peakY(cfg.n1_layer))];
  const sliceN2 = rN2.z[nearestIdx(rN2.y, peakY(cfg.n2_layer))];

  render3DLandscape(r3d);

  Plotly.react('plot-land-slice', [
    { x: rN1.x, y: sliceN1, name: `N1 (layer ${cfg.n1_layer})`, mode: 'lines', line: { color: COLORS.A, width: 2 } },
    { x: rN2.x, y: sliceN2, name: `N2 (layer ${cfg.n2_layer})`, mode: 'lines', line: { color: COLORS.B, width: 2 } },
  ], {
    ...darkLayout,
    xaxis: { ...darkLayout.xaxis, title: 'x' },
    yaxis: { ...darkLayout.yaxis, title: 'Signal' },
    showlegend: true,
  }, plotCfg);
}

function render3DLandscape(d) {
  const layerColor = { [d.n1]: COLORS.A, [d.n2]: COLORS.B };
  const layerLabel = { [d.n1]: `N1 (layer ${d.n1})`, [d.n2]: `N2 (layer ${d.n2})` };

  const traces = d.layers.map(layer => {
    const color = layerColor[layer.n];
    return {
      type: 'surface',
      x: d.x, y: d.y, z: layer.z,
      opacity: 0.82,
      colorscale: [[0, color], [1, color]],
      showscale: false,
      name: layerLabel[layer.n],
      hovertemplate: `<b>${layerLabel[layer.n]}</b><br>x: %{x:.2f}  y: %{y:.2f}  signal: %{z:.3f}<extra></extra>`,
      lighting: { ambient: 0.7, diffuse: 0.9, specular: 0.2 },
    };
  });

  // Invisible scatter traces just for a clean legend
  const legendTraces = d.layers.map(layer => ({
    type: 'scatter3d', mode: 'markers',
    x: [null], y: [null], z: [null],
    name: layerLabel[layer.n],
    marker: { color: layerColor[layer.n], size: 8 },
    showlegend: true,
  }));

  Plotly.react('plot-land-3d', [...traces, ...legendTraces], {
    paper_bgcolor: 'rgba(0,0,0,0)',
    scene: {
      bgcolor: '#1a1d27',
      xaxis: { title: 'x', gridcolor: '#2e3245', color: '#8892aa', tickfont: { size: 9 } },
      yaxis: { title: 'y', gridcolor: '#2e3245', color: '#8892aa', tickfont: { size: 9 } },
      zaxis: { title: 'Signal', gridcolor: '#2e3245', color: '#8892aa', tickfont: { size: 9 } },
      camera: { eye: { x: 1.7, y: 1.7, z: 1.2 } },
    },
    legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(26,29,39,0.8)', bordercolor: '#2e3245',
              borderwidth: 1, font: { color: '#e2e8f0', size: 11 } },
    font:   { color: '#8892aa', family: 'Inter, sans-serif', size: 11 },
    margin: { l: 0, r: 0, t: 0, b: 0 },
    showlegend: true,
  }, { ...plotCfg, displayModeBar: true, modeBarButtonsToRemove: ['toImage'] });
}

$('btn-landscape').addEventListener('click', () => {
  document.querySelector('[data-tab="landscape"]').click();
  loadLandscape();
});

// ── Run Experiment ────────────────────────────────────────────────────────
$('btn-run').addEventListener('click', async () => {
  const cfg = getConfig();
  state.totalIter  = cfg.n_iterations;
  state.totalSeeds = cfg.n_seeds;
  state.data       = { A: {}, B: {}, C: {} };
  state.since      = 0;
  state.jobId      = null;
  state.maxGpIter  = 1;
  state.gpSeed     = 0;
  state.runSeed    = 'all';

  populateSeedSelect(cfg.n_seeds);
  populateRunSeedSelect(cfg.n_seeds);
  $('run-seed-select').disabled = false;

  $('btn-run').disabled           = true;
  $('error-box').style.display    = 'none';
  $('summary-box').style.display  = 'none';
  setStatus('running', 'Starting experiment...');
  initConvergencePlots(cfg.n_iterations, cfg.n1_threshold);

  document.querySelector('[data-tab="run"]').click();

  try {
    const res = await fetch('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    });
    const { job_id } = await res.json();
    state.jobId = job_id;
    startPolling();
  } catch (e) {
    setStatus('error', 'Failed to start experiment');
    $('btn-run').disabled = false;
  }
});

function startPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(pollJob, 1200);
}

async function pollJob() {
  if (!state.jobId) return;
  try {
    const res  = await fetch(`/api/poll/${state.jobId}?since=${state.since}`);
    const body = await res.json();

    if (body.error && !body.events?.length) {
      clearInterval(state.pollTimer);
      setStatus('error', 'Error during run');
      $('error-box').textContent    = body.error;
      $('error-box').style.display  = 'block';
      $('btn-run').disabled         = false;
      return;
    }

    body.events.forEach(ev => {
      const { method, seed, iter, best, sel } = ev;
      if (!state.data[method][seed]) state.data[method][seed] = [];
      state.data[method][seed].push({ iter, best, sel });
      if (seed === 0 && iter > state.maxGpIter) state.maxGpIter = iter;
    });
    state.since = body.total;

    const cur = body.current;
    if (cur.method) {
      const totalEvents = state.totalIter * state.totalSeeds * 3;
      const pct = Math.min(100, Math.round(state.since / totalEvents * 100));
      $('progress-bar').style.width = pct + '%';
      setStatus('running',
        `Method ${cur.method} | seed ${cur.seed + 1}/${state.totalSeeds} | iter ${cur.iter}/${state.totalIter}`);
      $('status-progress').textContent = `${pct}%`;
    }

    if (body.events.length) updateConvergencePlots();

    // Keep GP slider max in sync
    $('gp-iter-slider').max    = state.maxGpIter;
    $('gp-next').disabled      = state.gpIter >= state.maxGpIter;

    if (body.done) {
      clearInterval(state.pollTimer);
      $('progress-bar').style.width    = '100%';
      setStatus('done', 'Experiment complete');
      $('status-progress').textContent = '';
      $('btn-run').disabled            = false;
      updateConvergencePlots(true);
      showSummary();
      $('gp-seed-select').disabled = false;
      $('gp-note').textContent = `GP maps available — ${state.maxGpIter} iterations × ${state.totalSeeds} seeds`;
    }
  } catch (e) {
    console.error('Poll error:', e);
  }
}

// ── Status ────────────────────────────────────────────────────────────────
function setStatus(s, msg) {
  $('status-dot').className    = 'status-dot ' + s;
  $('status-text').textContent = msg;
}

// ── Convergence plots ─────────────────────────────────────────────────────
function initConvergencePlots(nIter, threshold) {
  const thresh = {
    x: [1, nIter], y: [threshold, threshold],
    mode: 'lines', line: { color: '#8892aa', dash: 'dot', width: 1 },
    name: `N1 threshold (${threshold})`, showlegend: true,
  };
  const base = {
    ...darkLayout, showlegend: true,
    xaxis: { ...darkLayout.xaxis, title: 'Iteration', range: [1, nIter] },
  };
  Plotly.react('plot-conv', [thresh], { ...base, yaxis: { ...darkLayout.yaxis, title: 'Best value so far' } }, plotCfg);
  Plotly.react('plot-sel',  [thresh], { ...base, yaxis: { ...darkLayout.yaxis, title: 'Selected score'    } }, plotCfg);
}

function aggregateMethod(method) {
  const seeds = Object.values(state.data[method]);
  if (!seeds.length) return null;
  const maxLen = Math.max(...seeds.map(s => s.length));
  const iters  = Array.from({ length: maxLen }, (_, i) => i + 1);

  const avg = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
  const std = (arr) => {
    const m = avg(arr);
    return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
  };

  const means_best = iters.map(i => { const v = seeds.map(s => s[i-1]?.best).filter(x => x != null); return v.length ? avg(v) : null; });
  const stds_best  = iters.map(i => { const v = seeds.map(s => s[i-1]?.best).filter(x => x != null); return v.length > 1 ? std(v) : 0; });
  const means_sel  = iters.map(i => { const v = seeds.map(s => s[i-1]?.sel ).filter(x => x != null); return v.length ? avg(v) : null; });
  const stds_sel   = iters.map(i => { const v = seeds.map(s => s[i-1]?.sel ).filter(x => x != null); return v.length > 1 ? std(v) : 0; });

  return { iters, means_best, stds_best, means_sel, stds_sel };
}

function buildConvergenceTraces(threshold, nIter, seedFilter = 'all', withBands = false) {
  const thresh = {
    x: [1, nIter], y: [threshold, threshold],
    mode: 'lines', line: { color: '#8892aa', dash: 'dot', width: 1 },
    name: `N1 thr (${threshold})`, showlegend: true,
  };
  const traces_conv = [thresh], traces_sel = [thresh];

  for (const method of ['A', 'B', 'C']) {
    const color = COLORS[method];

    if (seedFilter === 'all') {
      const agg = aggregateMethod(method);
      if (!agg) continue;
      const { iters, means_best, stds_best, means_sel, stds_sel } = agg;
      const alpha = ALPHA[method];
      const band = (means, stds) => ({
        x: [...iters, ...iters.slice().reverse()],
        y: [...means.map((m, i) => m + stds[i]), ...means.map((m, i) => m - stds[i]).reverse()],
        fill: 'toself', fillcolor: alpha,
        line: { color: 'transparent' }, showlegend: false, hoverinfo: 'skip',
      });
      const multiSeed = withBands && Object.keys(state.data[method]).length > 1;
      traces_conv.push(
        { x: iters, y: means_best, mode: 'lines+markers', name: NAMES[method],
          line: { color, width: 2 }, marker: { size: 4 }, showlegend: true },
        ...(multiSeed ? [band(means_best, stds_best)] : [])
      );
      traces_sel.push(
        { x: iters, y: means_sel, mode: 'lines+markers', name: NAMES[method],
          line: { color, width: 2 }, marker: { size: 4 }, showlegend: true },
        ...(multiSeed ? [band(means_sel, stds_sel)] : [])
      );
    } else {
      const seed = parseInt(seedFilter);
      const sd = state.data[method][seed];
      if (!sd?.length) continue;
      traces_conv.push({
        x: sd.map(e => e.iter), y: sd.map(e => e.best),
        mode: 'lines+markers', name: NAMES[method],
        line: { color, width: 2 }, marker: { size: 4 }, showlegend: true,
      });
      traces_sel.push({
        x: sd.map(e => e.iter), y: sd.map(e => e.sel),
        mode: 'lines+markers', name: NAMES[method],
        line: { color, width: 2 }, marker: { size: 4 }, showlegend: true,
      });
    }
  }
  return { traces_conv, traces_sel };
}

function updateConvergencePlots(final = false) {
  const threshold = param('p-threshold');
  const nIter     = param('p-n-iter');
  const { traces_conv, traces_sel } = buildConvergenceTraces(
    threshold, nIter, state.runSeed, final
  );
  const base = { ...darkLayout, showlegend: true, xaxis: { ...darkLayout.xaxis, title: 'Iteration' } };
  Plotly.react('plot-conv', traces_conv, { ...base, yaxis: { ...darkLayout.yaxis, title: 'Best value so far' } }, plotCfg);
  Plotly.react('plot-sel',  traces_sel,  { ...base, yaxis: { ...darkLayout.yaxis, title: 'Selected score'    } }, plotCfg);
}

function populateRunSeedSelect(nSeeds) {
  const sel = $('run-seed-select');
  sel.innerHTML = '<option value="all">Mean ± std (all seeds)</option>';
  for (let i = 0; i < nSeeds; i++) {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `Seed ${i}`;
    sel.appendChild(opt);
  }
  sel.value = 'all';
  state.runSeed = 'all';
}

$('run-seed-select').addEventListener('change', e => {
  state.runSeed = e.target.value;
  updateConvergencePlots(state.jobId && $('status-dot').className.includes('done'));
});


// ── Summary table ─────────────────────────────────────────────────────────
function showSummary() {
  const nSeeds    = param('p-n-seeds');
  const winCounts = { A: 0, B: 0, C: 0 };
  const stats     = {};

  const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
  const std = arr => { const m = avg(arr); return Math.sqrt(arr.reduce((a, b) => a + (b-m)**2, 0) / arr.length); };

  for (const m of ['A', 'B', 'C']) {
    const seeds = Object.values(state.data[m]);
    if (!seeds.length) continue;
    const finals = seeds.map(s => s[s.length - 1]?.best ?? 0);
    const aucs   = seeds.map(s => s.reduce((a, e) => a + e.best, 0));
    stats[m] = {
      finalMean: avg(finals).toFixed(4), finalStd: std(finals).toFixed(4),
      aucMean:   avg(aucs).toFixed(1),   aucStd:   std(aucs).toFixed(1),
    };
  }

  for (let i = 0; i < nSeeds; i++) {
    const scores = {};
    for (const m of ['A', 'B', 'C']) {
      if (state.data[m][i]) scores[m] = state.data[m][i].at(-1)?.best ?? -Infinity;
    }
    const winner = Object.entries(scores).sort((a, b) => b[1] - a[1])[0]?.[0];
    if (winner) winCounts[winner]++;
  }

  const winner = Object.entries(winCounts).sort((a, b) => b[1] - a[1])[0][0];
  const labels = { A: 'A — HM+UCB', B: 'B — CEI', C: 'C — HM+EI' };
  const tbody  = $('summary-table').querySelector('tbody');
  tbody.innerHTML = '';

  for (const m of ['A', 'B', 'C']) {
    if (!stats[m]) continue;
    const tr = document.createElement('tr');
    if (m === winner) tr.className = 'winner-row';
    tr.innerHTML = `
      <td>${labels[m]}${m === winner ? ' ★' : ''}</td>
      <td>${stats[m].finalMean} ± ${stats[m].finalStd}</td>
      <td>${stats[m].aucMean} ± ${stats[m].aucStd}</td>
      <td>${winCounts[m]} / ${nSeeds}</td>
    `;
    tbody.appendChild(tr);
  }
  $('summary-box').style.display = 'block';
}

// ── GP Maps ───────────────────────────────────────────────────────────────
document.querySelectorAll('.method-pill').forEach(pill => {
  pill.addEventListener('click', () => {
    document.querySelectorAll('.method-pill').forEach(p => { p.className = 'method-pill'; });
    const m = pill.dataset.method;
    pill.className = `method-pill active-${m.toLowerCase()}`;
    state.gpMethod = m;
    loadGPMap();
  });
});

function populateSeedSelect(nSeeds) {
  const sel = $('gp-seed-select');
  sel.innerHTML = '';
  for (let i = 0; i < nSeeds; i++) {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = `Seed ${i}`;
    sel.appendChild(opt);
  }
  sel.value = 0;
  state.gpSeed = 0;
}

$('gp-seed-select').addEventListener('change', e => {
  state.gpSeed = parseInt(e.target.value);
  loadGPMap();
});

function setGpIter(val) {
  const slider   = $('gp-iter-slider');
  state.gpIter   = Math.max(1, Math.min(parseInt(slider.max), val));
  slider.value   = state.gpIter;
  $('gp-iter-val').textContent = state.gpIter;
  $('gp-prev').disabled = state.gpIter <= 1;
  $('gp-next').disabled = state.gpIter >= parseInt(slider.max);
  loadGPMap();
}

$('gp-iter-slider').addEventListener('input', e => setGpIter(parseInt(e.target.value)));
$('gp-prev').addEventListener('click', () => setGpIter(state.gpIter - 1));
$('gp-next').addEventListener('click', () => setGpIter(state.gpIter + 1));

async function loadGPMap() {
  if (!state.jobId) return;
  try {
    const url  = `/api/gp_map/${state.jobId}?method=${state.gpMethod}&seed=${state.gpSeed}&iter=${state.gpIter}`;
    const data = await fetch(url).then(r => r.json());
    if (data.error) return;
    renderGPMaps(data);
  } catch (e) {
    console.error('GP map error:', e);
  }
}

function renderGPMaps(d) {
  const queried = {
    x: d.qx, y: d.qy, mode: 'markers',
    marker: { color: '#fff', size: 5, symbol: 'circle', line: { color: '#333', width: 1 } },
    name: 'Queried', showlegend: false,
  };
  const baseHeat = {
    ...darkLayout,
    margin: { l: 50, r: 20, t: 10, b: 50 },
    xaxis: { ...darkLayout.xaxis, title: 'x' },
    yaxis: { ...darkLayout.yaxis, title: 'y', scaleanchor: 'x', scaleratio: 1 },
  };

  Plotly.react('plot-gp-mean', [
    { type: 'heatmap', x: d.x, y: d.y, z: d.mean, colorscale: 'RdBu', zmid: 0,
      colorbar: { thickness: 12, len: 0.8, tickfont: { size: 9 } } },
    queried,
  ], baseHeat, plotCfg);

  Plotly.react('plot-gp-std', [
    { type: 'heatmap', x: d.x, y: d.y, z: d.std, colorscale: 'Magma',
      colorbar: { thickness: 12, len: 0.8, tickfont: { size: 9 } } },
    queried,
  ], baseHeat, plotCfg);
}

// ── Init ──────────────────────────────────────────────────────────────────
loadLandscape();
