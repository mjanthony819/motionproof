/* MotionProof core: feature extraction, models, perturbations, windows, CSV validation. No dependencies. */
(function (root) {
  'use strict';
  var W = 100, FS = 50, C = 6, K = 9;
  var NAMES = ['walking','running','shuffling','stairs up','stairs down','standing','sitting','lying','cycling'];
  var BANDS = [[0.5,1],[1,2],[2,3.5],[3.5,6],[6,12]];
  // DFT basis per band (bins k where lo <= k*FS/W < hi)
  var BASIS = BANDS.map(function (b) {
    var ks = [];
    for (var k = 1; k <= W / 2; k++) { var f = k * FS / W; if (f >= b[0] && f < b[1]) ks.push(k); }
    return ks.map(function (k) {
      var cs = new Float64Array(W), sn = new Float64Array(W);
      for (var n = 0; n < W; n++) { cs[n] = Math.cos(2 * Math.PI * k * n / W); sn[n] = Math.sin(2 * Math.PI * k * n / W); }
      return [cs, sn];
    });
  });
  // window layout: Float64Array length W*C, row-major [t][c]
  function features(x) {
    var out = [], c, t;
    var mean = new Float64Array(C), std = new Float64Array(C), mn = new Float64Array(C), mx = new Float64Array(C), jerk = new Float64Array(C);
    for (c = 0; c < C; c++) {
      var s = 0, lo = Infinity, hi = -Infinity, jj = 0;
      for (t = 0; t < W; t++) { var v = x[t * C + c]; s += v; if (v < lo) lo = v; if (v > hi) hi = v; if (t > 0) jj += Math.abs(v - x[(t - 1) * C + c]); }
      mean[c] = s / W; mn[c] = lo; mx[c] = hi; jerk[c] = jj / (W - 1);
      var q = 0; for (t = 0; t < W; t++) { var d = x[t * C + c] - mean[c]; q += d * d; } std[c] = Math.sqrt(q / W);
    }
    [mean, std, mn, mx, jerk].forEach(function (a) { for (c = 0; c < C; c++) out.push(a[c]); });
    [0, 3].forEach(function (a) {
      var m = new Float64Array(W), s = 0; 
      for (t = 0; t < W; t++) { m[t] = Math.sqrt(Math.pow(x[t*C+a],2) + Math.pow(x[t*C+a+1],2) + Math.pow(x[t*C+a+2],2)); s += m[t]; }
      var mu = s / W, q = 0; for (t = 0; t < W; t++) q += (m[t] - mu) * (m[t] - mu);
      out.push(mu); out.push(Math.sqrt(q / W));
    });
    var tot = new Float64Array(C).fill(1e-9), pw = [];
    for (var b = 0; b < BANDS.length; b++) {
      var p = new Float64Array(C);
      for (c = 0; c < C; c++) {
        var acc = 0;
        for (var k = 0; k < BASIS[b].length; k++) {
          var re = 0, im = 0, cs = BASIS[b][k][0], sn = BASIS[b][k][1];
          for (t = 0; t < W; t++) { var xc = x[t * C + c] - mean[c]; re += xc * cs[t]; im += xc * sn[t]; }
          acc += re * re + im * im;
        }
        p[c] = acc / W; tot[c] += p[c];
      }
      pw.push(p);
    }
    pw.forEach(function (p) { for (c = 0; c < C; c++) out.push(p[c] / tot[c]); });
    for (var i = 0; i < 3; i++) {
      var sab = 0; for (t = 0; t < W; t++) sab += (x[t*C+i] - mean[i]) * (x[t*C+3+i] - mean[3+i]);
      out.push((sab / W) / (std[i] * std[3 + i] + 1e-6));
    }
    return out;
  }
  function softmaxT(z, T) {
    var m = -Infinity, i; for (i = 0; i < z.length; i++) if (z[i] > m) m = z[i];
    var e = new Array(z.length), s = 0; for (i = 0; i < z.length; i++) { e[i] = Math.exp((z[i] - m) / T); s += e[i]; }
    for (i = 0; i < z.length; i++) e[i] /= s; return e;
  }
  function lrLogits(M, x) {
    var f = features(x), d = f.length, o = new Array(K);
    for (var k = 0; k < K; k++) { var s = M.b[k]; for (var j = 0; j < d; j++) s += M.W[k][j] * (f[j] - M.mu[j]) / M.sd[j]; o[k] = s; }
    return o;
  }
  function conv(inp, L, cin, w, b, cout, ks, pad) {
    // inp [cin][L] -> [cout][L]
    var out = [];
    for (var o = 0; o < cout; o++) {
      var r = new Float64Array(L);
      for (var t = 0; t < L; t++) {
        var s = b[o];
        for (var i = 0; i < cin; i++) { var wi = w[o][i], xi = inp[i];
          for (var q = 0; q < ks; q++) { var tt = t + q - pad; if (tt >= 0 && tt < L) s += wi[q] * xi[tt]; } }
        r[t] = s > 0 ? s : 0;
      }
      out.push(r);
    }
    return out;
  }
  function pool2(a, L) { var h = L >> 1; return a.map(function (r) { var o = new Float64Array(h); for (var i = 0; i < h; i++) o[i] = Math.max(r[2*i], r[2*i+1]); return o; }); }
  function cnnLogits(M, x) {
    var inp = []; for (var c = 0; c < C; c++) { var r = new Float64Array(W); for (var t = 0; t < W; t++) r[t] = (x[t*C+c] - M.mu[c]) / M.sd[c]; inp.push(r); }
    var L = W, a = inp, cin = C;
    var ks = [5, 5, 3];
    for (var l = 0; l < 3; l++) {
      var Ly = M.layers[l]; var cout = Ly.b.length;
      a = conv(a, L, cin, Ly.w, Ly.b, cout, ks[l], (ks[l] - 1) / 2);
      if (l < 2) { a = pool2(a, L); L = L >> 1; }
      cin = cout;
    }
    var feat = new Array(cin * 2);
    for (var j = 0; j < cin; j++) { var s = 0, m = -Infinity; for (var u = 0; u < L; u++) { s += a[j][u]; if (a[j][u] > m) m = a[j][u]; } feat[j] = s / L; feat[cin + j] = m; }
    var o = new Array(K);
    for (var k = 0; k < K; k++) { var z = M.fcb[k]; for (var f = 0; f < feat.length; f++) z += M.fcW[k][f] * feat[f]; o[k] = z; }
    return o;
  }
  function logits(M, x) { return M.type === 'lr' ? lrLogits(M, x) : cnnLogits(M, x); }
  function predict(M, x) { var z = logits(M, x), p = softmaxT(z, M.T), a = 0; for (var i = 1; i < K; i++) if (p[i] > p[a]) a = i; return { logits: z, probs: p, label: a, conf: p[a] }; }

  // ---- perturbations on a continuous stream Float32Array length n*C (row-major) ----
  function mulberry(seed) { var a = seed >>> 0; return function () { a = (a + 0x6D2B79F5) >>> 0; var t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { var u = 1 - r(), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function rotate(x, ch0, deg, axis) {
    var th = deg * Math.PI / 180, c = Math.cos(th), s = Math.sin(th), ij = [[1,2],[0,2],[0,1]][axis], i = ij[0], j = ij[1];
    var n = x.length / C, y = Float64Array.from(x);
    for (var t = 0; t < n; t++) { var a = x[t*C+ch0+i], b = x[t*C+ch0+j]; y[t*C+ch0+i] = c*a - s*b; y[t*C+ch0+j] = s*a + c*b; }
    return y;
  }
  function stretchWin(x, f) { // per-window time stretch (python parity)
    var y = new Float64Array(x.length);
    for (var t = 0; t < W; t++) { var tt = Math.min(Math.max(t * f, 0), W - 1), i0 = Math.floor(tt), i1 = Math.min(i0 + 1, W - 1), fr = tt - i0;
      for (var c = 0; c < C; c++) y[t*C+c] = x[i0*C+c] * (1 - fr) + x[i1*C+c] * fr; }
    return y;
  }
  var PERTURBATIONS = {
    noise: { label: 'Gaussian noise (g, sigma)', min: 0, max: 0.3, step: 0.01, def: 0.1 },
    gain: { label: 'Sensor gain (x)', min: 0.7, max: 1.3, step: 0.05, def: 1.15 },
    thigh_rot: { label: 'Thigh sensor rotated (degrees)', min: 0, max: 60, step: 5, def: 30 },
    back_rot: { label: 'Back sensor rotated (degrees)', min: 0, max: 60, step: 5, def: 30 },
    drop_thigh: { label: 'Thigh sensor lost', min: 0, max: 0, step: 1, def: 0 },
    drop_back: { label: 'Back sensor lost', min: 0, max: 0, step: 1, def: 0 },
    stretch: { label: 'Time stretch (movement speed seen differently)', min: 0.7, max: 1.4, step: 0.05, def: 1.25 },
    clip: { label: 'Clip at +/- g (cheap sensor range)', min: 1, max: 4, step: 0.5, def: 2 }
  };
  // apply to a window (Float64Array W*C); rng only for noise
  function perturbWindow(x, kind, v, rng) {
    var n = x.length, y, i;
    if (kind === 'clean') return x;
    if (kind === 'noise') { y = new Float64Array(n); for (i = 0; i < n; i++) y[i] = x[i] + gauss(rng) * v; return y; }
    if (kind === 'gain') { y = new Float64Array(n); for (i = 0; i < n; i++) y[i] = x[i] * v; return y; }
    if (kind === 'thigh_rot') return rotate(x, 3, v, 1);
    if (kind === 'back_rot') return rotate(x, 0, v, 1);
    if (kind === 'drop_thigh') { y = Float64Array.from(x); for (i = 0; i < n; i += C) { y[i+3] = 0; y[i+4] = 0; y[i+5] = 0; } return y; }
    if (kind === 'drop_back') { y = Float64Array.from(x); for (i = 0; i < n; i += C) { y[i] = 0; y[i+1] = 0; y[i+2] = 0; } return y; }
    if (kind === 'stretch') return stretchWin(x, v);
    if (kind === 'clip') { y = new Float64Array(n); for (i = 0; i < n; i++) y[i] = Math.max(-v, Math.min(v, x[i])); return y; }
    throw new Error('unknown perturbation ' + kind);
  }
  // ---- stream windows ----
  function windowsOf(stream, n, stride) {
    var out = [];
    for (var s = 0; s + W <= n; s += (stride || 50)) out.push({ start: s, data: Float64Array.from(stream.subarray(s * C, (s + W) * C)) });
    return out;
  }
  function decodeStream(b64) {
    var bin = typeof atob === 'function' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
    var n = bin.length / 2, out = new Float32Array(n);
    for (var i = 0; i < n; i++) { var lo = bin.charCodeAt(2*i), hi = bin.charCodeAt(2*i+1), v = (hi << 8) | lo; if (v & 0x8000) v -= 0x10000; out[i] = v / 1000; }
    return out;
  }
  // ---- CSV import + validation ----
  var COLS = ['back_x','back_y','back_z','thigh_x','thigh_y','thigh_z'];
  function parseCSV(text, opts) {
    opts = opts || {};
    var rep = { ok: false, errors: [], warnings: [], info: {}, data: null };
    var lines = text.replace(/\r/g, '').split('\n').filter(function (l) { return l.trim().length; });
    if (!lines.length) { rep.errors.push('The file is empty.'); return rep; }
    var delim = lines[0].indexOf(',') >= 0 ? ',' : (lines[0].indexOf('\t') >= 0 ? '\t' : (lines[0].indexOf(';') >= 0 ? ';' : ','));
    var head = lines[0].split(delim).map(function (s) { return s.trim().replace(/^"|"$/g, ''); });
    var hasHeader = head.some(function (h) { return isNaN(Number(h)) ; });
    var idx = [], tsIdx = -1, start = hasHeader ? 1 : 0;
    if (hasHeader) {
      var low = head.map(function (h) { return h.toLowerCase(); });
      COLS.forEach(function (c) { idx.push(low.indexOf(c)); });
      tsIdx = low.indexOf('timestamp'); if (tsIdx < 0) tsIdx = low.indexOf('time');
      var miss = COLS.filter(function (c, i) { return idx[i] < 0; });
      if (miss.length) { rep.errors.push('Missing columns: ' + miss.join(', ') + '. Expected header: ' + COLS.join(',') + ' (optional timestamp column).'); return rep; }
    } else {
      if (head.length < 6) { rep.errors.push('Found ' + head.length + ' columns without a header; need 6 (back x,y,z then thigh x,y,z).'); return rep; }
      idx = [0,1,2,3,4,5]; rep.warnings.push('No header row: assuming columns are back_x,back_y,back_z,thigh_x,thigh_y,thigh_z in g.');
    }
    var n = lines.length - start, arr = new Float32Array(n * C), bad = 0, ts = tsIdx >= 0 ? new Float64Array(n) : null, used = 0;
    for (var r = 0; r < n; r++) {
      var p = lines[start + r].split(delim), ok = true;
      for (var c = 0; c < C; c++) { var v = Number(p[idx[c]]); if (!isFinite(v) || p[idx[c]] === undefined || p[idx[c]].trim() === '') { ok = false; break; } arr[used * C + c] = v; }
      if (!ok) { bad++; continue; }
      if (ts) { var tv = Number(p[tsIdx]); var tn = isFinite(tv) ? tv : Date.parse(p[tsIdx]) / 1000; ts[used] = tn; }
      used++;
    }
    if (bad) rep.warnings.push(bad + ' row(s) with missing or non-numeric values were skipped.');
    if (used < W) { rep.errors.push('Only ' + used + ' usable rows; need at least ' + W + ' (2 seconds at 50 Hz).'); return rep; }
    var data = arr.subarray(0, used * C), fs = FS;
    if (ts) {
      var dts = []; for (var i = 1; i < used; i++) dts.push(ts[i] - ts[i - 1]);
      dts.sort(function (a, b) { return a - b; }); var md = dts[dts.length >> 1];
      if (md > 0) { fs = 1 / md; rep.info.estimatedHz = Math.round(fs * 10) / 10; }
      if (md > 0 && Math.abs(fs - FS) / FS > 0.1) {
        // resample to 50 Hz by linear interpolation on timestamps
        var T0 = ts[0], T1 = ts[used - 1], m = Math.floor((T1 - T0) * FS), out = new Float32Array(m * C), j = 0;
        for (var q = 0; q < m; q++) { var tq = T0 + q / FS; while (j < used - 2 && ts[j + 1] < tq) j++; var span = ts[j + 1] - ts[j] || 1, fr = Math.min(Math.max((tq - ts[j]) / span, 0), 1);
          for (var cc = 0; cc < C; cc++) out[q * C + cc] = data[j * C + cc] * (1 - fr) + data[(j + 1) * C + cc] * fr; }
        data = out; used = m; rep.warnings.push('Sampling rate looks like ' + rep.info.estimatedHz + ' Hz, so the stream was resampled to 50 Hz (linear interpolation).');
      }
    } else rep.warnings.push('No timestamp column: assuming 50 Hz sampling.');
    // units: magnitudes of each sensor
    var magSum = [0, 0], cnt = Math.min(used, 5000);
    for (var t2 = 0; t2 < cnt; t2++) for (var s = 0; s < 2; s++) magSum[s] += Math.sqrt(Math.pow(data[t2*C+3*s],2) + Math.pow(data[t2*C+3*s+1],2) + Math.pow(data[t2*C+3*s+2],2));
    var mm = [magSum[0] / cnt, magSum[1] / cnt]; rep.info.meanMagnitude = mm.map(function (v) { return Math.round(v * 100) / 100; });
    if (mm[0] > 5 && mm[1] > 5 && opts.autoConvert !== false) {
      var g = 9.80665; data = Float32Array.from(data, function (v) { return v / g; });
      rep.warnings.push('Mean acceleration magnitude is about ' + mm.map(function (v) { return v.toFixed(1); }).join(' / ') + ', which looks like m/s^2. Values were divided by 9.81 to convert to g.');
    } else if (mm[0] < 0.3 || mm[1] < 0.3 || mm[0] > 3 || mm[1] > 3) rep.warnings.push('Mean acceleration magnitude is ' + mm.map(function (v) { return v.toFixed(2); }).join(' / ') + ' g (expected close to 1 g when the body is still). Check units and column order; predictions may be unreliable.');
    var mxv = 0; for (var u = 0; u < Math.min(used * C, 200000); u++) mxv = Math.max(mxv, Math.abs(data[u]));
    if (mxv > 16) rep.warnings.push('Values reach ' + mxv.toFixed(1) + ' (beyond a 16 g sensor range). Check units.');
    rep.info.rows = used; rep.info.seconds = Math.round(used / FS * 10) / 10; rep.info.hasTimestamp = !!ts; rep.info.usedHz = FS;
    rep.ok = true; rep.data = data; rep.n = used; return rep;
  }
  function rollingMajority(labels, confs, n) { return labels.map(function (_, i) { var c = new Array(K).fill(0); for (var j = Math.max(0, i - n + 1); j <= i; j++) c[labels[j]] += 1; var b = 0; for (var k = 1; k < K; k++) if (c[k] > c[b]) b = k; return b; }); }
  var api = { W: W, FS: FS, C: C, K: K, NAMES: NAMES, COLS: COLS, features: features, logits: logits, predict: predict, softmaxT: softmaxT, perturbWindow: perturbWindow, PERTURBATIONS: PERTURBATIONS, mulberry: mulberry, windowsOf: windowsOf, decodeStream: decodeStream, parseCSV: parseCSV, rollingMajority: rollingMajority };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.MP = api;
})(typeof window !== 'undefined' ? window : globalThis);
