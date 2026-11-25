export const VALUE_MIN = 1;
export const VALUE_MAX = 355;
export const DIMENSION = VALUE_MAX - VALUE_MIN + 1;
export const ALPHA = 0.5;
export const ORDERED_BLOCK_WEIGHT = 0.25;

export const ACCEPT_TAU_ABS = 0.46;
export const ACCEPT_DOM_RIVAL = 1.65;
export const ACCEPT_DOM_U = 1.40;

export const OPERATING_POINTS = {
  relaxed: { id: "relaxed", label: "宽松召回档", tau_abs: 0.40, dom_rival: 1.80, dom_u: 1.20 },
  default: { id: "default", label: "默认平衡档", tau_abs: 0.46, dom_rival: 1.65, dom_u: 1.40 },
  strict: { id: "strict", label: "严格高精档", tau_abs: 0.60, dom_rival: 3.50, dom_u: 2.00 },
};

const FAMILY_DISPLAY_NAMES = {
  gpt: "GPT",
  claude: "Claude",
};

const GAMMA_COND_VMF = 0.38;
const VMF_WEIGHTS_6 = [0.22, 0.15, 0.15, 0.15, 0.16, 0.17];
const U_UNIFORM_WEIGHT = 0.20;
const U_SIBLING_WEIGHT = 0.80;
const ALPHA_SIBLING = 0.65;
const SIBLING_CLIP = 6.0;
const RES_SCALES = [16.0, 64.0, 1e10];
const LAG_BINS_EDGE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 400];

function rangePush(target, lo, hi) {
  for (let i = lo; i < hi; i += 1) target.push(i);
}

const STABLE_CDF_COLS = (() => {
  const cols = [];
  rangePush(cols, 0, 64);
  rangePush(cols, 86, 154);
  rangePush(cols, 195, 225);
  return Int32Array.from(cols);
})();

const SUBLIMINAL_COLS = (() => {
  const cols = [];
  rangePush(cols, 0, 12);
  rangePush(cols, 22, 44);
  rangePush(cols, 54, 64);
  rangePush(cols, 86, 96);
  rangePush(cols, 96, 104);
  rangePush(cols, 113, 125);
  rangePush(cols, 125, 133);
  rangePush(cols, 142, 154);
  rangePush(cols, 195, 225);
  return Int32Array.from(cols);
})();

const MACRO_COLS = (() => {
  const subSet = new Set(SUBLIMINAL_COLS);
  const cols = [];
  for (let c = 0; c < 225; c += 1) {
    if (!subSet.has(c)) cols.push(c);
  }
  return Int32Array.from(cols);
})();

// ============================================================================
// 1. Base64 Tensor Unpackers (IEEE-754 Bit-Exact with Python v6_portable.py)
// ============================================================================

function decodeBase64Bytes(b64) {
  if (!b64) return new Uint8Array(0);
  if (typeof Buffer !== "undefined") {
    const buf = Buffer.from(b64, "base64");
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) out[i] = bin.charCodeAt(i);
  return out;
}

function unpackF64(val) {
  if (val && typeof val === "object" && typeof val.f32 === "string") {
    const bytes = decodeBase64Bytes(val.f32);
    const n = bytes.byteLength >>> 2;
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const out = new Float64Array(n);
    for (let i = 0; i < n; i += 1) out[i] = view.getFloat32(i * 4, true);
    return out;
  }
  const bytes = decodeBase64Bytes(val);
  const n = bytes.byteLength >>> 3;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 1) out[i] = view.getFloat64(i * 8, true);
  return out;
}

function unpackI32(b64) {
  const bytes = decodeBase64Bytes(b64);
  const n = bytes.byteLength >>> 2;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Int32Array(n);
  for (let i = 0; i < n; i += 1) out[i] = view.getInt32(i * 4, true);
  return out;
}

function unpackMatI16(obj) {
  const rows = Number(obj.shape[0]);
  const cols = Number(obj.shape[1]);
  const data = new Float64Array(rows * cols);
  if (rows === 0 || cols === 0) return { rows, cols, data };
  const sBytes = decodeBase64Bytes(obj.scales);
  const sView = new DataView(sBytes.buffer, sBytes.byteOffset, sBytes.byteLength);
  const qBytes = decodeBase64Bytes(obj.data);
  const qView = new DataView(qBytes.buffer, qBytes.byteOffset, qBytes.byteLength);
  const isI8 = Number(obj.bits || 16) === 8;
  for (let r = 0; r < rows; r += 1) {
    const sc = sView.getFloat32(r * 4, true);
    const rowOff = r * cols;
    if (isI8) {
      for (let c = 0; c < cols; c += 1) {
        data[rowOff + c] = qView.getInt8(rowOff + c) * sc;
      }
    } else {
      for (let c = 0; c < cols; c += 1) {
        data[rowOff + c] = qView.getInt16((rowOff + c) * 2, true) * sc;
      }
    }
  }
  return { rows, cols, data };
}

// ============================================================================
// 2. Static Lookup Tables (Built Once on Module Load)
// ============================================================================

function legPolyMatrix(x, maxDeg) {
  const n = x.length;
  const out = new Float64Array(n * maxDeg);
  const pKm1 = new Float64Array(n);
  const pK = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    pKm1[i] = 1.0;
    pK[i] = x[i];
    if (maxDeg >= 1) out[i * maxDeg] = x[i];
  }
  for (let k = 1; k < maxDeg; k += 1) {
    const c1 = 2.0 * k + 1.0;
    const invKp1 = 1.0 / (k + 1.0);
    for (let i = 0; i < n; i += 1) {
      const pKp1 = (c1 * x[i] * pK[i] - k * pKm1[i]) * invKp1;
      out[i * maxDeg + k] = pKp1;
      pKm1[i] = pK[i];
      pK[i] = pKp1;
    }
  }
  return out;
}

function makeLexPerms(m) {
  const res = [];
  const used = new Array(m).fill(false);
  const cur = [];
  function dfs() {
    if (cur.length === m) {
      res.push(cur.slice());
      return;
    }
    for (let i = 0; i < m; i += 1) {
      if (!used[i]) {
        used[i] = true;
        cur.push(i);
        dfs();
        cur.pop();
        used[i] = false;
      }
    }
  }
  dfs();
  return res;
}

const STATIC = (() => {
  const coord = new Float64Array(355);
  const xGrid = new Float64Array(355);
  const hundGrid = new Float64Array(355);
  for (let i = 0; i < 355; i += 1) {
    const v = i + 1;
    coord[i] = 2.0 * (v - 0.5) / 355.0 - 1.0;
    xGrid[i] = 2.0 * (v - 1.0) / 354.0 - 1.0;
    hundGrid[i] = 2.0 * Math.floor(v / 100) / 3.0 - 1.0;
  }

  const leg12 = legPolyMatrix(coord, 12);
  const resBasis = new Float64Array(355 * 32);
  for (let i = 0; i < 355; i += 1) {
    const v = i + 1;
    const off = i * 32;
    for (let k = 0; k < 12; k += 1) resBasis[off + k] = leg12[i * 12 + k];
    resBasis[off + 12 + (v % 10)] = 1.0;
    resBasis[off + 22 + (Math.floor(v / 10) % 10)] = 1.0;
  }

  const leg8 = legPolyMatrix(xGrid, 8);
  const legH3 = legPolyMatrix(hundGrid, 3);
  const margBasis = new Float64Array(355 * 29);
  for (let i = 0; i < 355; i += 1) {
    const v = i + 1;
    const u = v % 10;
    const t = Math.floor(v / 10) % 10;
    const off = i * 29;
    let p = 0;
    for (let k = 0; k < 8; k += 1) margBasis[off + p++] = leg8[i * 8 + k];
    for (let k = 1; k <= 5; k += 1) margBasis[off + p++] = Math.cos(2.0 * Math.PI * k * u / 10.0);
    for (let k = 1; k <= 4; k += 1) margBasis[off + p++] = Math.sin(2.0 * Math.PI * k * u / 10.0);
    for (let k = 1; k <= 5; k += 1) margBasis[off + p++] = Math.cos(2.0 * Math.PI * k * t / 10.0);
    for (let k = 1; k <= 4; k += 1) margBasis[off + p++] = Math.sin(2.0 * Math.PI * k * t / 10.0);
    for (let k = 0; k < 3; k += 1) margBasis[off + p++] = legH3[i * 3 + k];
  }

  const baseIdx = Array.from({ length: 355 }, (_, i) => i);
  const p0 = Int32Array.from(baseIdx);
  const p1 = Int32Array.from(baseIdx.slice().sort((a, b) => (((a + 1) % 10) * 1000 + a) - (((b + 1) % 10) * 1000 + b)));
  const p2 = Int32Array.from(baseIdx.slice().sort((a, b) => ((Math.floor((a + 1) / 10) % 10) * 1000 + a) - ((Math.floor((b + 1) / 10) % 10) * 1000 + b)));
  const cdfPerms = [p0, p1, p2];
  const invCdfPerms = cdfPerms.map((perm) => {
    const inv = new Int32Array(355);
    for (let i = 0; i < 355; i += 1) inv[perm[i]] = i;
    return inv;
  });

  const jumpOrder = new Int16Array(355 * 355);
  const invJumpOrder = new Int16Array(355 * 355);
  for (let pv = 0; pv < 355; pv += 1) {
    const jo = baseIdx.slice().sort((a, b) => (Math.abs(a - pv) * 1000 + a) - (Math.abs(b - pv) * 1000 + b));
    const rowOff = pv * 355;
    for (let j = 0; j < 355; j += 1) {
      jumpOrder[rowOff + j] = jo[j];
      invJumpOrder[rowOff + jo[j]] = j;
    }
  }

  const markovB = new Float64Array(355 * 355 * 11);
  const distMat = new Float64Array(355 * 355);
  const distSqMat = new Float64Array(355 * 355);
  for (let u = 0; u < 355; u += 1) {
    for (let v = 0; v < 355; v += 1) {
      const d = Math.abs(v - u);
      const dn = d / 355.0;
      const uv = u * 355 + v;
      distMat[uv] = dn;
      distSqMat[uv] = dn * dn;
      const bo = uv * 11;
      markovB[bo + 0] = d === 0 ? 1.0 : 0.0;
      markovB[bo + 1] = d === 1 ? 1.0 : 0.0;
      markovB[bo + 2] = d === 2 ? 1.0 : 0.0;
      markovB[bo + 3] = (d >= 3 && d <= 5) ? 1.0 : 0.0;
      markovB[bo + 4] = (d >= 6 && d <= 12) ? 1.0 : 0.0;
      markovB[bo + 5] = (d >= 13 && d <= 30) ? 1.0 : 0.0;
      markovB[bo + 6] = dn;
      markovB[bo + 7] = v > u ? 1.0 : 0.0;
      markovB[bo + 8] = (v % 10) === (u % 10) ? 1.0 : 0.0;
      markovB[bo + 9] = (v % 10) === ((u + 1) % 10) ? 1.0 : 0.0;
      markovB[bo + 10] = Math.floor(v / 100) === Math.floor(u / 100) ? 1.0 : 0.0;
    }
  }

  const lagLut = new Int8Array(52);
  lagLut[2] = 1;
  lagLut[3] = 2;
  for (let i = 4; i < 7; i += 1) lagLut[i] = 3;
  for (let i = 7; i < 13; i += 1) lagLut[i] = 4;
  for (let i = 13; i < 25; i += 1) lagLut[i] = 5;
  for (let i = 25; i < 51; i += 1) lagLut[i] = 6;

  const hCum = new Float64Array(355);
  let hs = 0.0;
  for (let i = 0; i < 355; i += 1) {
    hs += 1.0 / (355.0 - i);
    hCum[i] = hs;
  }

  const isPrime = new Float64Array(355);
  isPrime.fill(1.0);
  isPrime[0] = 0.0;
  for (let v = 2; v <= 355; v += 1) {
    const rLim = Math.floor(Math.sqrt(v));
    for (let d = 2; d <= rLim; d += 1) {
      if (v % d === 0) {
        isPrime[v - 1] = 0.0;
        break;
      }
    }
  }

  const hund = new Int32Array(355);
  const ten = new Int32Array(355);
  const unit = new Int32Array(355);
  const dlen = new Int32Array(355);
  const dsum = new Int32Array(355);
  const isRepdigit = new Float64Array(355);
  const isPalin = new Float64Array(355);
  const isAsc = new Float64Array(355);
  const isDesc = new Float64Array(355);
  for (let i = 0; i < 355; i += 1) {
    const v = i + 1;
    hund[i] = Math.floor(v / 100);
    ten[i] = Math.floor(v / 10) % 10;
    unit[i] = v % 10;
    dlen[i] = (v >= 10 ? 1 : 0) + (v >= 100 ? 1 : 0);
    dsum[i] = hund[i] + ten[i] + unit[i];
    const s = String(v);
    if (s.length >= 2) {
      if (new Set(s).size === 1) isRepdigit[i] = 1.0;
      let pal = true;
      let asc = true;
      let desc = true;
      for (let k = 0; k < s.length - 1; k += 1) {
        if (s[k] !== s[s.length - 1 - k]) pal = false;
        if (!(s[k] < s[k + 1])) asc = false;
        if (!(s[k] > s[k + 1])) desc = false;
      }
      if (pal) isPalin[i] = 1.0;
      if (asc) isAsc[i] = 1.0;
      if (desc) isDesc[i] = 1.0;
    }
  }

  const stepEdges = [
    -200, -135, -90, -60, -40, -26, -17, -11, -7, -4, -3, -2, -1, 0,
    1, 2, 3, 5, 8, 12, 18, 27, 41, 61, 91, 136, 201,
  ];
  const stepLut = new Int32Array(709);
  for (let d = -354; d <= 354; d += 1) {
    let b = 0;
    while (b < stepEdges.length && stepEdges[b] <= d) b += 1;
    stepLut[d + 354] = b;
  }

  const proxRads = [1, 2, 5, 12, 25];
  const proxLo = proxRads.map((rad) => Int32Array.from({ length: 355 }, (_, z) => Math.max(z - rad, 0)));
  const proxHi = proxRads.map((rad) => Int32Array.from({ length: 355 }, (_, z) => Math.min(z + rad + 1, 355)));

  return {
    resBasis,
    margBasis,
    cdfPerms,
    invCdfPerms,
    jumpOrder,
    invJumpOrder,
    markovB,
    distMat,
    distSqMat,
    lagLut,
    hCum,
    perms3: makeLexPerms(3),
    perms4: makeLexPerms(4),
    isPrime,
    hund,
    ten,
    unit,
    dlen,
    dsum,
    isRepdigit,
    isPalin,
    isAsc,
    isDesc,
    stepLut,
    proxLo,
    proxHi,
  };
})();

// ============================================================================
// 3. Parsing & Basic Math Helpers
// ============================================================================

export function parseNumbers(text) {
  const runs = [];
  let current = [];
  let previousEnd = 0;
  const str = String(text);
  for (const match of str.matchAll(/\d+/g)) {
    const separator = str.slice(previousEnd, match.index);
    const value = Number(match[0]);
    if (current.length && /\p{L}/u.test(separator)) {
      runs.push(current);
      current = [];
    }
    if (value >= VALUE_MIN && value <= VALUE_MAX) current.push(value);
    previousEnd = match.index + match[0].length;
  }
  if (current.length) runs.push(current);
  return runs.reduce((best, run) => (run.length > best.length ? run : best), []);
}

export function countNumbers(numbers) {
  const counts = Array(DIMENSION).fill(0);
  for (let i = 0; i < numbers.length; i += 1) counts[numbers[i] - VALUE_MIN] += 1;
  return counts;
}

function jsSimilarity(left, right) {
  let leftTotal = 0;
  let rightTotal = ALPHA * DIMENSION;
  for (let i = 0; i < DIMENSION; i += 1) {
    leftTotal += left[i];
    rightTotal += right[i];
  }
  let js = 0.0;
  for (let i = 0; i < DIMENSION; i += 1) {
    const p = left[i] / leftTotal;
    const q = (right[i] + ALPHA) / rightTotal;
    const m = 0.5 * (p + q);
    if (p > 0) js += 0.5 * p * Math.log(p / m);
    if (q > 0) js += 0.5 * q * Math.log(q / m);
  }
  return 1.0 - Math.sqrt(Math.max(js, 0.0) / Math.LN2);
}

function meanArr(a) {
  const n = a.length;
  if (n === 0) return 0.0;
  let s = 0.0;
  for (let i = 0; i < n; i += 1) s += a[i];
  return s / n;
}

function stdArr(a) {
  const n = a.length;
  if (n === 0) return 0.0;
  const m = meanArr(a);
  let s = 0.0;
  for (let i = 0; i < n; i += 1) {
    const d = a[i] - m;
    s += d * d;
  }
  return Math.sqrt(s / n);
}

function dotArr(a, b) {
  let s = 0.0;
  for (let i = 0; i < a.length; i += 1) s += a[i] * b[i];
  return s;
}

function normArr(a) {
  return Math.sqrt(dotArr(a, a));
}

function corrcoef1d(a, b) {
  const n = a.length;
  if (n < 2) return 0.0;
  const ma = meanArr(a);
  const mb = meanArr(b);
  let sab = 0.0;
  let saa = 0.0;
  let sbb = 0.0;
  for (let i = 0; i < n; i += 1) {
    const da = a[i] - ma;
    const db = b[i] - mb;
    sab += da * db;
    saa += da * da;
    sbb += db * db;
  }
  const den = Math.sqrt(saa * sbb);
  return den > 1e-12 ? sab / den : 0.0;
}

function quantileLinear(vals, qs) {
  const sorted = Float64Array.from(vals).sort();
  const n = sorted.length;
  const out = new Float64Array(qs.length);
  for (let i = 0; i < qs.length; i += 1) {
    const idx = qs[i] * (n - 1);
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, n - 1);
    out[i] = sorted[lo] + (idx - lo) * (sorted[hi] - sorted[lo]);
  }
  return out;
}

function splitChunks(arr, k) {
  const n = arr.length;
  const base = Math.floor(n / k);
  const rem = n % k;
  const out = [];
  let pos = 0;
  for (let i = 0; i < k; i += 1) {
    const sz = base + (i < rem ? 1 : 0);
    out.push(arr.subarray(pos, pos + sz));
    pos += sz;
  }
  return out;
}

function logsumexp1d(v) {
  let m = -Infinity;
  for (let i = 0; i < v.length; i += 1) {
    if (v[i] > m) m = v[i];
  }
  if (!Number.isFinite(m)) return m;
  let s = 0.0;
  for (let i = 0; i < v.length; i += 1) s += Math.exp(v[i] - m);
  return m + Math.log(s);
}

function logaddexp(a, b) {
  if (a === -Infinity) return b;
  if (b === -Infinity) return a;
  const mx = a > b ? a : b;
  return mx + Math.log(Math.exp(a - mx) + Math.exp(b - mx));
}

function logaddexp0(val) {
  return val > 0.0 ? val + Math.log1p(Math.exp(-val)) : Math.log1p(Math.exp(val));
}

function medianSortedCopy(arr) {
  const s = Float64Array.from(arr).sort();
  const n = s.length;
  const mid = n >>> 1;
  return (n & 1) === 1 ? s[mid] : 0.5 * (s[mid - 1] + s[mid]);
}

// ============================================================================
// 4. Exact Feature Extractors (1-to-1 Mirror of v6_portable.py)
// ============================================================================

function distinctOrder(arr) {
  const seen = new Uint8Array(DIMENSION);
  const tmp = [];
  for (let i = 0; i < arr.length; i += 1) {
    const z = arr[i] - 1;
    if (z >= 0 && z < DIMENSION && !seen[z]) {
      seen[z] = 1;
      tmp.push(z);
    }
  }
  return Int32Array.from(tmp);
}

function precomputeCdfMarg99(arr) {
  const L = arr.length;
  const out = new Float64Array(99);
  let outPos = 0;
  const taus = [16.0, 64.0, 1e9];

  // marg_mat [L, 29]
  for (let ti = 0; ti < 3; ti += 1) {
    const tau = taus[ti];
    const w = new Float64Array(L);
    let ws = 0.0;
    for (let i = 0; i < L; i += 1) {
      const wi = Math.exp(-i / tau);
      w[i] = wi;
      ws += wi;
    }
    for (let i = 0; i < L; i += 1) w[i] /= ws;
    for (let c = 0; c < 29; c += 1) {
      let acc = 0.0;
      for (let i = 0; i < L; i += 1) {
        acc += w[i] * STATIC.margBasis[(arr[i] - 1) * 29 + c];
      }
      out[outPos++] = acc;
    }
  }

  // jumps_mat [L-1, 4]
  const Lj = L - 1;
  for (let ti = 0; ti < 3; ti += 1) {
    const tau = taus[ti];
    const w = new Float64Array(Lj);
    let ws = 0.0;
    for (let i = 0; i < Lj; i += 1) {
      const wi = Math.exp(-i / tau);
      w[i] = wi;
      ws += wi;
    }
    for (let i = 0; i < Lj; i += 1) w[i] /= ws;
    for (let pwr = 1; pwr <= 4; pwr += 1) {
      let acc = 0.0;
      for (let i = 0; i < Lj; i += 1) {
        const dj = Math.abs(arr[i + 1] - arr[i]) / 355.0;
        acc += w[i] * (dj ** pwr);
      }
      out[outPos++] = acc;
    }
  }
  return out;
}

function fastExtractCdf225(order, marg99, hazard) {
  const n = order.length;
  const out = new Float64Array(225);

  let hSum = 0.0;
  const wbSum = new Float64Array(32);
  const wb2Sum = new Float64Array(32);
  for (let z = 0; z < 355; z += 1) {
    const hz = hazard[z];
    hSum += hz;
    const bo = z * 32;
    for (let c = 0; c < 32; c += 1) {
      const b = STATIC.resBasis[bo + c];
      wbSum[c] += hz * b;
      wb2Sum[c] += hz * b * b;
    }
  }

  const total = new Float64Array(n);
  let cumChosen = 0.0;
  for (let t = 0; t < n; t += 1) {
    total[t] = Math.max(hSum - cumChosen, 1e-9);
    cumChosen += hazard[order[t]];
  }

  const residual = new Float64Array(n * 32);
  const variance = new Float64Array(n * 32);
  const already = new Float64Array(32);
  const already2 = new Float64Array(32);
  for (let t = 0; t < n; t += 1) {
    const z = order[t];
    const hz = hazard[z];
    const tot = total[t];
    const bo = z * 32;
    const ro = t * 32;
    for (let c = 0; c < 32; c += 1) {
      const b = STATIC.resBasis[bo + c];
      const ex = (wbSum[c] - already[c]) / tot;
      const vr = Math.max((wb2Sum[c] - already2[c]) / tot - ex * ex, 0.0);
      residual[ro + c] = b - ex;
      variance[ro + c] = vr;
      already[c] += hz * b;
      already2[c] += hz * b * b;
    }
  }

  let outOff = 0;
  for (let si = 0; si < 3; si += 1) {
    const tau = RES_SCALES[si];
    const w = new Float64Array(n);
    const w2 = new Float64Array(n);
    for (let t = 0; t < n; t += 1) {
      const wt = Math.exp(-t / tau);
      w[t] = wt;
      w2[t] = wt * wt;
    }
    for (let c = 0; c < 32; c += 1) {
      let num = 0.0;
      let vsum = 0.0;
      for (let t = 0; t < n; t += 1) {
        num += w[t] * residual[t * 32 + c];
        vsum += w2[t] * variance[t * 32 + c];
      }
      out[outOff++] = num / Math.sqrt(vsum + 0.1);
    }
  }

  for (let i = 0; i < 99; i += 1) out[outOff++] = marg99[i];

  const rankPos = new Int32Array(355);
  rankPos.fill(n);
  for (let t = 0; t < n; t += 1) rankPos[order[t]] = t;

  const prob = new Float64Array(n * 355);
  for (let t = 0; t < n; t += 1) {
    const tot = total[t];
    const po = t * 355;
    let pSum = 0.0;
    for (let z = 0; z < 355; z += 1) {
      if (t <= rankPos[z]) {
        const pv = hazard[z] / tot;
        prob[po + z] = pv;
        pSum += pv;
      }
    }
    for (let z = 0; z < 355; z += 1) prob[po + z] /= pSum;
  }

  const hPermArr = Array.from({ length: 355 }, (_, i) => i).sort((a, b) => (hazard[a] - hazard[b]) || (a - b));
  const hPerm = Int32Array.from(hPermArr);
  const invHPerm = new Int32Array(355);
  for (let i = 0; i < 355; i += 1) invHPerm[hPerm[i]] = i;

  // 5 permutations total: 3 static + jump + hazard
  const UMat = new Float64Array(n * 5);
  const SMat = new Float64Array(n * 5);

  for (let pi = 0; pi < 3; pi += 1) {
    const perm = STATIC.cdfPerms[pi];
    const invP = STATIC.invCdfPerms[pi];
    for (let t = 0; t < n; t += 1) {
      const po = t * 355;
      const targetIdx = invP[order[t]];
      let cs = 0.0;
      for (let j = 0; j <= targetIdx; j += 1) cs += prob[po + perm[j]];
      UMat[t * 5 + pi] = cs;
      SMat[t * 5 + pi] = prob[po + perm[targetIdx]];
    }
  }

  for (let t = 0; t < n; t += 1) {
    const prev = t === 0 ? 177 : order[t - 1];
    const joOff = prev * 355;
    const targetIdx = STATIC.invJumpOrder[joOff + order[t]];
    const po = t * 355;
    let cs = 0.0;
    for (let j = 0; j <= targetIdx; j += 1) cs += prob[po + STATIC.jumpOrder[joOff + j]];
    UMat[t * 5 + 3] = cs;
    SMat[t * 5 + 3] = prob[po + STATIC.jumpOrder[joOff + targetIdx]];
  }

  for (let t = 0; t < n; t += 1) {
    const po = t * 355;
    const targetIdx = invHPerm[order[t]];
    let cs = 0.0;
    for (let j = 0; j <= targetIdx; j += 1) cs += prob[po + hPerm[j]];
    UMat[t * 5 + 4] = cs;
    SMat[t * 5 + 4] = prob[po + hPerm[targetIdx]];
  }

  const m5 = n * 5;
  const hi = new Float64Array(m5);
  const lo = new Float64Array(m5);
  for (let i = 0; i < m5; i += 1) {
    hi[i] = 2.0 * UMat[i] - 1.0;
    lo[i] = 2.0 * (UMat[i] - SMat[i]) - 1.0;
  }
  const PHi = [new Float64Array(m5).fill(1.0), hi];
  const PLo = [new Float64Array(m5).fill(1.0), lo];
  for (let k = 1; k < 7; k += 1) {
    const nextHi = new Float64Array(m5);
    const nextLo = new Float64Array(m5);
    const c1 = 2.0 * k + 1.0;
    const invKp1 = 1.0 / (k + 1.0);
    for (let i = 0; i < m5; i += 1) {
      nextHi[i] = (c1 * hi[i] * PHi[k][i] - k * PHi[k - 1][i]) * invKp1;
      nextLo[i] = (c1 * lo[i] * PLo[k][i] - k * PLo[k - 1][i]) * invKp1;
    }
    PHi.push(nextHi);
    PLo.push(nextLo);
  }

  const w16 = new Float64Array(n);
  let w16SqSum = 0.0;
  for (let t = 0; t < n; t += 1) {
    const wt = Math.exp(-t / 16.0);
    w16[t] = wt;
    w16SqSum += wt * wt;
  }

  // pit_cols: shape [5, 6] raveled row-major (perm index outer, deg inner!)
  // Wait: check v6_portable.py line 432:
  // pit_cols has 6 elements, each shape (5,); np.stack(pit_cols, axis=1) has shape (5, 6), .ravel() -> perm outer, deg inner!
  const pitMat = new Float64Array(5 * 6);
  for (let deg = 1; deg <= 6; deg += 1) {
    const denomD = 2.0 * (2.0 * deg + 1.0);
    const normD = Math.sqrt(w16SqSum / (2.0 * deg + 1.0) + 0.1);
    const phiP = PHi[deg + 1];
    const phiM = PHi[deg - 1];
    const ploP = PLo[deg + 1];
    const ploM = PLo[deg - 1];
    for (let pi = 0; pi < 5; pi += 1) {
      let acc = 0.0;
      for (let t = 0; t < n; t += 1) {
        const idx = t * 5 + pi;
        const invS = 1.0 / Math.max(SMat[idx], 1e-12);
        const coordVal = ((phiP[idx] - phiM[idx]) - (ploP[idx] - ploM[idx])) / denomD * invS;
        acc += w16[t] * coordVal;
      }
      pitMat[pi * 6 + (deg - 1)] = acc / normD;
    }
  }
  for (let i = 0; i < 30; i += 1) out[outOff++] = pitMat[i];
  return out;
}

function fastFullResidual(order, hazard) {
  const n = order.length;
  let hSum = 0.0;
  for (let z = 0; z < 355; z += 1) hSum += hazard[z];
  const total = new Float64Array(n);
  let cum = 0.0;
  for (let t = 0; t < n; t += 1) {
    total[t] = Math.max(hSum - cum, 1e-12);
    cum += hazard[order[t]];
  }
  const mSteps = new Int32Array(355);
  mSteps.fill(n);
  for (let t = 0; t < n; t += 1) mSteps[order[t]] = t + 1;

  const out = new Float64Array(355 * 3);
  for (let si = 0; si < 3; si += 1) {
    const tau = RES_SCALES[si];
    const w = new Float64Array(n);
    const A = new Float64Array(n + 1);
    const B = new Float64Array(n + 1);
    const C = new Float64Array(n + 1);
    for (let t = 0; t < n; t += 1) {
      const wt = Math.exp(-t / tau);
      const w2 = wt * wt;
      const tot = total[t];
      w[t] = wt;
      A[t + 1] = A[t] + wt / tot;
      B[t + 1] = B[t] + w2 / tot;
      C[t + 1] = C[t] + w2 / (tot * tot);
    }
    const num = new Float64Array(355);
    for (let z = 0; z < 355; z += 1) {
      num[z] = -hazard[z] * A[mSteps[z]];
    }
    for (let t = 0; t < n; t += 1) num[order[t]] += w[t];
    const off = si * 355;
    for (let z = 0; z < 355; z += 1) {
      const hz = hazard[z];
      const ms = mSteps[z];
      const vr = Math.max(hz * B[ms] - hz * hz * C[ms], 0.0);
      out[off + z] = num[z] / Math.sqrt(vr + 0.1);
    }
  }
  return out;
}

function extractSiblingOpen(arr, hazard) {
  const Lp = Math.min(arr.length, 360);
  const sub = arr.subarray(0, Lp);
  const z0 = new Int32Array(Lp);
  const b64 = new Int32Array(Lp);
  const dec = new Int32Array(Lp);
  const m10 = new Int32Array(Lp);
  for (let t = 0; t < Lp; t += 1) {
    const v = sub[t];
    const z = v - 1;
    z0[t] = z;
    b64[t] = Math.floor((z * 64) / DIMENSION);
    dec[t] = Math.floor(v / 10);
    m10[t] = v % 10;
  }

  const out = new Float64Array(2105);
  let pos = 0;
  for (const tau of [3.0, 10.0, 35.0]) {
    const w = new Float64Array(Lp);
    let wSum = 0.0;
    for (let t = 0; t < Lp; t += 1) {
      const wt = Math.exp(-t / tau);
      w[t] = wt;
      wSum += wt;
    }
    const cTok = new Float64Array(DIMENSION);
    const cB64 = new Float64Array(64);
    const cDec = new Float64Array(36);
    const cM10 = new Float64Array(10);
    for (let t = 0; t < Lp; t += 1) {
      const wn = w[t] / wSum;
      cTok[z0[t]] += wn;
      cB64[b64[t]] += wn;
      cDec[dec[t]] += wn;
      cM10[m10[t]] += wn;
    }
    for (let i = 0; i < DIMENSION; i += 1) out[pos++] = Math.sqrt(cTok[i] + 1e-4);
    for (let i = 0; i < 64; i += 1) out[pos++] = Math.sqrt(cB64[i] + 1e-3);
    for (let i = 0; i < 36; i += 1) out[pos++] = Math.sqrt(cDec[i] + 1e-3);
    for (let i = 0; i < 10; i += 1) out[pos++] = Math.sqrt(cM10[i] + 2e-3);
  }

  const order = distinctOrder(sub);
  const n = order.length;
  let hSum = 0.0;
  for (let z = 0; z < DIMENSION; z += 1) hSum += hazard[z];
  const total = new Float64Array(n);
  let cum = 0.0;
  for (let t = 0; t < n; t += 1) {
    total[t] = Math.max(hSum - cum, 1e-12);
    cum += hazard[order[t]];
  }
  const mSteps = new Int32Array(DIMENSION);
  mSteps.fill(n);
  for (let t = 0; t < n; t += 1) mSteps[order[t]] = t + 1;

  for (const tau of [4.5, 10.0]) {
    const w = new Float64Array(n);
    const A = new Float64Array(n + 1);
    const B = new Float64Array(n + 1);
    const C = new Float64Array(n + 1);
    for (let t = 0; t < n; t += 1) {
      const wt = Math.exp(-t / tau);
      const w2 = wt * wt;
      const tot = total[t];
      w[t] = wt;
      A[t + 1] = A[t] + wt / tot;
      B[t + 1] = B[t] + w2 / tot;
      C[t + 1] = C[t] + w2 / (tot * tot);
    }
    const num = new Float64Array(DIMENSION);
    for (let z = 0; z < DIMENSION; z += 1) num[z] = -hazard[z] * A[mSteps[z]];
    for (let t = 0; t < n; t += 1) num[order[t]] += w[t];
    for (let z = 0; z < DIMENSION; z += 1) {
      const hz = hazard[z];
      const ms = mSteps[z];
      const vr = Math.max(hz * B[ms] - hz * hz * C[ms], 0.0);
      out[pos++] = num[z] / Math.sqrt(vr + 0.08);
    }
  }
  return out;
}

function extractWindowedCdfLexOrders(arr, orderFull, marg99Full, hazard) {
  if (arr.length <= 460) {
    return {
      cdf: fastExtractCdf225(orderFull, marg99Full, hazard),
      lex: fastFullResidual(orderFull, hazard),
      orders: [orderFull],
    };
  }
  const nWin = Math.max(2, Math.floor(arr.length / 325.0 + 0.5));
  const chunks = splitChunks(arr, nWin);
  const cdfs = [];
  const lexs = [];
  const orders = [];
  for (let i = 0; i < chunks.length; i += 1) {
    const ch = chunks[i];
    if (ch.length >= 50) {
      const ordC = distinctOrder(ch);
      const m99C = precomputeCdfMarg99(ch);
      cdfs.push(fastExtractCdf225(ordC, m99C, hazard));
      lexs.push(fastFullResidual(ordC, hazard));
      orders.push(ordC);
    }
  }
  if (!cdfs.length) {
    return {
      cdf: fastExtractCdf225(orderFull, marg99Full, hazard),
      lex: fastFullResidual(orderFull, hazard),
      orders: [orderFull],
    };
  }
  const cdfAvg = new Float64Array(225);
  const lexAvg = new Float64Array(1065);
  const invK = 1.0 / cdfs.length;
  for (let k = 0; k < cdfs.length; k += 1) {
    for (let j = 0; j < 225; j += 1) cdfAvg[j] += cdfs[k][j] * invK;
    for (let j = 0; j < 1065; j += 1) lexAvg[j] += lexs[k][j] * invK;
  }
  return { cdf: cdfAvg, lex: lexAvg, orders };
}

function periodogram16(zcScaled) {
  const L = zcScaled.length;
  const pwr = new Float64Array(16);
  let pSum = 0.0;
  for (let m = 0; m < 16; m += 1) {
    const freq = (2.0 * Math.PI * (m + 1)) / 34.0;
    let re = 0.0;
    let im = 0.0;
    for (let t = 0; t < L; t += 1) {
      const ang = freq * t;
      re += zcScaled[t] * Math.cos(ang);
      im += zcScaled[t] * Math.sin(ang);
    }
    const pm = re * re + im * im;
    pwr[m] = pm;
    pSum += pm;
  }
  const den = pSum + 1e-9;
  for (let m = 0; m < 16; m += 1) pwr[m] /= den;
  return pwr;
}

function buildUniqueSortedCounts(rawIdx) {
  if (rawIdx.length === 0) return { idx: new Int32Array(0), cnt: new Float64Array(0) };
  const sorted = Int32Array.from(rawIdx).sort();
  const idxOut = [];
  const cntOut = [];
  let cur = sorted[0];
  let count = 1;
  for (let i = 1; i < sorted.length; i += 1) {
    if (sorted[i] === cur) count += 1;
    else {
      idxOut.push(cur);
      cntOut.push(count);
      cur = sorted[i];
      count = 1;
    }
  }
  idxOut.push(cur);
  cntOut.push(count);
  return { idx: Int32Array.from(idxOut), cnt: Float64Array.from(cntOut) };
}

function hellingerCounts(counts, alpha = 0.25) {
  const n = counts.length;
  const out = new Float64Array(n);
  let cSum = 0.0;
  for (let i = 0; i < n; i += 1) {
    const ci = counts[i] + alpha;
    out[i] = ci;
    cSum += ci;
  }
  const invSum = 1.0 / cSum;
  for (let i = 0; i < n; i += 1) out[i] = Math.sqrt(out[i] * invSum);
  return out;
}

function indepRes2(a, b, na, nb) {
  const n = a.length;
  const out = new Float64Array(na * nb);
  if (n < 1) return out;
  const invN = 1.0 / n;
  const ca = new Float64Array(na);
  const cb = new Float64Array(nb);
  const obs = new Float64Array(na * nb);
  for (let t = 0; t < n; t += 1) {
    const ai = a[t];
    const bi = b[t];
    ca[ai] += 1.0;
    cb[bi] += 1.0;
    obs[ai * nb + bi] += 1.0;
  }
  const invDenA = 1.0 / (n + 0.5 * na);
  const invDenB = 1.0 / (n + 0.5 * nb);
  const pa = new Float64Array(na);
  const pb = new Float64Array(nb);
  for (let i = 0; i < na; i += 1) pa[i] = (ca[i] + 0.5) * invDenA;
  for (let j = 0; j < nb; j += 1) pb[j] = (cb[j] + 0.5) * invDenB;
  for (let i = 0; i < na; i += 1) {
    const pai = pa[i];
    const rowOff = i * nb;
    for (let j = 0; j < nb; j += 1) {
      const idx = rowOff + j;
      const ob = obs[idx] * invN;
      const ex = pai * pb[j];
      out[idx] = (ob - ex) / Math.sqrt(ex * invN + 1e-3);
    }
  }
  return out;
}

export function extractTabooFast(arr) {
  const L = arr.length;
  const z0 = new Int32Array(L);
  const b32 = new Int32Array(L);
  const b16 = new Int32Array(L);
  const b8 = new Int32Array(L);
  const un = new Int32Array(L);
  const te = new Int32Array(L);
  const hu = new Int32Array(L);
  const dl = new Int32Array(L);
  const m2 = new Int32Array(L);
  const m5 = new Int32Array(L);
  const m8 = new Int32Array(L);

  const cnt355 = new Float64Array(DIMENSION);
  const cB32 = new Float64Array(32);
  const cB16 = new Float64Array(16);
  const cB8 = new Float64Array(8);
  const cUn = new Float64Array(10);
  const cTe = new Float64Array(10);
  const cHu = new Float64Array(4);
  const cDl = new Float64Array(3);
  const cM2 = new Float64Array(2);
  const cM5 = new Float64Array(5);
  const cM8 = new Float64Array(8);

  for (let t = 0; t < L; t += 1) {
    const v = arr[t];
    const z = v - 1;
    z0[t] = z;
    cnt355[z] += 1.0;
    const zb32 = Math.floor((z * 32) / DIMENSION); b32[t] = zb32; cB32[zb32] += 1.0;
    const zb16 = Math.floor((z * 16) / DIMENSION); b16[t] = zb16; cB16[zb16] += 1.0;
    const zb8 = Math.floor((z * 8) / DIMENSION); b8[t] = zb8; cB8[zb8] += 1.0;
    const zu = STATIC.unit[z]; un[t] = zu; cUn[zu] += 1.0;
    const zt = STATIC.ten[z]; te[t] = zt; cTe[zt] += 1.0;
    const zh = STATIC.hund[z]; hu[t] = zh; cHu[zh] += 1.0;
    const zd = STATIC.dlen[z]; dl[t] = zd; cDl[zd] += 1.0;
    const vm2 = v & 1; m2[t] = vm2; cM2[vm2] += 1.0;
    const vm5 = v % 5; m5[t] = vm5; cM5[vm5] += 1.0;
    const vm8 = v & 7; m8[t] = vm8; cM8[vm8] += 1.0;
  }

  const cs355 = new Float64Array(DIMENSION + 1);
  for (let z = 0; z < DIMENSION; z += 1) cs355[z + 1] = cs355[z] + cnt355[z];
  const denomPairs = Math.max(L * (L - 1), 1.0);

  function pairRate(cntArr) {
    let s = 0.0;
    for (let i = 0; i < cntArr.length; i += 1) s += cntArr[i] * (cntArr[i] - 1.0);
    return s / denomPairs;
  }

  const e0Disc = Float64Array.of(
    pairRate(cnt355), pairRate(cB32), pairRate(cB16), pairRate(cB8),
    pairRate(cUn), pairRate(cTe), pairRate(cHu), pairRate(cDl),
    pairRate(cM2), pairRate(cM5), pairRate(cM8),
  );
  const vbDisc = new Float64Array(11);
  for (let d = 0; d < 11; d += 1) vbDisc[d] = Math.max(e0Disc[d] * (1.0 - e0Disc[d]), 1e-4);

  const e0Prox = new Float64Array(5);
  const vbProx = new Float64Array(5);
  for (let ri = 0; ri < 5; ri += 1) {
    const lo = STATIC.proxLo[ri];
    const hi = STATIC.proxHi[ri];
    let dot = 0.0;
    for (let z = 0; z < DIMENSION; z += 1) dot += cnt355[z] * (cs355[hi[z]] - cs355[lo[z]]);
    e0Prox[ri] = Math.max((dot - L) / denomPairs, 0.0);
    vbProx[ri] = Math.max(e0Prox[ri] * (1.0 - e0Prox[ri]), 1e-4);
  }

  const maxK = Math.min(36, L - 1);
  const discOut = new Float64Array(11 * 18);
  const proxOut = new Float64Array(5 * 18);
  const winOut = new Float64Array(35);

  if (maxK >= 1) {
    const cntDAll = new Float64Array(11 * maxK);
    const cntPAll = new Float64Array(5 * maxK);
    const chans = [z0, b32, b16, b8, un, te, hu, dl, m2, m5, m8];
    for (let k = 1; k <= maxK; k += 1) {
      const lk = k - 1;
      for (let d = 0; d < 11; d += 1) {
        const ch = chans[d];
        let hits = 0;
        for (let t = k; t < L; t += 1) {
          if (ch[t] === ch[t - k]) hits += 1;
        }
        cntDAll[d * maxK + lk] = hits;
      }
      let p0 = 0; let p1 = 0; let p2 = 0; let p3 = 0; let p4 = 0;
      for (let t = k; t < L; t += 1) {
        const ad = Math.abs(arr[t] - arr[t - k]);
        if (ad <= 25) {
          p4 += 1;
          if (ad <= 12) {
            p3 += 1;
            if (ad <= 5) {
              p2 += 1;
              if (ad <= 2) {
                p1 += 1;
                if (ad <= 1) p0 += 1;
              }
            }
          }
        }
      }
      cntPAll[0 * maxK + lk] = p0;
      cntPAll[1 * maxK + lk] = p1;
      cntPAll[2 * maxK + lk] = p2;
      cntPAll[3 * maxK + lk] = p3;
      cntPAll[4 * maxK + lk] = p4;
    }

    const k16 = Math.min(16, maxK);
    for (let lk = 0; lk < k16; lk += 1) {
      const invNk = 1.0 / (L - (lk + 1));
      for (let d = 0; d < 11; d += 1) {
        discOut[d * 18 + lk] = (cntDAll[d * maxK + lk] * invNk - e0Disc[d]) / Math.sqrt(vbDisc[d] * invNk + 2e-4);
      }
      for (let ri = 0; ri < 5; ri += 1) {
        proxOut[ri * 18 + lk] = (cntPAll[ri * maxK + lk] * invNk - e0Prox[ri]) / Math.sqrt(vbProx[ri] * invNk + 2e-4);
      }
    }

    if (maxK >= 17) {
      const kHi1 = Math.min(24, maxK);
      let sumN1 = 0.0;
      for (let k = 17; k <= kHi1; k += 1) sumN1 += (L - k);
      const invB1 = 1.0 / sumN1;
      for (let d = 0; d < 11; d += 1) {
        let s = 0.0;
        for (let lk = 16; lk < kHi1; lk += 1) s += cntDAll[d * maxK + lk];
        discOut[d * 18 + 16] = (s * invB1 - e0Disc[d]) / Math.sqrt(vbDisc[d] * invB1 + 2e-4);
      }
      for (let ri = 0; ri < 5; ri += 1) {
        let s = 0.0;
        for (let lk = 16; lk < kHi1; lk += 1) s += cntPAll[ri * maxK + lk];
        proxOut[ri * 18 + 16] = (s * invB1 - e0Prox[ri]) / Math.sqrt(vbProx[ri] * invB1 + 2e-4);
      }
    }

    if (maxK >= 25) {
      let sumN2 = 0.0;
      for (let k = 25; k <= maxK; k += 1) sumN2 += (L - k);
      const invB2 = 1.0 / sumN2;
      for (let d = 0; d < 11; d += 1) {
        let s = 0.0;
        for (let lk = 24; lk < maxK; lk += 1) s += cntDAll[d * maxK + lk];
        discOut[d * 18 + 17] = (s * invB2 - e0Disc[d]) / Math.sqrt(vbDisc[d] * invB2 + 2e-4);
      }
      for (let ri = 0; ri < 5; ri += 1) {
        let s = 0.0;
        for (let lk = 24; lk < maxK; lk += 1) s += cntPAll[ri * maxK + lk];
        proxOut[ri * 18 + 17] = (s * invB2 - e0Prox[ri]) / Math.sqrt(vbProx[ri] * invB2 + 2e-4);
      }
    }

    const maxW = Math.min(20, maxK);
    if (maxW >= 3) {
      const minHitMat = new Uint8Array(L * 7);
      minHitMat.fill(99);
      for (let t = 1; t < L; t += 1) {
        const limK = Math.min(maxW, t);
        const ro = t * 7;
        for (let k = 1; k <= limK; k += 1) {
          const prev = t - k;
          const ad = Math.abs(arr[t] - arr[prev]);
          if (minHitMat[ro + 0] > k && z0[t] === z0[prev]) minHitMat[ro + 0] = k;
          if (minHitMat[ro + 1] > k && ad <= 2) minHitMat[ro + 1] = k;
          if (minHitMat[ro + 2] > k && ad <= 5) minHitMat[ro + 2] = k;
          if (minHitMat[ro + 3] > k && ad <= 12) minHitMat[ro + 3] = k;
          if (minHitMat[ro + 4] > k && un[t] === un[prev]) minHitMat[ro + 4] = k;
          if (minHitMat[ro + 5] > k && te[t] === te[prev]) minHitMat[ro + 5] = k;
          if (minHitMat[ro + 6] > k && b16[t] === b16[prev]) minHitMat[ro + 6] = k;
        }
      }
      const baseEs = [e0Disc[0], e0Prox[1], e0Prox[2], e0Prox[3], e0Disc[4], e0Disc[5], e0Disc[2]];
      const oneMinusE = baseEs.map((e) => 1.0 - Math.min(0.999, Math.max(0.0, e)));
      let posW = 0;
      for (const W of [3, 5, 8, 12, 20]) {
        if (L > W) {
          const nw = L - W;
          for (let ch = 0; ch < 7; ch += 1) {
            let hits = 0;
            for (let t = W; t < L; t += 1) {
              if (minHitMat[t * 7 + ch] <= W) hits += 1;
            }
            const obsW = hits / nw;
            const ew = 1.0 - (oneMinusE[ch] ** W);
            const vw = Math.max(ew * (1.0 - ew), 1e-4);
            winOut[posW + ch] = (obsW - ew) / Math.sqrt(vw / nw + 5e-4);
          }
        }
        posW += 7;
      }
    }
  }

  return concatF64(discOut, proxOut, winOut);
}

export function extractSeqGrammar(arr) {
  const L = arr.length;
  const z0 = new Int32Array(L);
  const b16 = new Int32Array(L);
  const un = new Int32Array(L);
  const te = new Int32Array(L);
  const hu = new Int32Array(L);
  const dl = new Int32Array(L);
  const ds = new Int32Array(L);
  const m5 = new Int32Array(L);
  const m2I = new Int32Array(L);
  const cDl = new Float64Array(3);
  const cHu = new Float64Array(4);
  const cDs = new Float64Array(20);

  let nGe10 = 0;
  for (let t = 0; t < L; t += 1) {
    const v = arr[t];
    const z = v - 1;
    z0[t] = z;
    b16[t] = Math.floor((z * 16) / DIMENSION);
    un[t] = STATIC.unit[z];
    te[t] = STATIC.ten[z];
    hu[t] = STATIC.hund[z];
    dl[t] = STATIC.dlen[z];
    ds[t] = STATIC.dsum[z] - 1;
    m5[t] = v % 5;
    m2I[t] = v & 1;
    cDl[dl[t]] += 1.0;
    cHu[hu[t]] += 1.0;
    cDs[ds[t]] += 1.0;
    if (v >= 10) nGe10 += 1;
  }

  const parts = [];
  parts.push(L > 1 ? indepRes2(b16.subarray(0, L - 1), b16.subarray(1), 16, 16) : new Float64Array(256));
  parts.push(L > 2 ? indepRes2(b16.subarray(0, L - 2), b16.subarray(2), 16, 16) : new Float64Array(256));
  parts.push(L > 1 ? indepRes2(un.subarray(0, L - 1), un.subarray(1), 10, 10) : new Float64Array(100));
  parts.push(L > 2 ? indepRes2(un.subarray(0, L - 2), un.subarray(2), 10, 10) : new Float64Array(100));

  const cntDl3 = new Float64Array(27);
  const cntHu3 = new Float64Array(64);
  if (L >= 3) {
    parts.push(indepRes2(dl.subarray(0, L - 1), dl.subarray(1), 3, 3));
    const invDlDen = 1.0 / (L + 1.5);
    const pDl = Float64Array.from(cDl, (c) => (c + 0.5) * invDlDen);
    for (let t = 0; t < L - 2; t += 1) cntDl3[(dl[t] * 3 + dl[t + 1]) * 3 + dl[t + 2]] += 1.0;
    const invL2 = 1.0 / (L - 2);
    const liftDl3 = new Float64Array(27);
    for (let i = 0; i < 3; i += 1) {
      for (let j = 0; j < 3; j += 1) {
        const pij = pDl[i] * pDl[j];
        for (let k = 0; k < 3; k += 1) {
          const idx = (i * 3 + j) * 3 + k;
          const ex = pij * pDl[k];
          liftDl3[idx] = (cntDl3[idx] * invL2 - ex) / Math.sqrt(ex * invL2 + 1e-3);
        }
      }
    }
    parts.push(liftDl3);

    parts.push(indepRes2(hu.subarray(0, L - 1), hu.subarray(1), 4, 4));
    parts.push(indepRes2(hu.subarray(0, L - 2), hu.subarray(2), 4, 4));
    const invHuDen = 1.0 / (L + 2.0);
    const pHu = Float64Array.from(cHu, (c) => (c + 0.5) * invHuDen);
    for (let t = 0; t < L - 2; t += 1) cntHu3[(hu[t] * 4 + hu[t + 1]) * 4 + hu[t + 2]] += 1.0;
    const liftHu3 = new Float64Array(64);
    for (let i = 0; i < 4; i += 1) {
      for (let j = 0; j < 4; j += 1) {
        const pij = pHu[i] * pHu[j];
        for (let k = 0; k < 4; k += 1) {
          const idx = (i * 4 + j) * 4 + k;
          const ex = pij * pHu[k];
          liftHu3[idx] = (cntHu3[idx] * invL2 - ex) / Math.sqrt(ex * invL2 + 1e-3);
        }
      }
    }
    parts.push(liftHu3);
  } else {
    parts.push(new Float64Array(132));
  }

  const teGe = new Int32Array(nGe10);
  const unGe = new Int32Array(nGe10);
  let pGe = 0;
  for (let t = 0; t < L; t += 1) {
    if (arr[t] >= 10) {
      teGe[pGe] = te[t];
      unGe[pGe] = un[t];
      pGe += 1;
    }
  }
  parts.push(nGe10 >= 5 ? indepRes2(teGe, unGe, 10, 10) : new Float64Array(100));

  if (L >= 3) {
    parts.push(indepRes2(m5.subarray(0, L - 1), m5.subarray(1), 5, 5));
    parts.push(hellingerCounts(cDs, 0.15));
  } else {
    parts.push(new Float64Array(45));
  }

  const a9 = new Float64Array(9 * L);
  for (let t = 0; t < L; t += 1) {
    const z = z0[t];
    const odd = m2I[t];
    const isM5 = m5[t] === 0;
    a9[0 * L + t] = odd;
    a9[1 * L + t] = isM5 ? 1.0 : 0.0;
    a9[2 * L + t] = un[t] === 0 ? 1.0 : 0.0;
    a9[3 * L + t] = (odd === 1 && !isM5) ? 1.0 : 0.0;
    a9[4 * L + t] = STATIC.isPrime[z];
    a9[5 * L + t] = STATIC.isRepdigit[z];
    a9[6 * L + t] = STATIC.isPalin[z];
    a9[7 * L + t] = STATIC.isAsc[z];
    a9[8 * L + t] = STATIC.isDesc[z];
  }
  const lex18 = new Float64Array(18);
  for (let r = 0; r < 9; r += 1) {
    const ro = r * L;
    let s = 0.0;
    for (let t = 0; t < L; t += 1) s += a9[ro + t];
    const mv = L > 0 ? s / L : 0.0;
    let ac = 0.0;
    if (L >= 2) {
      let sp = 0.0;
      for (let t = 0; t < L - 1; t += 1) sp += a9[ro + t] * a9[ro + t + 1];
      const c1 = sp / (L - 1) - mv * mv;
      ac = c1 / Math.sqrt(Math.max(mv * (1.0 - mv), 1e-3) / (L - 1) + 1e-3);
    }
    lex18[2 * r] = mv;
    lex18[2 * r + 1] = ac;
  }

  let cntAllOdd = 0;
  let cntAllEven = 0;
  let cntUnEqTe = 0;
  let cntGe100Dup = 0;
  let cntMod25 = 0;
  let cntMod12 = 0;
  for (let t = 0; t < L; t += 1) {
    const v = arr[t];
    const uo = un[t] & 1;
    const to = te[t] & 1;
    const ho = hu[t] & 1;
    const ao = v >= 100 ? (uo & to & ho) : (v >= 10 ? (uo & to) : uo);
    const ae = v >= 100 ? ((1 - uo) & (1 - to) & (1 - ho)) : (v >= 10 ? ((1 - uo) & (1 - to)) : (1 - uo));
    if (ao) cntAllOdd += 1;
    if (ae) cntAllEven += 1;
    if (un[t] === te[t]) cntUnEqTe += 1;
    if (v >= 100 && (hu[t] === te[t] || te[t] === un[t] || hu[t] === un[t])) cntGe100Dup += 1;
    if (v % 25 === 0) cntMod25 += 1;
    if (v % 12 === 0) cntMod12 += 1;
  }
  const invL = 1.0 / Math.max(L, 1);
  const lexTail = Float64Array.of(
    cntAllOdd * invL,
    cntAllEven * invL,
    L > 0 ? cntUnEqTe * invL : 0.0,
    cntGe100Dup * invL,
    cntMod25 * invL,
    cntMod12 * invL,
  );
  parts.push(lex18, lexTail);

  parts.push(hellingerCounts(cDl, 0.25));
  if (L >= 4) {
    const cDl2 = new Float64Array(9);
    const cDl4 = new Float64Array(81);
    for (let t = 0; t < L - 1; t += 1) cDl2[dl[t] * 3 + dl[t + 1]] += 1.0;
    for (let t = 0; t < L - 3; t += 1) cDl4[((dl[t] * 3 + dl[t + 1]) * 3 + dl[t + 2]) * 3 + dl[t + 3]] += 1.0;
    parts.push(hellingerCounts(cDl2, 0.20), hellingerCounts(cntDl3, 0.12), hellingerCounts(cDl4, 0.06));
  } else {
    parts.push(new Float64Array(117));
  }

  if (L >= 3) {
    const cHu2L1 = new Float64Array(16);
    const cHu2L2 = new Float64Array(16);
    for (let t = 0; t < L - 1; t += 1) cHu2L1[hu[t] * 4 + hu[t + 1]] += 1.0;
    for (let t = 0; t < L - 2; t += 1) cHu2L2[hu[t] * 4 + hu[t + 2]] += 1.0;
    parts.push(hellingerCounts(cHu2L1, 0.15), hellingerCounts(cHu2L2, 0.15), hellingerCounts(cntHu3, 0.08));
  } else {
    parts.push(new Float64Array(96));
  }

  if (nGe10 >= 5) {
    const cTeUn = new Float64Array(100);
    for (let i = 0; i < nGe10; i += 1) cTeUn[teGe[i] * 10 + unGe[i]] += 1.0;
    parts.push(hellingerCounts(cTeUn, 0.08));
  } else {
    parts.push(new Float64Array(100).fill(0.1));
  }

  if (L >= 4) {
    const cM23 = new Float64Array(8);
    const cM24 = new Float64Array(16);
    for (let t = 0; t < L - 2; t += 1) cM23[(m2I[t] * 2 + m2I[t + 1]) * 2 + m2I[t + 2]] += 1.0;
    for (let t = 0; t < L - 3; t += 1) cM24[((m2I[t] * 2 + m2I[t + 1]) * 2 + m2I[t + 2]) * 2 + m2I[t + 3]] += 1.0;
    const m3Arr = Int32Array.from(arr, (v) => v % 3);
    const m4Arr = Int32Array.from(arr, (v) => v & 3);
    const cDsDiff = new Float64Array(24);
    for (let t = 0; t < L - 1; t += 1) {
      const dd = Math.min(23, Math.max(0, ds[t + 1] - ds[t] + 12));
      cDsDiff[dd] += 1.0;
    }
    parts.push(
      hellingerCounts(cM23, 0.20),
      hellingerCounts(cM24, 0.15),
      indepRes2(m3Arr.subarray(0, L - 1), m3Arr.subarray(1), 3, 3),
      indepRes2(m4Arr.subarray(0, L - 1), m4Arr.subarray(1), 4, 4).subarray(0, 15),
      hellingerCounts(cDsDiff, 0.15),
    );
  } else {
    parts.push(new Float64Array(72));
  }

  return concatF64(...parts);
}

export function extractSparseSeqEvents(arr) {
  const L = arr.length;
  if (L < 2) return { idx: new Int32Array(0), val: new Float64Array(0) };
  const z0 = new Int32Array(L);
  const b32 = new Int32Array(L);
  for (let t = 0; t < L; t += 1) {
    const z = arr[t] - 1;
    z0[t] = z;
    b32[t] = Math.floor((z * 32) / DIMENSION);
  }

  const n1 = L - 1;
  if (L < 4) {
    const ev1 = new Int32Array(3 * n1);
    for (let t = 0; t < n1; t += 1) {
      ev1[t] = z0[t] * DIMENSION + z0[t + 1];
      ev1[n1 + t] = 126025 + b32[t] * DIMENSION + z0[t + 1];
      ev1[2 * n1 + t] = 137385 + z0[t] * 32 + b32[t + 1];
    }
    const u = buildUniqueSortedCounts(ev1);
    const val = new Float64Array(u.cnt.length);
    for (let i = 0; i < val.length; i += 1) val[i] = u.cnt[i] / n1;
    return { idx: u.idx, val };
  }

  const b16 = new Int32Array(L);
  const b8 = new Int32Array(L);
  const un = new Int32Array(L);
  const te = new Int32Array(L);
  const dl = new Int32Array(L);
  for (let t = 0; t < L; t += 1) {
    const z = z0[t];
    b16[t] = Math.floor((z * 16) / DIMENSION);
    b8[t] = Math.floor((z * 8) / DIMENSION);
    un[t] = STATIC.unit[z];
    te[t] = STATIC.ten[z];
    dl[t] = STATIC.dlen[z];
  }

  const gL1 = new Int32Array(7 * n1);
  for (let t = 0; t < n1; t += 1) {
    const d1 = arr[t + 1] - arr[t];
    gL1[t] = z0[t] * DIMENSION + z0[t + 1];
    gL1[n1 + t] = 126025 + b32[t] * DIMENSION + z0[t + 1];
    gL1[2 * n1 + t] = 137385 + z0[t] * 32 + b32[t + 1];
    gL1[3 * n1 + t] = 171465 + b32[t] * 32 + b32[t + 1];
    gL1[4 * n1 + t] = 173769 + d1 + 354;
    gL1[5 * n1 + t] = 175187 + b32[t] * 28 + STATIC.stepLut[d1 + 354];
    gL1[6 * n1 + t] = 177595 + (te[t] * 10 + un[t]) * 10 + te[t + 1];
  }

  const n2 = L - 2;
  const gL2 = new Int32Array(7 * n2);
  for (let t = 0; t < n2; t += 1) {
    const d2 = arr[t + 2] - arr[t];
    gL2[t] = 148745 + b32[t] * DIMENSION + z0[t + 2];
    gL2[n2 + t] = 160105 + z0[t] * 32 + b32[t + 2];
    gL2[2 * n2 + t] = 172489 + b32[t] * 32 + b32[t + 2];
    gL2[3 * n2 + t] = 174478 + d2 + 354;
    gL2[4 * n2 + t] = 176083 + (b8[t] * 8 + b8[t + 1]) * 8 + b8[t + 2];
    gL2[5 * n2 + t] = 176595 + (un[t] * 10 + un[t + 1]) * 10 + un[t + 2];
    gL2[6 * n2 + t] = 178595 + (dl[t] * 3 + dl[t + 1]) * 3 + dl[t + 2];
  }

  const n3 = L - 3;
  const gL3 = new Int32Array(n3);
  for (let t = 0; t < n3; t += 1) gL3[t] = 173513 + b16[t] * 16 + b16[t + 3];

  const u1 = buildUniqueSortedCounts(gL1);
  const u2 = buildUniqueSortedCounts(gL2);
  const u3 = buildUniqueSortedCounts(gL3);

  const tot = u1.idx.length + u2.idx.length + u3.idx.length;
  const cIdx = new Int32Array(tot);
  const cVal = new Float64Array(tot);
  let i1 = 0; let i2 = 0; let i3 = 0; let p = 0;
  while (p < tot) {
    const k1 = i1 < u1.idx.length ? u1.idx[i1] : 0x7fffffff;
    const k2 = i2 < u2.idx.length ? u2.idx[i2] : 0x7fffffff;
    const k3 = i3 < u3.idx.length ? u3.idx[i3] : 0x7fffffff;
    if (k1 <= k2 && k1 <= k3) {
      cIdx[p] = k1;
      cVal[p] = u1.cnt[i1] / n1;
      i1 += 1;
    } else if (k2 <= k1 && k2 <= k3) {
      cIdx[p] = k2;
      cVal[p] = u2.cnt[i2] / n2;
      i2 += 1;
    } else {
      cIdx[p] = k3;
      cVal[p] = u3.cnt[i3] / n3;
      i3 += 1;
    }
    p += 1;
  }
  return { idx: cIdx, val: cVal };
}

function extractModelFree(arr) {
  const L = arr.length;
  const z0 = new Int32Array(L);
  const cnt = new Float64Array(DIMENSION);
  for (let i = 0; i < L; i += 1) {
    const z = arr[i] - 1;
    z0[i] = z;
    cnt[z] += 1.0;
  }

  let entTok = 0.0;
  for (let z = 0; z < DIMENSION; z += 1) {
    if (cnt[z] > 0) {
      const p = cnt[z] / L;
      entTok -= p * Math.log(p);
    }
  }
  entTok /= Math.log(DIMENSION);

  const b16 = new Int32Array(L);
  const h16Cnt = new Float64Array(16);
  const m10Cnt = new Float64Array(10);
  const tenCnt = new Float64Array(10);
  const hunCnt = new Float64Array(4);
  for (let i = 0; i < L; i += 1) {
    const v = arr[i];
    const sb = Math.min(Math.floor((z0[i] * 16) / DIMENSION), 15);
    b16[i] = sb;
    h16Cnt[sb] += 1.0;
    m10Cnt[v % 10] += 1.0;
    tenCnt[Math.floor(v / 10) % 10] += 1.0;
    hunCnt[Math.floor(v / 100)] += 1.0;
  }
  const h16 = new Float64Array(16);
  for (let i = 0; i < 16; i += 1) h16[i] = Math.sqrt((h16Cnt[i] + 0.10) / (L + 1.6));
  const hMod10 = new Float64Array(10);
  const hTens = new Float64Array(10);
  for (let i = 0; i < 10; i += 1) {
    hMod10[i] = Math.sqrt((m10Cnt[i] + 0.10) / (L + 1.0));
    hTens[i] = Math.sqrt((tenCnt[i] + 0.10) / (L + 1.0));
  }
  const hHund = new Float64Array(4);
  for (let i = 0; i < 4; i += 1) hHund[i] = Math.sqrt((hunCnt[i] + 0.10) / (L + 0.4));

  const segEdges = [0, Math.floor(L / 4), Math.floor((2 * L) / 4), Math.floor((3 * L) / 4), L];
  const segH16 = [];
  const segMeans = new Float64Array(4);
  const segStds = new Float64Array(4);
  for (let s = 0; s < 4; s += 1) {
    const sub = z0.subarray(segEdges[s], segEdges[s + 1]);
    if (sub.length > 0) {
      const sc = new Float64Array(16);
      for (let i = 0; i < sub.length; i += 1) sc[Math.min(Math.floor((sub[i] * 16) / DIMENSION), 15)] += 1.0;
      const sh = new Float64Array(16);
      for (let b = 0; b < 16; b += 1) sh[b] = Math.sqrt((sc[b] + 0.10) / (sub.length + 1.6));
      segH16.push(sh);
      segMeans[s] = meanArr(sub) / DIMENSION;
      segStds[s] = stdArr(sub) / DIMENSION;
    } else {
      segH16.push(new Float64Array(16).fill(0.25));
      segMeans[s] = 0.5;
      segStds[s] = 0.28;
    }
  }

  const nDiff = Math.max(L - 1, 0);
  const diffs = new Float64Array(nDiff);
  const adiffs = new Float64Array(nDiff);
  const adiffsScaled = new Float64Array(nDiff);
  let negCnt = 0;
  let zeroCnt = 0;
  let posCnt = 0;
  for (let i = 0; i < nDiff; i += 1) {
    const df = arr[i + 1] - arr[i];
    diffs[i] = df;
    const ad = Math.abs(df);
    adiffs[i] = ad;
    adiffsScaled[i] = ad / DIMENSION;
    if (df < 0) negCnt += 1;
    else if (df === 0) zeroCnt += 1;
    else posCnt += 1;
  }
  const nStep = Math.max(nDiff, 1);
  const stepSigns = [negCnt / nStep, zeroCnt / nStep, posCnt / nStep];
  const dQuant = nDiff > 0 ? quantileLinear(adiffsScaled, [0.1, 0.25, 0.5, 0.75, 0.9, 0.98]) : new Float64Array(6);
  const smallSteps = [1, 2, 3, 5, 10, 20, 50].map((k) => {
    if (nDiff === 0) return 0.0;
    let c = 0;
    for (let i = 0; i < nDiff; i += 1) if (adiffs[i] <= k) c += 1;
    return c / nDiff;
  });

  const markov8 = new Float64Array(64);
  if (L >= 2) {
    for (let i = 0; i < L - 1; i += 1) {
      const bu = Math.min(Math.floor((z0[i] * 8) / DIMENSION), 7);
      const bv = Math.min(Math.floor((z0[i + 1] * 8) / DIMENSION), 7);
      markov8[bu * 8 + bv] += 1.0;
    }
  }
  for (let i = 0; i < 64; i += 1) markov8[i] = Math.sqrt((markov8[i] + 0.10) / (nStep + 6.4));

  const mod10Tr = new Float64Array(100);
  if (L >= 2) {
    for (let i = 0; i < L - 1; i += 1) {
      mod10Tr[(arr[i] % 10) * 10 + (arr[i + 1] % 10)] += 1.0;
    }
  }
  for (let i = 0; i < 100; i += 1) mod10Tr[i] = Math.sqrt((mod10Tr[i] + 0.05) / (nStep + 5.0));

  const meanA = meanArr(arr);
  const zc = new Float64Array(L);
  const zcScaled = new Float64Array(L);
  let zcDenom = 1e-9;
  for (let i = 0; i < L; i += 1) {
    const v = arr[i] - meanA;
    zc[i] = v;
    zcScaled[i] = v / DIMENSION;
    zcDenom += v * v;
  }
  const acf = new Float64Array(8);
  for (let lag = 1; lag <= 8; lag += 1) {
    if (L > lag) {
      let s = 0.0;
      for (let i = 0; i < L - lag; i += 1) s += zc[i] * zc[i + lag];
      acf[lag - 1] = s / zcDenom;
    }
  }

  const seen = new Map();
  const repLags = [];
  let firstDupPos = 1.0;
  let foundDup = false;
  for (let i = 0; i < L; i += 1) {
    const val = arr[i];
    if (seen.has(val)) {
      if (!foundDup) {
        firstDupPos = i / L;
        foundDup = true;
      }
      repLags.push(i - seen.get(val));
    }
    seen.set(val, i);
  }
  const nUniq = seen.size;
  const uniqRatio = nUniq / L;
  const expectedUniq = DIMENSION * (1.0 - (((DIMENSION - 1.0) / DIMENSION) ** L));
  const couponExcess = (nUniq - expectedUniq) / Math.max(Math.sqrt(L), 1.0);
  const lagHist = new Float64Array(LAG_BINS_EDGE.length + 1);
  if (repLags.length > 0) {
    for (let i = 0; i < repLags.length; i += 1) {
      const lg = repLags[i];
      let bi = 0;
      for (let e = 0; e < LAG_BINS_EDGE.length; e += 1) {
        if (lg >= LAG_BINS_EDGE[e]) bi += 1;
        else break;
      }
      lagHist[bi] += 1.0;
    }
    for (let b = 0; b < lagHist.length; b += 1) lagHist[b] /= repLags.length;
  }

  const uVals = new Float64Array(L);
  for (let i = 0; i < L; i += 1) uVals[i] = (arr[i] - 0.5) / DIMENSION;
  const uGrid = Float64Array.from({ length: 19 }, (_, i) => 0.05 * (i + 1));
  const empCdfQ = quantileLinear(uVals, uGrid);
  const empCdf = new Float64Array(19);
  for (let i = 0; i < 19; i += 1) empCdf[i] = empCdfQ[i] - uGrid[i];

  const specBins = periodogram16(zcScaled);
  let turnRate = 0.5;
  if (diffs.length >= 2) {
    let tc = 0;
    for (let i = 0; i < diffs.length - 1; i += 1) {
      if (diffs[i] * diffs[i + 1] < 0) tc += 1;
    }
    turnRate = tc / (diffs.length - 1);
  }

  const fhParts = [
    h16, hMod10, hTens, hHund, segH16[0], segH16[1], segMeans, segStds,
    stepSigns, dQuant, smallSteps, markov8, acf,
    [Math.log(L / 200.0), L / 400.0, uniqRatio, couponExcess, entTok, firstDupPos, repLags.length / L, turnRate],
    lagHist, empCdf, specBins,
  ];
  let fhLen = 0;
  for (const p of fhParts) fhLen += p.length;
  const fh = new Float64Array(fhLen);
  let fhOff = 0;
  for (const p of fhParts) {
    for (let i = 0; i < p.length; i += 1) fh[fhOff++] = p[i];
  }

  const order = distinctOrder(arr);
  const m = order.length;
  const fPl = new Float64Array(DIMENSION + 28);
  const pos = new Float64Array(m);
  const orderF = new Float64Array(m);
  for (let i = 0; i < m; i += 1) {
    pos[i] = i;
    orderF[i] = order[i];
    fPl[order[i]] = (1.0 - i / Math.max(m, 1)) - (0.5 * m / DIMENSION);
  }
  let diffAbsSum = 0.0;
  let diffP1 = 0;
  let diffM1 = 0;
  let diffLe5 = 0;
  for (let i = 0; i < m - 1; i += 1) {
    const df = order[i + 1] - order[i];
    const ad = Math.abs(df);
    diffAbsSum += ad;
    if (df === 1) diffP1 += 1;
    if (df === -1) diffM1 += 1;
    if (ad <= 5) diffLe5 += 1;
  }
  const plTail = [
    m / DIMENSION,
    (DIMENSION - m) / DIMENSION,
    Math.log((m + 1.0) / 200.0),
    meanArr(orderF) / DIMENSION - 0.5,
    stdArr(orderF) / DIMENSION,
    orderF[0] / DIMENSION - 0.5,
    orderF[m - 1] / DIMENSION - 0.5,
    meanArr(orderF.subarray(0, Math.min(10, m))) / DIMENSION - 0.5,
    meanArr(orderF.subarray(0, Math.min(30, m))) / DIMENSION - 0.5,
    meanArr(orderF.subarray(Math.max(0, m - 30))) / DIMENSION - 0.5,
    m >= 3 ? corrcoef1d(pos, orderF) : 0.0,
    m >= 2 ? (diffAbsSum / (m - 1)) / DIMENSION : 0.0,
    m >= 2 ? diffP1 / (m - 1) : 0.0,
    m >= 2 ? diffM1 / (m - 1) : 0.0,
    m >= 2 ? diffLe5 / (m - 1) : 0.0,
    STATIC.hCum[m - 1] / 6.5,
  ];
  for (let i = 0; i < 16; i += 1) fPl[DIMENSION + i] = plTail[i];
  for (let q = 0; q < 12; q += 1) {
    const loQ = Math.floor((q * m) / 12.0);
    const hiQ = q < 11 ? Math.floor(((q + 1) * m) / 12.0) : m;
    const subO = orderF.subarray(loQ, hiQ);
    fPl[DIMENSION + 16 + q] = subO.length > 0 ? meanArr(subO) / DIMENSION - 0.5 : 0.0;
  }

  const tr32 = new Float64Array(1024);
  if (L >= 2) {
    for (let i = 0; i < L - 1; i += 1) {
      const bu = Math.min(Math.floor((z0[i] * 32) / DIMENSION), 31);
      const bv = Math.min(Math.floor((z0[i + 1] * 32) / DIMENSION), 31);
      tr32[bu * 32 + bv] += 1.0;
    }
  }
  const tr32Sub = new Float64Array(256);
  for (let i = 0; i < 256; i += 1) tr32Sub[i] = Math.sqrt((tr32[i * 4] + 0.02) / (nStep + 1024 * 0.02));

  const tr16L2 = new Float64Array(256);
  if (L >= 3) {
    for (let i = 0; i < L - 2; i += 1) {
      tr16L2[b16[i] * 16 + b16[i + 2]] += 1.0;
    }
  }
  const tr16Sub = new Float64Array(128);
  const denL2 = Math.max(L - 2, 1) + 256 * 0.05;
  for (let i = 0; i < 128; i += 1) tr16Sub[i] = Math.sqrt((tr16L2[i * 2] + 0.05) / denL2);

  const cntSub = new Float64Array(23);
  for (let i = 0; i < 23; i += 1) cntSub[i] = Math.sqrt((cnt[i * 16] + 0.1) / (L + 35.5));

  const fDyn = new Float64Array(256 + 128 + 100 + 16 + 16 + 23);
  let dOff = 0;
  for (const part of [tr32Sub, tr16Sub, mod10Tr, segH16[2], segH16[3], cntSub]) {
    for (let i = 0; i < part.length; i += 1) fDyn[dOff++] = part[i];
  }

  const fet = extractSparseSeqEvents(arr);
  return { fh, fPl, fDyn, fet };
}

function extractFreqAndReplag(arr) {
  const L = arr.length;
  const cAll = new Float64Array(DIMENSION);
  for (let i = 0; i < L; i += 1) cAll[arr[i] - 1] += 1.0;
  const hAll = new Float64Array(DIMENSION);
  for (let z = 0; z < DIMENSION; z += 1) hAll[z] = Math.sqrt((cAll[z] + 0.10) / (L + 0.10 * DIMENSION));

  const half = Math.floor(L / 2);
  const c1 = new Float64Array(DIMENSION);
  const c2 = new Float64Array(DIMENSION);
  for (let i = 0; i < half; i += 1) c1[arr[i] - 1] += 1.0;
  for (let i = half; i < L; i += 1) c2[arr[i] - 1] += 1.0;
  const h1 = new Float64Array(DIMENSION);
  const h2 = new Float64Array(DIMENSION);
  const d1 = Math.max(half, 1) + 0.05 * DIMENSION;
  const d2 = Math.max(L - half, 1) + 0.05 * DIMENSION;
  let diffSq = 0.0;
  for (let z = 0; z < DIMENSION; z += 1) {
    h1[z] = Math.sqrt((c1[z] + 0.05) / d1);
    h2[z] = Math.sqrt((c2[z] + 0.05) / d2);
    const df = h1[z] - h2[z];
    diffSq += df * df;
  }
  const driftL2 = Math.sqrt(diffSq);
  const driftCos = dotArr(h1, h2) / (normArr(h1) * normArr(h2) + 1e-12);

  const multHist = new Float64Array(7);
  for (let z = 0; z < DIMENSION; z += 1) {
    const k = cAll[z];
    if (k >= 6) multHist[6] += 1.0;
    else multHist[k] += 1.0;
  }
  for (let k = 0; k < 7; k += 1) multHist[k] /= DIMENSION;

  const lastPos = new Map();
  const lags = [];
  let immRep = 0;
  for (let i = 0; i < L; i += 1) {
    const x = arr[i];
    if (lastPos.has(x)) {
      const lg = i - lastPos.get(x);
      lags.push(lg);
      if (lg <= 3) immRep += 1;
    }
    lastPos.set(x, i);
  }
  const repRate = lags.length / Math.max(L, 1);
  let meanLag = 1.0;
  let qLag = new Float64Array([1, 1, 1, 1, 1]);
  if (lags.length > 0) {
    const lagsScaled = Float64Array.from(lags, (lg) => lg / L);
    meanLag = meanArr(lagsScaled);
    qLag = quantileLinear(lagsScaled, [0.1, 0.25, 0.5, 0.75, 0.9]);
  }

  const out = new Float64Array(DIMENSION + 7 + 5 + 5);
  let pos = 0;
  for (let i = 0; i < DIMENSION; i += 1) out[pos++] = hAll[i];
  for (let i = 0; i < 7; i += 1) out[pos++] = multHist[i];
  out[pos++] = driftL2;
  out[pos++] = driftCos;
  out[pos++] = repRate;
  out[pos++] = immRep / Math.max(L, 1);
  out[pos++] = meanLag;
  for (let i = 0; i < 5; i += 1) out[pos++] = qLag[i];
  return out;
}

function extractGeometryFast(arr) {
  const n = arr.length;
  const u = new Float64Array(n);
  const x = new Float64Array(n);
  const cnt = new Float64Array(DIMENSION);
  for (let i = 0; i < n; i += 1) {
    u[i] = (arr[i] - 0.5) / 355.0;
    x[i] = 2.0 * u[i] - 1.0;
    cnt[arr[i] - 1] += 1.0;
  }
  const avgRank = new Float64Array(DIMENSION);
  let cum = 0.0;
  for (let z = 0; z < DIMENSION; z += 1) {
    cum += cnt[z];
    avgRank[z] = cum - cnt[z] + 0.5 * (cnt[z] + 1.0);
  }
  const r = new Float64Array(n);
  const rx = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    r[i] = (avgRank[arr[i] - 1] - 0.5) / n;
    rx[i] = 2.0 * r[i] - 1.0;
  }

  const feats = [];
  const pLeg = legPolyMatrix(x, 16);
  for (let d = 0; d < 16; d += 1) {
    let s = 0.0;
    for (let i = 0; i < n; i += 1) s += pLeg[i * 16 + d];
    feats.push(s / n);
  }
  for (const k of [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]) {
    let sc = 0.0;
    let ss = 0.0;
    const f = 2.0 * Math.PI * k;
    for (let i = 0; i < n; i += 1) {
      const ang = f * u[i];
      sc += Math.cos(ang);
      ss += Math.sin(ang);
    }
    feats.push(sc / n, ss / n);
  }

  const qs = Float64Array.from({ length: 19 }, (_, i) => 0.05 * (i + 1));
  const uq = quantileLinear(u, qs);
  for (let i = 0; i < 19; i += 1) feats.push(uq[i] - qs[i]);
  let cLt05 = 0;
  let cGt95 = 0;
  let cMid = 0;
  for (let i = 0; i < n; i += 1) {
    if (u[i] < 0.05) cLt05 += 1;
    if (u[i] > 0.95) cGt95 += 1;
    if (u[i] > 0.45 && u[i] < 0.55) cMid += 1;
  }
  feats.push(
    meanArr(u),
    stdArr(u),
    uq[14] - uq[4],
    uq[17] - uq[1],
    cLt05 / n,
    cGt95 / n,
    cMid / n,
    Math.log(n / 200.0),
  );

  const meanAf = meanArr(arr);
  const stdAf = stdArr(arr) + 1e-9;
  const meanR = meanArr(r);
  const stdR = stdArr(r) + 1e-9;
  const z = new Float64Array(n);
  const rz = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    z[i] = (arr[i] - meanAf) / stdAf;
    rz[i] = (r[i] - meanR) / stdR;
  }

  for (const lag of [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32]) {
    if (n > lag) {
      const m = n - lag;
      let zz = 0.0;
      let rzz = 0.0;
      let lr = 0.0;
      let l2r = 0.0;
      let lr2 = 0.0;
      let l2r2 = 0.0;
      let ad = 0.0;
      let d2 = 0.0;
      let d3 = 0.0;
      let sgn = 0.0;
      let lt02 = 0;
      let lt10 = 0;
      let rlrr = 0.0;
      for (let i = 0; i < m; i += 1) {
        const j = i + lag;
        const lv = x[i];
        const rv = x[j];
        const dv = u[j] - u[i];
        const adv = Math.abs(dv);
        zz += z[i] * z[j];
        rzz += rz[i] * rz[j];
        lr += lv * rv;
        l2r += lv * lv * rv;
        lr2 += lv * rv * rv;
        l2r2 += lv * lv * rv * rv;
        ad += adv;
        d2 += dv * dv;
        d3 += dv * dv * dv;
        sgn += Math.sign(dv);
        if (adv < 0.02) lt02 += 1;
        if (adv < 0.10) lt10 += 1;
        rlrr += rx[i] * rx[j];
      }
      feats.push(zz / m, rzz / m, lr / m, l2r / m, lr2 / m, l2r2 / m, ad / m, d2 / m, d3 / m, sgn / m, lt02 / m, lt10 / m, rlrr / m);
    } else {
      for (let k = 0; k < 13; k += 1) feats.push(0.0);
    }
  }

  for (const a of [x, rx]) {
    for (const lag of [1, 2]) {
      if (n > lag) {
        const m = n - lag;
        let lr = 0.0;
        let l2r = 0.0;
        let lr2 = 0.0;
        let l2r2 = 0.0;
        let ad = 0.0;
        let d2 = 0.0;
        let d3 = 0.0;
        let sgn = 0.0;
        let lt10 = 0;
        for (let i = 0; i < m; i += 1) {
          const lv = a[i];
          const rv = a[i + lag];
          const dv = rv - lv;
          const adv = Math.abs(dv);
          lr += lv * rv;
          l2r += lv * lv * rv;
          lr2 += lv * rv * rv;
          l2r2 += lv * lv * rv * rv;
          ad += adv;
          d2 += dv * dv;
          d3 += dv * dv * dv;
          sgn += Math.sign(dv);
          if (adv < 0.1) lt10 += 1;
        }
        feats.push(lr / m, l2r / m, lr2 / m, l2r2 / m, ad / m, d2 / m, d3 / m, sgn / m, lt10 / m);
      } else {
        for (let k = 0; k < 9; k += 1) feats.push(0.0);
      }
    }
    for (const blk of splitChunks(a, 4)) {
      if (blk.length >= 2) {
        let ad = 0.0;
        let pos = 0;
        for (let i = 0; i < blk.length - 1; i += 1) {
          const df = blk[i + 1] - blk[i];
          ad += Math.abs(df);
          if (df > 0) pos += 1;
        }
        feats.push(meanArr(blk), stdArr(blk), ad / (blk.length - 1), pos / (blk.length - 1));
      } else {
        feats.push(0.0, 0.0, 0.0, 0.0);
      }
    }
    const bins = new Int32Array(n);
    for (let i = 0; i < n; i += 1) bins[i] = Math.min(3, Math.max(0, Math.floor((a[i] + 1.0) * 2.0)));
    const bCnt = new Float64Array(16);
    for (let i = 0; i < n - 1; i += 1) bCnt[bins[i] * 4 + bins[i + 1]] += 1.0;
    const bDen = Math.max(n - 1, 1);
    for (let b = 0; b < 16; b += 1) feats.push(bCnt[b] / bDen);
  }

  if (n >= 4) {
    const n3 = n - 2;
    const c3Cnt = new Int32Array(27);
    let tie3 = 0;
    for (let t = 0; t < n3; t += 1) {
      const a0 = arr[t];
      const a1 = arr[t + 1];
      const a2 = arr[t + 2];
      if (a0 === a1 || a1 === a2 || a0 === a2) tie3 += 1;
      const vals3 = [a0, a1, a2];
      const idx3 = [0, 1, 2].sort((i, j) => (vals3[i] - vals3[j]) || (i - j));
      c3Cnt[idx3[0] * 9 + idx3[1] * 3 + idx3[2]] += 1;
    }
    for (const p of STATIC.perms3) feats.push(c3Cnt[p[0] * 9 + p[1] * 3 + p[2]] / n3);
    feats.push(tie3 / n3);

    const n4 = n - 3;
    const c4Cnt = new Int32Array(256);
    let tie4 = 0;
    for (let t = 0; t < n4; t += 1) {
      const b0 = arr[t];
      const b1 = arr[t + 1];
      const b2 = arr[t + 2];
      const b3 = arr[t + 3];
      if (b0 === b1 || b0 === b2 || b0 === b3 || b1 === b2 || b1 === b3 || b2 === b3) tie4 += 1;
      const vals4 = [b0, b1, b2, b3];
      const idx4 = [0, 1, 2, 3].sort((i, j) => (vals4[i] - vals4[j]) || (i - j));
      c4Cnt[idx4[0] * 64 + idx4[1] * 16 + idx4[2] * 4 + idx4[3]] += 1;
    }
    for (const p of STATIC.perms4) feats.push(c4Cnt[p[0] * 64 + p[1] * 16 + p[2] * 4 + p[3]] / n4);
    feats.push(tie4 / n4);
  } else {
    for (let k = 0; k < 32; k += 1) feats.push(0.0);
  }

  const digUnit = Int32Array.from(arr, (v) => v % 10);
  const digTen = Int32Array.from(arr, (v) => Math.floor(v / 10) % 10);
  for (const dig of [digUnit, digTen]) {
    for (const lag of [1, 2, 4, 8]) {
      if (n > lag) {
        const m = n - lag;
        const bc = new Float64Array(10);
        for (let i = 0; i < m; i += 1) bc[((dig[i + lag] - dig[i]) % 10 + 10) % 10] += 1.0;
        for (let d = 0; d < 10; d += 1) feats.push(bc[d] / m);
      } else {
        for (let d = 0; d < 10; d += 1) feats.push(0.1);
      }
    }
  }

  if (n >= 3) {
    const m = n - 2;
    let sProd = 0.0;
    let sAdiff = 0.0;
    let sSgnNeq = 0;
    let sAsum = 0.0;
    let sAsub = 0.0;
    for (let i = 0; i < m; i += 1) {
      const dz0 = z[i + 1] - z[i];
      const dz1 = z[i + 2] - z[i + 1];
      sProd += dz0 * dz1;
      sAdiff += Math.abs(dz0) - Math.abs(dz1);
      if (Math.sign(dz0) !== Math.sign(dz1)) sSgnNeq += 1;
      sAsum += Math.abs(dz1 + dz0);
      sAsub += Math.abs(dz1 - dz0);
    }
    feats.push(sProd / m, sAdiff / m, sSgnNeq / m, sAsum / m, sAsub / m);
  } else {
    feats.push(0.0, 0.0, 0.0, 0.0, 0.0);
  }

  return Float64Array.from(feats);
}

function rootHist(vals, sz) {
  const c = new Float64Array(sz);
  c.fill(0.5);
  for (let i = 0; i < vals.length; i += 1) c[vals[i]] += 1.0;
  const tot = vals.length + 0.5 * sz;
  for (let i = 0; i < sz; i += 1) c[i] = Math.sqrt(c[i] / tot);
  return c;
}

function extractMarkov1497(arr) {
  const L = arr.length;
  const z0 = Int32Array.from(arr, (v) => v - 1);
  const bins = Int32Array.from(z0, (z) => Math.floor((z * 16) / 355));
  const digs = Int32Array.from(arr, (v) => v % 10);
  const pieces = [rootHist(z0, 355)];
  for (const ch of splitChunks(bins, 4)) pieces.push(rootHist(ch, 16));
  pieces.push(rootHist(digs, 10));
  for (const lag of [1, 2, 4]) {
    if (L > lag) {
      const m = L - lag;
      const b2 = new Int32Array(m);
      const d2 = new Int32Array(m);
      for (let i = 0; i < m; i += 1) {
        b2[i] = 16 * bins[i] + bins[i + lag];
        d2[i] = 10 * digs[i] + digs[i + lag];
      }
      pieces.push(rootHist(b2, 256), rootHist(d2, 100));
    } else {
      pieces.push(new Float64Array(256).fill(1.0 / 16.0), new Float64Array(100).fill(0.1));
    }
  }
  const out = new Float64Array(1497);
  let off = 0;
  for (const p of pieces) {
    for (let i = 0; i < p.length; i += 1) out[off++] = p[i];
  }
  return out;
}

function extractChunk1065(arr) {
  const z0 = Int32Array.from(arr, (v) => v - 1);
  const chunks = splitChunks(z0, 3);
  const a = rootHist(chunks[0], 355);
  const b = rootHist(chunks[1], 355);
  const c = rootHist(chunks[2], 355);
  const out = new Float64Array(1065);
  const s3 = 1.0 / Math.sqrt(3.0);
  const s2 = 1.0 / Math.sqrt(2.0);
  const s6 = 1.0 / Math.sqrt(6.0);
  for (let i = 0; i < 355; i += 1) {
    out[i] = (a[i] + b[i] + c[i]) * s3;
    out[355 + i] = (a[i] - c[i]) * s2;
    out[710 + i] = (a[i] - 2.0 * b[i] + c[i]) * s6;
  }
  return out;
}

function extractArFisher(arr, beta0, alpha0, mBase) {
  const L = arr.length;
  const z0 = Int32Array.from(arr, (v) => v - 1);

  const occCur = new Int32Array(355);
  const lastSeen = new Int32Array(355);
  lastSeen.fill(-10000);
  const occC = new Int8Array(L * 355);
  const lagB = new Int8Array(L * 355);

  for (let t = 0; t < L; t += 1) {
    const ro = t * 355;
    for (let c = 0; c < 355; c += 1) {
      occC[ro + c] = occCur[c] > 2 ? 2 : occCur[c];
      const lg = Math.min(Math.max(t - lastSeen[c], 0), 51);
      lagB[ro + c] = STATIC.lagLut[lg];
    }
    const zt = z0[t];
    occCur[zt] += 1;
    lastSeen[zt] = t;
  }

  const aHi = alpha0.subarray(11);
  const lagWLut = [0.0, aHi[0], aHi[1], aHi[2], aHi[3], aHi[4], aHi[5]];
  const occWLut = [0.0, aHi[6], aHi[6] + aHi[7]];
  const aAp = aHi[8];

  const P = new Float64Array(L * 355);
  const cPad = new Float64Array(L * 356);
  const surprisal = new Float64Array(L);
  const stepEnt = new Float64Array(L);
  const uMid = new Float64Array(L);
  const xPit = new Float64Array(L);

  const apLo = L >= 3 ? new Int32Array(L - 2) : new Int32Array(0);
  const apHi = L >= 3 ? new Int32Array(L - 2) : new Int32Array(0);
  if (L >= 3) {
    for (let t = 2; t < L; t += 1) {
      const targ = 2 * z0[t - 1] - z0[t - 2];
      apLo[t - 2] = Math.min(355, Math.max(0, targ - 2));
      apHi[t - 2] = Math.min(355, Math.max(0, targ + 3));
    }
  }

  const rowLogits = new Float64Array(355);
  for (let t = 0; t < L; t += 1) {
    const ro = t * 355;
    const prevOff = t > 0 ? z0[t - 1] * 355 : -1;
    for (let c = 0; c < 355; c += 1) {
      rowLogits[c] = lagWLut[lagB[ro + c]] + occWLut[occC[ro + c]] + (t === 0 ? beta0[c] : mBase[prevOff + c]);
    }
    if (t >= 2 && aAp !== 0.0) {
      const targ = 2 * z0[t - 1] - z0[t - 2];
      const loC = Math.max(0, targ - 2);
      const hiC = Math.min(355, targ + 3);
      for (let c = loC; c < hiC; c += 1) rowLogits[c] += aAp;
    }
    const lse = logsumexp1d(rowLogits);
    let ent = 0.0;
    let cs = 0.0;
    const co = t * 356;
    cPad[co] = 0.0;
    for (let c = 0; c < 355; c += 1) {
      const pv = Math.exp(rowLogits[c] - lse);
      P[ro + c] = pv;
      cs += pv;
      cPad[co + c + 1] = cs;
      ent -= pv * Math.log(Math.max(pv, 1e-12));
    }
    const zt = z0[t];
    const pCh = P[ro + zt];
    surprisal[t] = -Math.log(Math.max(pCh, 1e-12));
    stepEnt[t] = ent;
    const um = cPad[co + zt + 1] - 0.5 * pCh;
    uMid[t] = um;
    xPit[t] = 2.0 * um - 1.0;
  }

  function rsum(t, loIdx, hiIdx) {
    const loC = Math.min(355, Math.max(0, loIdx));
    const hiC = Math.min(355, Math.max(0, hiIdx));
    return Math.max(cPad[t * 356 + hiC] - cPad[t * 356 + loC], 0.0);
  }

  const obsD = new Float64Array(L * 20);
  const expD = new Float64Array(L * 20);
  const varD = new Float64Array(L * 20);

  for (let t = 0; t < L; t += 1) {
    const ro = t * 355;
    const doff = t * 20;
    const zt = z0[t];

    if (t >= 1) {
      const u = z0[t - 1];
      const bo = (u * 355 + zt) * 11;
      for (let k = 0; k < 11; k += 1) obsD[doff + k] = STATIC.markovB[bo + k];

      expD[doff + 0] = P[ro + u];
      expD[doff + 1] = rsum(t, u - 1, u) + rsum(t, u + 1, u + 2);
      expD[doff + 2] = rsum(t, u - 2, u - 1) + rsum(t, u + 2, u + 3);
      expD[doff + 3] = rsum(t, u - 5, u - 2) + rsum(t, u + 3, u + 6);
      expD[doff + 4] = rsum(t, u - 12, u - 5) + rsum(t, u + 6, u + 13);
      expD[doff + 5] = rsum(t, u - 30, u - 12) + rsum(t, u + 13, u + 31);

      let eDist = 0.0;
      let eDist2 = 0.0;
      const pm10 = new Float64Array(10);
      const pd100 = new Float64Array(4);
      const uOff = u * 355;
      for (let c = 0; c < 355; c += 1) {
        const pv = P[ro + c];
        eDist += pv * STATIC.distMat[uOff + c];
        eDist2 += pv * STATIC.distSqMat[uOff + c];
        pm10[c % 10] += pv;
        pd100[Math.floor(c / 100)] += pv;
      }
      expD[doff + 6] = eDist;
      expD[doff + 7] = rsum(t, u + 1, 355);
      expD[doff + 8] = pm10[u % 10];
      expD[doff + 9] = pm10[(u + 1) % 10];
      expD[doff + 10] = pd100[Math.floor(u / 100)];

      for (let k = 0; k < 11; k += 1) {
        const em = expD[doff + k];
        const em2 = k === 6 ? eDist2 : em;
        varD[doff + k] = Math.max(em2 - em * em, 0.0);
      }
    }

    const lbCh = lagB[ro + zt];
    const ocCh = occC[ro + zt];
    for (let bId = 1; bId <= 6; bId += 1) obsD[doff + 10 + bId] = lbCh === bId ? 1.0 : 0.0;
    obsD[doff + 17] = ocCh >= 1 ? 1.0 : 0.0;
    obsD[doff + 18] = ocCh >= 2 ? 1.0 : 0.0;

    const lagGrid = new Float64Array(7);
    const occGrid = new Float64Array(3);
    for (let c = 0; c < 355; c += 1) {
      const pv = P[ro + c];
      lagGrid[lagB[ro + c]] += pv;
      occGrid[occC[ro + c]] += pv;
    }
    for (let bId = 1; bId <= 6; bId += 1) expD[doff + 10 + bId] = lagGrid[bId];
    expD[doff + 17] = occGrid[1] + occGrid[2];
    expD[doff + 18] = occGrid[2];

    if (t >= 2) {
      const loA = apLo[t - 2];
      const hiA = apHi[t - 2];
      obsD[doff + 19] = (zt >= loA && zt < hiA) ? 1.0 : 0.0;
      expD[doff + 19] = cPad[t * 356 + hiA] - cPad[t * 356 + loA];
    }
    for (let k = 11; k < 20; k += 1) {
      const eh = expD[doff + k];
      varD[doff + k] = Math.max(eh * (1.0 - eh), 0.0);
    }
  }

  const out = new Float64Array(355 * 3 + 20 * 3 + 23);
  let outPos = 0;
  const dynParts = [];
  for (const tau of [32.0, 128.0, 1e10]) {
    const w = new Float64Array(L);
    const w2 = new Float64Array(L);
    for (let t = 0; t < L; t += 1) {
      const wt = Math.exp(-t / tau);
      w[t] = wt;
      w2[t] = wt * wt;
    }
    const obsTw = new Float64Array(355);
    const expTw = new Float64Array(355);
    const varTw = new Float64Array(355);
    for (let t = 0; t < L; t += 1) {
      const wt = w[t];
      const w2t = w2[t];
      obsTw[z0[t]] += wt;
      const ro = t * 355;
      for (let c = 0; c < 355; c += 1) {
        const pv = P[ro + c];
        expTw[c] += wt * pv;
        varTw[c] += w2t * pv * (1.0 - pv);
      }
    }
    for (let c = 0; c < 355; c += 1) out[outPos++] = (obsTw[c] - expTw[c]) / Math.sqrt(varTw[c] + 0.10);

    const dynVec = new Float64Array(20);
    for (let k = 0; k < 20; k += 1) {
      let rSum = 0.0;
      let vSum = 0.0;
      for (let t = 0; t < L; t += 1) {
        const idx = t * 20 + k;
        rSum += w[t] * (obsD[idx] - expD[idx]);
        vSum += w2[t] * varD[idx];
      }
      dynVec[k] = rSum / Math.sqrt(vSum + 0.05);
    }
    dynParts.push(dynVec);
  }
  for (const dv of dynParts) {
    for (let k = 0; k < 20; k += 1) out[outPos++] = dv[k];
  }

  const sMean = meanArr(surprisal);
  const sStd = stdArr(surprisal);
  const sQ = quantileLinear(surprisal, [0.1, 0.25, 0.5, 0.75, 0.9]);
  const sTail = [4.5, 5.5, 6.5, 7.5].map((thr) => {
    let c = 0;
    for (let t = 0; t < L; t += 1) if (surprisal[t] > thr) c += 1;
    return c / L;
  });
  const sMinusE = new Float64Array(L);
  let p1 = 0.0;
  let p2 = 0.0;
  let p3 = 0.0;
  let p4 = 0.0;
  let uLt05 = 0;
  let uGt95 = 0;
  for (let t = 0; t < L; t += 1) {
    sMinusE[t] = surprisal[t] - stepEnt[t];
    const xp = xPit[t];
    const xp2 = xp * xp;
    p1 += xp;
    p2 += 0.5 * (3.0 * xp2 - 1.0);
    p3 += 0.5 * (5.0 * xp2 * xp - 3.0 * xp);
    p4 += 0.125 * (35.0 * xp2 * xp2 - 30.0 * xp2 + 3.0);
    if (uMid[t] < 0.05) uLt05 += 1;
    if (uMid[t] > 0.95) uGt95 += 1;
  }
  const tailStats = [
    sMean, sStd, ...sQ, ...sTail,
    meanArr(stepEnt), stdArr(stepEnt), meanArr(sMinusE),
    p1 / L, p2 / L, p3 / L, p4 / L,
    uLt05 / L, uGt95 / L,
    (L >= 4 && sStd > 1e-8) ? corrcoef1d(surprisal.subarray(0, L - 1), surprisal.subarray(1)) : 0.0,
    (L >= 4 && stdArr(xPit) > 1e-8) ? corrcoef1d(xPit.subarray(0, L - 1), xPit.subarray(1)) : 0.0,
    Math.log(L / 300.0),
  ];
  for (let i = 0; i < tailStats.length; i += 1) out[outPos++] = tailStats[i];
  return out;
}

function scorePlRankOnePortable(coefs, M, shared, order) {
  const unobs = new Uint8Array(355);
  unobs.fill(1);
  for (let i = 0; i < order.length; i += 1) unobs[order[i]] = 0;
  let anyUnobs = false;
  for (let z = 0; z < 355; z += 1) if (unobs[z]) { anyUnobs = true; break; }

  const nRows = M + 1;
  const nO = order.length;
  const sc = new Float64Array(nRows);

  for (let r = 0; r < nRows; r += 1) {
    const row = r < M ? coefs.subarray(r * 355, (r + 1) * 355) : shared;
    let outside = -Infinity;
    if (anyUnobs) {
      let mx = -Infinity;
      for (let z = 0; z < 355; z += 1) if (unobs[z] && row[z] > mx) mx = row[z];
      let s = 0.0;
      for (let z = 0; z < 355; z += 1) if (unobs[z]) s += Math.exp(row[z] - mx);
      outside = mx + Math.log(s);
    }
    let suf = row[order[nO - 1]];
    let lossSum = logaddexp(suf, outside) - suf;
    for (let t = nO - 2; t >= 0; t -= 1) {
      const ch = row[order[t]];
      suf = logaddexp(suf, ch);
      lossSum += logaddexp(suf, outside) - ch;
    }
    sc[r] = -(lossSum / nO);
  }
  return { cls: sc.subarray(0, M), pop: sc[M] };
}

// ============================================================================
// 5. Portable Bank Decoding & K=1/2/3 Scoring Engine
// ============================================================================

const DECODED_BANK_CACHE = new WeakMap();

function decodePortableBank(portable) {
  const cached = DECODED_BANK_CACHE.get(portable);
  if (cached) return cached;

  const M = Number(portable.M);
  const arB0 = unpackF64(portable.ar_b0);
  const arA0 = unpackF64(portable.ar_a0);
  const arMBase = new Float64Array(355 * 355);
  for (let u = 0; u < 355; u += 1) {
    for (let v = 0; v < 355; v += 1) {
      const bo = (u * 355 + v) * 11;
      let s = arB0[v];
      for (let k = 0; k < 11; k += 1) s += STATIC.markovB[bo + k] * arA0[k];
      arMBase[u * 355 + v] = s;
    }
  }

  const splines = {};
  for (const [k, sp] of Object.entries(portable.psi_splines)) {
    splines[Number(k)] = {
      xMax: Number(sp.x_max),
      y: unpackF64(sp.y),
      dy: unpackF64(sp.dy),
    };
  }

  const pvs = portable.pvs.map((pv) => {
    const qShared = pv.Q ? unpackMatI16(pv.Q) : null;
    return {
      mu0: unpackF64(pv.mu0),
      sc: unpackF64(pv.sc),
      uCot: unpackF64(pv.u_cot),
      dampC: Number(pv.damp_c),
      W1: unpackMatI16(pv.W1),
      Q1: qShared || unpackMatI16(pv.Q1),
      b1: unpackF64(pv.b1),
      r1: Number(pv.r1),
      W3: unpackMatI16(pv.W3),
      Q3: qShared || unpackMatI16(pv.Q3),
      b3: unpackF64(pv.b3),
      r3: Number(pv.r3),
    };
  });

  const vmfs = portable.vmfs.map((vm) => {
    const r = Number(vm.r);
    const openEst = {};
    for (const [ok, ov] of Object.entries(vm.open_est)) openEst[ok] = unpackF64(ov);
    return {
      r,
      spline: splines[r],
      mu0: unpackF64(vm.mu0),
      sc: unpackF64(vm.sc),
      PProj: unpackMatI16(vm.P_proj),
      offsetSub: unpackF64(vm.offset_sub),
      MUnitBase: unpackF64(vm.M_unit_base),
      kappaC: unpackF64(vm.kappa_c),
      rhoC: unpackF64(vm.rho_c),
      rhoSib: unpackF64(vm.rho_sib),
      effN: unpackF64(vm.eff_n),
      openEst,
    };
  });

  const kr = portable.kr_mm;
  const krMm = {
    nLm: Number(kr.n_lm),
    bw: Number(kr.bw),
    mu: unpackF64(kr.mu),
    sd: unpackF64(kr.sd),
    PKrr: unpackMatI16(kr.P_krr),
    ZLm: unpackMatI16(kr.Z_lm),
    alphaRbf: unpackF64(kr.alpha_rbf),
    alphaLap: unpackF64(kr.alpha_lap),
    HRbf: unpackMatI16(kr.H_rbf),
    HLap: unpackMatI16(kr.H_lap),
  };

  const ref = portable.ref_et;
  const refEt = {
    muC: unpackF64(ref.mu_c),
    scC: unpackF64(ref.sc_c),
    muG: unpackF64(ref.mu_g),
    scG: unpackF64(ref.sc_g),
    pairs: ref.pairs.map((pm) => {
      const wi = unpackI32(pm.wi);
      const wv = unpackF64(pm.wv);
      const wMap = new Map();
      for (let i = 0; i < wi.length; i += 1) wMap.set(wi[i], wv[i]);
      return {
        a: Number(pm.a),
        b: Number(pm.b),
        w1: unpackF64(pm.w1),
        m1: unpackF64(pm.m1),
        wMap,
        emid: Number(pm.emid),
        w2: unpackF64(pm.w2),
        m2: unpackF64(pm.m2),
        muA: unpackF64(pm.mu_a),
        muB: unpackF64(pm.mu_b),
        var2: unpackF64(pm.var2),
        rel: Number(pm.rel),
      };
    }),
  };

  const sb4 = portable.sref_b4;
  const srefB4 = {
    scalers: sb4.scalers.map((s) => {
      const mu = unpackF64(s.mu);
      const sc = unpackF64(s.sc);
      if (s.rc !== undefined) {
        const rc = Number(s.rc);
        return {
          mu,
          sc,
          uc: rc > 0 ? unpackF64(s.uc) : new Float64Array(0),
          dmps: rc > 0 ? unpackF64(s.dmp) : new Float64Array(0),
          rc,
        };
      }
      const dmpVal = Number(s.dmp);
      return {
        mu,
        sc,
        uc: dmpVal > 0.0 ? unpackF64(s.uc) : new Float64Array(0),
        dmps: dmpVal > 0.0 ? Float64Array.of(dmpVal) : new Float64Array(0),
        rc: dmpVal > 0.0 ? 1 : 0,
      };
    }),
    pairs: sb4.pairs.map((pm) => {
      const wi = unpackI32(pm.wi);
      const wv = unpackF64(pm.wv);
      const wMap = new Map();
      for (let i = 0; i < wi.length; i += 1) wMap.set(wi[i], wv[i]);
      return {
        a: Number(pm.a),
        b: Number(pm.b),
        ws: pm.ws.map((w) => unpackF64(w)),
        mids: pm.mids.map((m) => unpackF64(m)),
        wMap,
        emidS: Number(pm.emid_s),
        bias: Number(pm.bias),
        rel: Number(pm.rel),
        vmfMid: Number(pm.vmf_mid || 0.0),
      };
    }),
  };

  const tau1 = Number(portable.tau_1);
  const tau3 = Number(portable.tau_3);
  const tau2 = portable.tau_2 !== undefined ? Number(portable.tau_2) : 0.5 * (tau1 + tau3);

  const decoded = {
    M,
    presentSources: portable.present_sources.slice(),
    hazard: unpackF64(portable.hazard),
    arB0,
    arA0,
    arMBase,
    rvCoefs: unpackF64(portable.rv_coefs),
    rvShared: unpackF64(portable.rv_shared),
    pvs,
    vmfs,
    krMm,
    refEt,
    srefB4,
    theta1: unpackF64(portable.theta_1),
    theta3: unpackF64(portable.theta_3),
    tau1,
    tau2,
    tau3,
  };
  DECODED_BANK_CACHE.set(portable, decoded);
  return decoded;
}

function evalPsiSplineScalar(sp, x) {
  const { xMax, y, dy } = sp;
  const n = y.length;
  if (x >= xMax) return y[n - 1] + dy[n - 1] * (x - xMax);
  const xi = x > 0.0 ? x : 0.0;
  const h = xMax / (n - 1);
  const pos = xi / h;
  const k = Math.min(Math.floor(pos), n - 2);
  const t = pos - k;
  const t2 = t * t;
  const t3 = t2 * t;
  return (2.0 * t3 - 3.0 * t2 + 1.0) * y[k]
    + (t3 - 2.0 * t2 + t) * h * dy[k]
    + (-2.0 * t3 + 3.0 * t2) * y[k + 1]
    + (t3 - t2) * h * dy[k + 1];
}

function evalPsiSplineVec(sp, xs) {
  const out = new Float64Array(xs.length);
  for (let i = 0; i < xs.length; i += 1) out[i] = evalPsiSplineScalar(sp, xs[i]);
  return out;
}

function matVecMul(mat, vec) {
  const { rows, cols, data } = mat;
  const out = new Float64Array(rows);
  for (let r = 0; r < rows; r += 1) {
    const off = r * cols;
    let s = 0.0;
    for (let c = 0; c < cols; c += 1) s += data[off + c] * vec[c];
    out[r] = s;
  }
  return out;
}

function cauchyRadialRow(raw, radius) {
  const rSafe = Math.max(radius, 1e-6);
  const P = raw.length;
  const u = new Float64Array(P);
  let sq = 0.0;
  for (let i = 0; i < P; i += 1) {
    const ui = raw[i] / rSafe;
    u[i] = ui;
    sq += ui * ui;
  }
  const scale = 1.35 / Math.sqrt(1.0 + 0.65 * (sq / P));
  for (let i = 0; i < P; i += 1) u[i] *= scale;
  return u;
}

function scorePvOnePacket(pv, rowsX) {
  const K = rowsX.length;
  const D = pv.mu0.length;
  const P = pv.b1.length;
  const t1Rows = [];
  const t3Rows = [];

  for (let k = 0; k < K; k += 1) {
    const x = rowsX[k];
    const z = new Float64Array(D);
    for (let j = 0; j < D; j += 1) z[j] = (x[j] - pv.mu0[j]) / pv.sc[j];
    if (pv.dampC > 0.0) {
      const proj = pv.dampC * dotArr(z, pv.uCot);
      for (let j = 0; j < D; j += 1) z[j] -= proj * pv.uCot[j];
    }
    const h1 = matVecMul(pv.Q1, z);
    const raw1 = matVecMul(pv.W1, h1);
    for (let p = 0; p < P; p += 1) raw1[p] += pv.b1[p];
    const h3 = pv.Q1 === pv.Q3 ? h1 : matVecMul(pv.Q3, z);
    const raw3 = matVecMul(pv.W3, h3);
    for (let p = 0; p < P; p += 1) raw3[p] += pv.b3[p];
    t1Rows.push(cauchyRadialRow(raw1, pv.r1));
    t3Rows.push(cauchyRadialRow(raw3, pv.r3));
  }

  const mOut = new Float64Array(P);
  const cOut = new Float64Array(P);
  if (K === 1) {
    for (let p = 0; p < P; p += 1) {
      const tb = 0.72 * t1Rows[0][p] + 0.28 * t3Rows[0][p];
      mOut[p] = tb;
      cOut[p] = Math.tanh(tb);
    }
    return { m: mOut, c: cOut };
  }
  if (K === 2) {
    for (let p = 0; p < P; p += 1) {
      const tb0 = 0.50 * t1Rows[0][p] + 0.50 * t3Rows[0][p];
      const tb1 = 0.50 * t1Rows[1][p] + 0.50 * t3Rows[1][p];
      mOut[p] = 0.5 * (tb0 + tb1);
      const th0 = Math.tanh(tb0);
      const th1 = Math.tanh(tb1);
      const mTh = 0.5 * (th0 + th1);
      const vTh = 0.5 * ((th0 - mTh) ** 2 + (th1 - mTh) ** 2);
      cOut[p] = mTh * Math.max(0.25, 1.0 - 0.60 * vTh);
    }
    return { m: mOut, c: cOut };
  }

  const colBuf = new Float64Array(K);
  const thBuf = new Float64Array(K);
  for (let p = 0; p < P; p += 1) {
    let sumTb = 0.0;
    let sumTh = 0.0;
    for (let k = 0; k < K; k += 1) {
      const tb = 0.30 * t1Rows[k][p] + 0.70 * t3Rows[k][p];
      colBuf[k] = tb;
      sumTb += tb;
      const th = Math.tanh(tb);
      thBuf[k] = th;
      sumTh += th;
    }
    colBuf.sort();
    const mid = K >>> 1;
    const med = (K & 1) === 1 ? colBuf[mid] : 0.5 * (colBuf[mid - 1] + colBuf[mid]);
    mOut[p] = 0.65 * (sumTb / K) + 0.35 * med;
    const mTh = sumTh / K;
    let vTh = 0.0;
    for (let k = 0; k < K; k += 1) vTh += (thBuf[k] - mTh) ** 2;
    vTh /= K;
    cOut[p] = mTh * Math.max(0.25, 1.0 - 0.85 * vTh);
  }
  return { m: mOut, c: cOut };
}

function scoreVmfOnePacket(vm, rowsX) {
  const K = rowsX.length;
  const r = vm.r;
  const M = vm.kappaC.length;
  const sp = vm.spline;
  const kapC = vm.kappaC;
  const medKc = medianSortedCopy(kapC);
  const kapU0 = new Float64Array(M);
  const rhoRing = new Float64Array(M);
  for (let c = 0; c < M; c += 1) {
    kapU0[c] = Math.min(kapC[c], medKc * 1.25);
    rhoRing[c] = Math.min(vm.rhoSib[c], 0.52 * vm.rhoC[c]);
  }

  const pb0 = evalPsiSplineVec(sp, vm.rhoC);
  const ps0 = evalPsiSplineVec(sp, vm.rhoSib);
  const pr0 = evalPsiSplineVec(sp, rhoRing);
  const logNormC = evalPsiSplineVec(sp, kapC);
  const logNormU0 = evalPsiSplineVec(sp, kapU0);

  const D = vm.mu0.length;
  const Su = new Float64Array(r);
  const zbar = new Float64Array(r);
  for (let k = 0; k < K; k += 1) {
    const x = rowsX[k];
    const z = new Float64Array(D);
    for (let j = 0; j < D; j += 1) z[j] = (x[j] - vm.mu0[j]) / vm.sc[j];
    const zw = matVecMul(vm.PProj, z);
    for (let j = 0; j < r; j += 1) zw[j] -= vm.offsetSub[j];
    const nrm = Math.max(normArr(zw), 1e-12);
    for (let j = 0; j < r; j += 1) {
      Su[j] += zw[j] / nrm;
      zbar[j] += zw[j] / K;
    }
  }
  const suNorm = normArr(Su);

  const llNoNorm = new Float64Array(M);
  const llc = new Float64Array(M);
  const lls = new Float64Array(M);
  const llu0 = new Float64Array(M);
  const normRing = new Float64Array(M);

  for (let c = 0; c < M; c += 1) {
    const muOff = c * r;
    const kc = kapC[c];
    const rc = vm.rhoC[c];
    const rs = vm.rhoSib[c];
    const rr = rhoRing[c];
    let nb2 = 0.0;
    let ns2 = 0.0;
    let nr2 = 0.0;
    for (let j = 0; j < r; j += 1) {
      const mJ = vm.MUnitBase[muOff + j];
      const uq = kc * Su[j];
      const vb = mJ * rc + uq;
      const vs = mJ * rs + uq;
      const vr = mJ * rr + uq;
      nb2 += vb * vb;
      ns2 += vs * vs;
      nr2 += vr * vr;
    }
    const lnn = evalPsiSplineScalar(sp, Math.sqrt(nb2)) - pb0[c];
    llNoNorm[c] = lnn;
    llc[c] = -K * logNormC[c] + lnn;
    lls[c] = -K * logNormC[c] + evalPsiSplineScalar(sp, Math.sqrt(ns2)) - ps0[c];
    llu0[c] = -K * logNormU0[c] + evalPsiSplineScalar(sp, suNorm * kapU0[c]);
    normRing[c] = Math.sqrt(nr2);
  }

  const llu = logaddexp(
    Math.log(U_UNIFORM_WEIGHT / M) + logsumexp1d(llu0),
    Math.log(U_SIBLING_WEIGHT / M) + logsumexp1d(lls),
  );
  const coh = new Float64Array(M);
  const ring = new Float64Array(M);
  for (let c = 0; c < M; c += 1) {
    coh[c] = llc[c] - llu;
    const llUnifR = evalPsiSplineScalar(sp, suNorm * kapC[c]);
    const llHaloR = evalPsiSplineScalar(sp, normRing[c]) - pr0[c];
    ring[c] = llNoNorm[c] - logaddexp(Math.log(0.28) + llUnifR, Math.log(0.72) + llHaloR);
  }

  const zbNorm = Math.max(normArr(zbar), 1e-12);
  const ubar = new Float64Array(r);
  for (let j = 0; j < r; j += 1) ubar[j] = zbar[j] / zbNorm;

  const est = vm.openEst;
  let kapO; let lnO; let khO; let lnhO;
  if (K >= 3) {
    kapO = est.kap3; lnO = est.ln3; khO = est.kh3; lnhO = est.lnh3;
  } else if (K === 2) {
    kapO = new Float64Array(M);
    khO = new Float64Array(M);
    for (let c = 0; c < M; c += 1) {
      kapO[c] = 0.5 * (est.kap1[c] + est.kap3[c]);
      khO[c] = 0.5 * (est.kh1[c] + est.kh3[c]);
    }
    lnO = evalPsiSplineVec(sp, kapO);
    lnhO = evalPsiSplineVec(sp, khO);
  } else {
    kapO = est.kap1; lnO = est.ln1; khO = est.kh1; lnhO = est.lnh1;
  }

  const llCO = new Float64Array(M);
  const llHO = new Float64Array(M);
  for (let c = 0; c < M; c += 1) {
    const muOff = c * r;
    let cosV = 0.0;
    for (let j = 0; j < r; j += 1) cosV += vm.MUnitBase[muOff + j] * ubar[j];
    llCO[c] = kapO[c] * cosV - lnO[c];
    llHO[c] = khO[c] * cosV - lnhO[c];
  }
  const aM = Math.log(0.30);
  const llUMix = logaddexp(aM, Math.log(0.70 / M) + logsumexp1d(llHO));
  const centAll = new Float64Array(M);
  const centSelf = new Float64Array(M);
  for (let c = 0; c < M; c += 1) {
    centAll[c] = llCO[c] - llUMix;
    centSelf[c] = llCO[c] - logaddexp(aM, Math.log(0.70) + llHO[c]);
  }
  return { coh, ring, centAll, centSelf };
}

function scoreKrrOnePacket(kr, rowsX, M) {
  const K = rowsX.length;
  const D = kr.mu.length;
  const nLm = kr.nLm;
  const dSub = kr.PKrr.rows;
  const out = new Float64Array(M);

  const lmN2 = new Float64Array(nLm);
  for (let l = 0; l < nLm; l += 1) {
    const off = l * dSub;
    let s = 0.0;
    for (let j = 0; j < dSub; j += 1) {
      const v = kr.ZLm.data[off + j];
      s += v * v;
    }
    lmN2[l] = s;
  }

  for (let k = 0; k < K; k += 1) {
    const x = rowsX[k];
    const z = new Float64Array(D);
    for (let j = 0; j < D; j += 1) z[j] = (x[j] - kr.mu[j]) / kr.sd[j];
    const zq = matVecMul(kr.PKrr, z);
    const qN2 = dotArr(zq, zq);

    const kRbf = new Float64Array(nLm);
    const kLap = new Float64Array(nLm);
    for (let l = 0; l < nLm; l += 1) {
      const off = l * dSub;
      let dotL = 0.0;
      for (let j = 0; j < dSub; j += 1) dotL += zq[j] * kr.ZLm.data[off + j];
      const d2 = Math.max(qN2 + lmN2[l] - 2.0 * dotL, 0.0);
      kRbf[l] = Math.exp(-d2 / kr.bw);
      kLap[l] = Math.exp(-Math.sqrt(d2 / kr.bw));
    }

    let exRbf = 0.0;
    let exLap = 0.0;
    for (let i = 0; i < nLm; i += 1) {
      const rOff = i * nLm;
      let hr = 0.0;
      let hl = 0.0;
      for (let j = 0; j < nLm; j += 1) {
        hr += kRbf[j] * kr.HRbf.data[rOff + j];
        hl += kLap[j] * kr.HLap.data[rOff + j];
      }
      exRbf += kRbf[i] * hr;
      exLap += kLap[i] * hl;
    }
    const denRbf = Math.sqrt(Math.max(1.0 - exRbf, 0.0) + 0.08);
    const denLap = Math.sqrt(Math.max(1.0 - exLap, 0.0) + 0.08);

    for (let c = 0; c < M; c += 1) {
      let fRbf = 0.0;
      let fLap = 0.0;
      for (let l = 0; l < nLm; l += 1) {
        const aOff = l * M + c;
        fRbf += kRbf[l] * kr.alphaRbf[aOff];
        fLap += kLap[l] * kr.alphaLap[aOff];
      }
      out[c] += (0.45 * (fRbf / denRbf) + 0.55 * (fLap / denLap)) / K;
    }
  }
  return out;
}

export function evaluateSelectiveDecision(knownProbs, pU, presentSources, options = {}) {
  const opKey = options.operating_point && OPERATING_POINTS[options.operating_point] ? options.operating_point : "default";
  const preset = OPERATING_POINTS[opKey];
  const tauAbs = options.tau_abs != null ? Number(options.tau_abs) : preset.tau_abs;
  const domRival = options.dom_rival != null ? Number(options.dom_rival) : preset.dom_rival;
  const domU = options.dom_u != null ? Number(options.dom_u) : preset.dom_u;

  const M = presentSources.length;
  let topIdx = 0;
  for (let i = 1; i < M; i += 1) {
    if (knownProbs[i] > knownProbs[topIdx]) topIdx = i;
  }
  const p1 = knownProbs[topIdx];
  let p2 = 0.0;
  let runnerLabel = null;
  if (M >= 2) {
    let runnerIdx = -1;
    for (let i = 0; i < M; i += 1) {
      if (i !== topIdx && (runnerIdx === -1 || knownProbs[i] > knownProbs[runnerIdx])) runnerIdx = i;
    }
    p2 = knownProbs[runnerIdx];
    runnerLabel = String(presentSources[runnerIdx]);
  }

  const rivalRatio = p1 / Math.max(p2, 1e-12);
  const uRatio = p1 / Math.max(pU, 1e-12);

  let decision = "accept";
  let predictedSource = String(presentSources[topIdx]);
  let rejectReason = null;
  let rejectReasonZh = null;

  if (!(p1 >= tauAbs && rivalRatio >= domRival && uRatio >= domU)) {
    decision = "reject";
    predictedSource = null;
    if (uRatio < domU) {
      rejectReason = "outside_library_or_incompatible";
      rejectReasonZh = `未知/库外来源概率过高（P(U) = ${(pU * 100).toFixed(1)}%，Top-1 / U = ${uRatio.toFixed(2)}× < ${domU.toFixed(1)}×），疑似未收录型号或受强系统提示词干扰`;
    } else if (rivalRatio < domRival) {
      rejectReason = "insufficient_margin_over_rival";
      rejectReasonZh = `候选近亲区分度不足（领先次优 ${runnerLabel} 仅 ${rivalRatio.toFixed(2)}× < ${domRival.toFixed(1)}×），建议提供完整 3 条挑战回答`;
    } else {
      rejectReason = "below_absolute_probability_floor";
      rejectReasonZh = `最高候选绝对概率不足（P_abs = ${(p1 * 100).toFixed(1)}% < ${(tauAbs * 100).toFixed(0)}%）`;
    }
  }

  return {
    decision,
    predicted_source: predictedSource,
    reject_reason: rejectReason,
    reject_reason_zh: rejectReasonZh,
    top_candidate: String(presentSources[topIdx]),
    absolute_probability: p1,
    runner_up_candidate: runnerLabel,
    runner_up_probability: p2,
    unknown_probability: Number(pU),
    rival_dominance_ratio: rivalRatio,
    unknown_dominance_ratio: uRatio,
    operating_point: opKey,
    thresholds: { tau_abs: tauAbs, dom_rival: domRival, dom_u: domU },
  };
}

function compareIntSeq(a, b) {
  const m = Math.min(a.length, b.length);
  for (let i = 0; i < m; i += 1) {
    if (a[i] !== b[i]) return a[i] - b[i];
  }
  return a.length - b.length;
}

function concatF64(...arrays) {
  let len = 0;
  for (const a of arrays) len += a.length;
  const out = new Float64Array(len);
  let off = 0;
  for (const a of arrays) {
    out.set(a, off);
    off += a.length;
  }
  return out;
}

function takeCols(vec, cols) {
  const out = new Float64Array(cols.length);
  for (let i = 0; i < cols.length; i += 1) out[i] = vec[cols[i]];
  return out;
}

function scoreV6PortablePacket(sequences, decoded, options = {}) {
  const canonMembers = sequences
    .map((seq) => Int32Array.from(seq.filter((v) => v >= VALUE_MIN && v <= VALUE_MAX)))
    .sort(compareIntSeq);
  const K = canonMembers.length;
  const M = decoded.M;

  const v1s = []; const v2s = []; const v3s = []; const v4s = []; const v5s = []; const krrs = []; const m14fs = [];
  const cdfs = []; const cdfStables = []; const lexs = []; const dgs = []; const fulls = []; const vms = []; const geoms = []; const ets = []; const opens = [];
  const hg6s = []; const taboos = []; const grams = [];
  const nSb4Scalers = decoded.srefB4.scalers.length;
  const hasB5 = decoded.srefB4.pairs.length > 0 && nSb4Scalers === 5;
  const hasB6 = decoded.srefB4.pairs.length > 0 && nSb4Scalers === 6;
  const plSum = new Float64Array(M);
  let plPopSum = 0.0;

  for (let k = 0; k < K; k += 1) {
    const arr = canonMembers[k];
    const ordA = distinctOrder(arr);
    const { fh, fPl, fDyn, fet } = extractModelFree(arr);
    const fFreq = extractFreqAndReplag(arr);
    const fGeom = extractGeometryFast(arr);
    const marg99 = precomputeCdfMarg99(arr);
    const f1497 = extractMarkov1497(arr);
    const fChunk = extractChunk1065(arr);
    const fAr = extractArFisher(arr, decoded.arB0, decoded.arA0, decoded.arMBase);
    const { cdf: fc, lex: fl, orders: ow } = extractWindowedCdfLexOrders(arr, ordA, marg99, decoded.hazard);

    const invOw = 1.0 / ow.length;
    for (let w = 0; w < ow.length; w += 1) {
      const { cls, pop } = scorePlRankOnePortable(decoded.rvCoefs, M, decoded.rvShared, ow[w]);
      for (let c = 0; c < M; c += 1) plSum[c] += cls[c] * invOw;
      plPopSum += pop * invOw;
    }

    v1s.push(fc.subarray(0, 195));
    v2s.push(concatF64(fl, fc.subarray(96, 195), fFreq));
    v3s.push(fAr);
    v4s.push(concatF64(f1497, fChunk));
    v5s.push(concatF64(fc, fDyn, fGeom, fPl));
    krrs.push(concatF64(fAr.subarray(0, 1088), f1497, fc, fGeom));
    m14fs.push(concatF64(f1497, fFreq));

    cdfs.push(fc);
    cdfStables.push(takeCols(fc, STABLE_CDF_COLS));
    lexs.push(fl);
    dgs.push(concatF64(fDyn, fGeom));
    fulls.push(concatF64(fc, fPl, fh));
    vms.push(concatF64(f1497, fChunk.subarray(355)));
    geoms.push(fGeom);
    ets.push(fet);
    if (hasB5) opens.push(extractSiblingOpen(arr, decoded.hazard));
    if (hasB6) {
      hg6s.push(concatF64(fh, fGeom));
      taboos.push(extractTabooFast(arr));
      grams.push(extractSeqGrammar(arr));
    }
  }

  const plGain = new Float64Array(M);
  for (let c = 0; c < M; c += 1) plGain[c] = plSum[c] - plPopSum;

  const P = (M * (M - 1)) / 2;
  const pFirst = new Int32Array(P);
  const pSecond = new Int32Array(P);
  let pIdx = 0;
  for (let a = 0; a < M; a += 1) {
    for (let b = a + 1; b < M; b += 1) {
      pFirst[pIdx] = a;
      pSecond[pIdx] = b;
      pIdx += 1;
    }
  }

  const pvOut1 = scorePvOnePacket(decoded.pvs[0], v1s);
  const pvOut2 = scorePvOnePacket(decoded.pvs[1], v2s);
  const pvOut3 = scorePvOnePacket(decoded.pvs[2], v3s);
  const pvOut4 = scorePvOnePacket(decoded.pvs[3], v4s);
  const pvOut5 = scorePvOnePacket(decoded.pvs[4], v5s);

  const krS = scoreKrrOnePacket(decoded.krMm, krrs, M);
  const medKr = medianSortedCopy(krS);
  const stdKr = Math.max(stdArr(krS), 0.12);
  const krZ = new Float64Array(M);
  for (let c = 0; c < M; c += 1) krZ[c] = (krS[c] - medKr) / stdKr;

  const vInputs = [v3s, cdfStables, dgs, fulls, lexs, vms];
  const cohTot = new Float64Array(M);
  const rngTot = new Float64Array(M);
  const centAll = new Float64Array(M);
  const centSelf = new Float64Array(M);
  for (let ch = 0; ch < 6; ch += 1) {
    const wV = VMF_WEIGHTS_6[ch];
    const vRes = scoreVmfOnePacket(decoded.vmfs[ch], vInputs[ch]);
    for (let c = 0; c < M; c += 1) {
      cohTot[c] += wV * vRes.coh[c];
      rngTot[c] += wV * vRes.ring[c];
      centAll[c] += wV * vRes.centAll[c];
      centSelf[c] += wV * vRes.centSelf[c];
    }
  }

  const b4Covered = new Set(decoded.srefB4.pairs.map((pm) => `${pm.a}:${pm.b}`));
  const rVmf0 = new Float64Array(M);
  const rVmf = new Float64Array(M);
  for (let c = 0; c < M; c += 1) {
    const v = GAMMA_COND_VMF * cohTot[c];
    rVmf0[c] = v;
    rVmf[c] = v;
  }

  if (decoded.refEt.pairs.length > 0) {
    const lseR = logsumexp1d(rVmf0);
    const q0 = new Float64Array(M);
    for (let c = 0; c < M; c += 1) q0[c] = Math.exp(rVmf0[c] - lseR);

    const zc = new Float64Array(225);
    for (let j = 0; j < 225; j += 1) {
      let s = 0.0;
      for (let k = 0; k < K; k += 1) s += (cdfs[k][j] - decoded.refEt.muC[j]) / decoded.refEt.scC[j];
      zc[j] = s / K;
    }
    const gDim = decoded.refEt.muG.length;
    const zg = new Float64Array(gDim);
    for (let j = 0; j < gDim; j += 1) {
      let s = 0.0;
      for (let k = 0; k < K; k += 1) s += (geoms[k][j] - decoded.refEt.muG[j]) / decoded.refEt.scG[j];
      zg[j] = s / K;
    }
    const zSub = takeCols(zc, SUBLIMINAL_COLS);
    const zMg = concatF64(takeCols(zc, MACRO_COLS), zg);
    const adj = new Float64Array(M);

    for (const pm of decoded.refEt.pairs) {
      const { a, b } = pm;
      if (b4Covered.has(`${a}:${b}`) || q0[a] + q0[b] < 0.01) continue;
      let etSum = 0.0;
      if (pm.wMap.size > 0) {
        for (let k = 0; k < K; k += 1) {
          const { idx, val } = ets[k];
          for (let i = 0; i < idx.length; i += 1) {
            const wv = pm.wMap.get(idx[i]);
            if (wv !== undefined) etSum += val[i] * wv;
          }
        }
      }
      const etSc = (etSum / K) - pm.emid;
      let dot1 = 0.0;
      for (let j = 0; j < zSub.length; j += 1) dot1 += (zSub[j] - pm.m1[j]) * pm.w1[j];
      const s1 = 0.55 * dot1 + 0.45 * etSc;
      let s2 = 0.0;
      for (let j = 0; j < zMg.length; j += 1) s2 += (zMg[j] - pm.m2[j]) * pm.w2[j];

      const v0 = pm.var2[0] / K;
      const v1 = pm.var2[1] / K;
      const lla = -0.5 * (((s1 - pm.muA[0]) ** 2) / v0 + ((s2 - pm.muA[1]) ** 2) / v1);
      const llb = -0.5 * (((s1 - pm.muB[0]) ** 2) / v0 + ((s2 - pm.muB[1]) ** 2) / v1);
      const lamAb = Math.min(SIBLING_CLIP, Math.max(-SIBLING_CLIP, ALPHA_SIBLING * pm.rel * (lla - llb)));
      adj[a] += q0[b] * lamAb;
      adj[b] -= q0[a] * lamAb;
    }
    for (let c = 0; c < M; c += 1) rVmf[c] = rVmf0[c] + adj[c];
  }

  const b4PairMat = new Float64Array(M * M);
  if (decoded.srefB4.pairs.length > 0) {
    const kGain = K / (1.0 + (K - 1) * 0.12);
    let rawBlocks;
    if (hasB6 && decoded.srefB4.scalers[0].mu.length === 96) {
      rawBlocks = [
        cdfs.map((fc) => fc.subarray(0, 96)),
        cdfStables,
        m14fs.map((mf) => mf.subarray(355, 1497)),
        m14fs,
        taboos,
        grams,
      ];
    } else if (hasB6) {
      rawBlocks = [cdfs, hg6s, m14fs, lexs, taboos, grams];
    } else if (nSb4Scalers === 4 && decoded.srefB4.scalers[0].mu.length === 96) {
      rawBlocks = [
        cdfs.map((fc) => fc.subarray(0, 96)),
        cdfStables,
        m14fs.map((mf) => mf.subarray(355, 1497)),
        m14fs,
      ];
    } else if (hasB5) {
      rawBlocks = [
        cdfs.map((fc) => fc.subarray(0, 96)),
        cdfs,
        m14fs.map((mf) => concatF64(mf.subarray(0, 355), mf.subarray(1497))),
        lexs,
        opens,
      ];
    } else {
      rawBlocks = [cdfs, dgs, m14fs, lexs];
    }
    const nBUse = rawBlocks.length;
    const zBars = [];
    for (let bi = 0; bi < nBUse; bi += 1) {
      const blkRows = rawBlocks[bi];
      const { mu, sc, uc, dmps, rc } = decoded.srefB4.scalers[bi];
      const Db = mu.length;
      const zbMean = new Float64Array(Db);
      const projs = rc > 0 ? new Float64Array(rc) : null;
      for (let k = 0; k < K; k += 1) {
        const row = blkRows[k];
        const z = new Float64Array(Db);
        for (let j = 0; j < Db; j += 1) z[j] = (row[j] - mu[j]) / sc[j];
        if (rc > 0) {
          for (let r = 0; r < rc; r += 1) {
            const off = r * Db;
            let s = 0.0;
            for (let j = 0; j < Db; j += 1) s += z[j] * uc[off + j];
            projs[r] = dmps[r] * s;
          }
          for (let r = 0; r < rc; r += 1) {
            const p = projs[r];
            const off = r * Db;
            for (let j = 0; j < Db; j += 1) z[j] -= p * uc[off + j];
          }
        }
        for (let j = 0; j < Db; j += 1) zbMean[j] += z[j] / K;
      }
      zBars.push(zbMean);
    }

    for (const pm of decoded.srefB4.pairs) {
      const { a, b } = pm;
      let etSc = 0.0;
      if (pm.wMap.size > 0) {
        let etSum = 0.0;
        for (let k = 0; k < K; k += 1) {
          const { idx, val } = ets[k];
          for (let i = 0; i < idx.length; i += 1) {
            const wv = pm.wMap.get(idx[i]);
            if (wv !== undefined) etSum += val[i] * wv;
          }
        }
        etSc = (etSum / K) - pm.emidS;
      }
      let tot1 = etSc + pm.bias;
      for (let bi = 0; bi < nBUse; bi += 1) {
        const zb = zBars[bi];
        const mid = pm.mids[bi];
        const w = pm.ws[bi];
        for (let j = 0; j < zb.length; j += 1) tot1 += (zb[j] - mid[j]) * w[j];
      }
      const llrK = Math.min(9.5, Math.max(-9.5, pm.rel * kGain * tot1));
      b4PairMat[a * M + b] = llrK;
      b4PairMat[b * M + a] = -llrK;
    }
  }

  let thUse; let tauUse;
  if (K >= 3) {
    thUse = decoded.theta3;
    tauUse = decoded.tau3;
  } else if (K === 2) {
    thUse = new Float64Array(13);
    for (let j = 0; j < 13; j += 1) thUse[j] = 0.45 * decoded.theta1[j] + 0.55 * decoded.theta3[j];
    tauUse = decoded.tau2 !== undefined ? decoded.tau2 : 0.50 * (decoded.tau1 + decoded.tau3);
  } else {
    thUse = decoded.theta1;
    tauUse = decoded.tau1;
  }

  const LMat = new Float64Array(M * M);
  for (let p = 0; p < P; p += 1) {
    const a = pFirst[p];
    const b = pSecond[p];
    const logit = pvOut1.m[p] * thUse[0] + pvOut1.c[p] * thUse[1]
      + pvOut2.m[p] * thUse[2] + pvOut2.c[p] * thUse[3]
      + pvOut3.m[p] * thUse[4] + pvOut3.c[p] * thUse[5]
      + pvOut4.m[p] * thUse[6] + pvOut4.c[p] * thUse[7]
      + pvOut5.m[p] * thUse[8] + pvOut5.c[p] * thUse[9]
      + (plGain[a] - plGain[b]) * thUse[10]
      + (rVmf[a] - rVmf[b]) * thUse[11]
      + (krZ[a] - krZ[b]) * thUse[12];
    LMat[a * M + b] = logit;
    LMat[b * M + a] = -logit;
  }
  for (const pm of decoded.srefB4.pairs) {
    const { a, b, rel } = pm;
    if (rel >= 0.70) {
      let lDebiased;
      if (a < b) {
        const p = a * M - ((a * (a + 1)) / 2) + (b - a - 1);
        lDebiased = pvOut1.m[p] * thUse[0] + pvOut1.c[p] * thUse[1]
          + pvOut4.m[p] * thUse[6] + pvOut4.c[p] * thUse[7]
          + (plGain[a] - plGain[b]) * thUse[10]
          + (rVmf[a] - rVmf[b]) * thUse[11];
      } else {
        const p = b * M - ((b * (b + 1)) / 2) + (a - b - 1);
        lDebiased = -(pvOut1.m[p] * thUse[0] + pvOut1.c[p] * thUse[1]
          + pvOut4.m[p] * thUse[6] + pvOut4.c[p] * thUse[7]
          + (plGain[b] - plGain[a]) * thUse[10]
          + (rVmf[b] - rVmf[a]) * thUse[11]);
      }
      const loAb = 0.25 * lDebiased + 1.20 * b4PairMat[a * M + b];
      LMat[a * M + b] = loAb;
      LMat[b * M + a] = -loAb;
    } else {
      const sibBoost = 1.00 * b4PairMat[a * M + b];
      LMat[a * M + b] += sibBoost;
      LMat[b * M + a] -= sibBoost;
    }
  }

  const scaledCond = new Float64Array(M);
  const negRow = new Float64Array(M);
  for (let a = 0; a < M; a += 1) {
    for (let b = 0; b < M; b += 1) negRow[b] = -LMat[a * M + b];
    scaledCond[a] = tauUse * (-logsumexp1d(negRow));
  }
  const lseCond = logsumexp1d(scaledCond);
  const logQCond = new Float64Array(M);
  let topIdx = 0;
  for (let c = 0; c < M; c += 1) {
    logQCond[c] = scaledCond[c] - lseCond;
    if (scaledCond[c] > scaledCond[topIdx]) topIdx = c;
  }

  const piUnknown = options.pi_unknown != null ? Number(options.pi_unknown) : 0.5;
  const logPriorOdds = Math.log((1.0 - piUnknown) / piUnknown);
  const nTop = decoded.vmfs[3].effN[topIdx];
  const sizeScale = nTop / (nTop + 4.5);

  const lbfCoh = Math.max(0.76 * (logsumexp1d(cohTot) - Math.log(M)), -3.60);
  let tauRing; let wKrr; let krrFlr; let centThr;
  if (K >= 3) {
    tauRing = 1.06 * sizeScale; wKrr = 2.20; krrFlr = 0.12; centThr = 1.48 * sizeScale;
  } else if (K === 2) {
    tauRing = 0.75 * sizeScale; wKrr = 1.75; krrFlr = 0.11; centThr = 1.01 * sizeScale;
  } else {
    tauRing = 0.44 * sizeScale; wKrr = 1.30; krrFlr = 0.10; centThr = 0.54 * sizeScale;
  }

  const lbfRing = 0.85 * (rngTot[topIdx] - tauRing) + wKrr * (krS[topIdx] - krrFlr);
  const centLse = logsumexp1d(centAll) - Math.log(M);
  const lbfCent = 0.45 * (0.45 * centLse + 0.55 * centSelf[topIdx] - centThr);
  const lbfKnownTotal = 0.44 * lbfCoh + 0.44 * lbfRing + 0.12 * lbfCent + logPriorOdds;

  const logPU = -logaddexp0(lbfKnownTotal);
  const logPKnown = -logaddexp0(-lbfKnownTotal);
  const jointLp = new Float64Array(M + 1);
  for (let c = 0; c < M; c += 1) jointLp[c] = logPKnown + logQCond[c];
  jointLp[M] = logPU;
  const lseJoint = logsumexp1d(jointLp);
  const probs = new Float64Array(M + 1);
  let probSum = 0.0;
  for (let i = 0; i <= M; i += 1) {
    const p = Math.exp(jointLp[i] - lseJoint);
    probs[i] = p;
    probSum += p;
  }
  for (let i = 0; i <= M; i += 1) probs[i] /= probSum;

  const knownProbs = probs.subarray(0, M);
  const pU = probs[M];
  const condProbs = new Float64Array(M);
  let condSum = 0.0;
  for (let c = 0; c < M; c += 1) {
    const cp = Math.exp(logQCond[c]);
    condProbs[c] = cp;
    condSum += cp;
  }
  for (let c = 0; c < M; c += 1) condProbs[c] /= condSum;

  const decisionInfo = evaluateSelectiveDecision(knownProbs, pU, decoded.presentSources, options);
  return {
    knownProbabilities: knownProbs,
    unknownProbability: pU,
    conditionalProbabilities: condProbs,
    rawScores: scaledCond,
    ...decisionInfo,
  };
}

export function analyzeGlobalOutputs(outputs, bank, options = {}) {
  const modelIds = bank.models.map((model) => model.id);
  const modelEntries = Object.fromEntries(bank.models.map((model) => [model.id, model]));
  const valid = [];
  const diagnostics = [];

  outputs.forEach((item, index) => {
    const expected = Number(item.expected_count || 0);
    const numbers = parseNumbers(item.text || "");
    const minimum = expected ? Math.max(80, Math.ceil(expected * 0.55)) : 80;
    const accepted = numbers.length >= minimum;
    diagnostics.push({ index, parsed_numbers: numbers.length, minimum_numbers: minimum, accepted });
    if (accepted) valid.push({ numbers, counts: countNumbers(numbers) });
  });
  if (!valid.length) {
    throw new Error("没有可用回答：请粘贴完整数字序列；拒答或严重截断的回答不会计入。");
  }

  const pooledCounts = Array.from({ length: DIMENSION }, (_, idx) => valid.reduce((sum, item) => sum + item.counts[idx], 0));
  const calibrationKey = String(Math.min(valid.length, 3));
  const calEntry = bank.calibration?.[calibrationKey] || { beta: 1.0, cv_accuracy: null };

  const decoded = decodePortableBank(bank.v6_portable);
  const v6Out = scoreV6PortablePacket(valid.map((item) => item.numbers), decoded, options);
  const srcMap = Object.fromEntries(decoded.presentSources.map((src, idx) => [src, idx]));

  const familyOrder = [...new Set(bank.models.map((model) => model.family || "models"))];
  const familyNames = Object.fromEntries(familyOrder.map((family) => [
    family,
    bank.models.find((model) => (model.family || "models") === family)?.family_name || FAMILY_DISPLAY_NAMES[family] || family,
  ]));

  const results = modelIds.map((model) => {
    const idx = srcMap[model];
    const entry = modelEntries[model];
    const family = entry.family || "models";
    return {
      model,
      display_name: entry.display_name,
      probability: v6Out.knownProbabilities[idx],
      in_library_probability: v6Out.conditionalProbabilities[idx],
      profile_similarity: jsSimilarity(pooledCounts, entry.counts),
      score: v6Out.rawScores[idx],
      family,
      family_name: familyNames[family],
    };
  }).sort((left, right) => (right.probability - left.probability) || (right.in_library_probability - left.in_library_probability));

  const familyProbabilities = Object.fromEntries(familyOrder.map((family) => [
    family,
    results.filter((item) => item.family === family).reduce((sum, item) => sum + item.probability, 0),
  ]));
  results.forEach((item) => {
    const fp = familyProbabilities[item.family];
    item.conditional_probability = fp > 0 ? item.probability / fp : 0.0;
  });
  const winningFamily = familyOrder.reduce((best, family) => (familyProbabilities[family] > familyProbabilities[best] ? family : best));
  const acceptedId = v6Out.predicted_source;
  const acceptedName = acceptedId ? modelEntries[acceptedId].display_name : "拒绝归因（未达显著优势阈值）";

  return {
    prediction: results[0].model,
    prediction_name: results[0].display_name,
    probability: results[0].probability,
    in_library_probability: results[0].in_library_probability,
    decision: v6Out.decision,
    accepted_model: acceptedId,
    accepted_model_name: acceptedName,
    reject_reason: v6Out.reject_reason,
    reject_reason_zh: v6Out.reject_reason_zh,
    unknown_probability: v6Out.unknown_probability,
    known_total_probability: 1.0 - v6Out.unknown_probability,
    runner_up_candidate: v6Out.runner_up_candidate,
    runner_up_probability: v6Out.runner_up_probability,
    rival_dominance_ratio: v6Out.rival_dominance_ratio,
    unknown_dominance_ratio: v6Out.unknown_dominance_ratio,
    operating_point: v6Out.operating_point,
    thresholds: v6Out.thresholds,
    operating_points: OPERATING_POINTS,
    used_outputs: valid.length,
    results,
    diagnostics,
    calibration: { queries: calibrationKey, beta: Number(calEntry.beta || 1.0), cv_accuracy: calEntry.cv_accuracy ?? null },
    family_prediction: winningFamily,
    family_prediction_name: familyNames[winningFamily],
    family_probability: familyProbabilities[winningFamily],
    family_probabilities: familyOrder.map((family) => ({
      family,
      display_name: familyNames[family],
      probability: familyProbabilities[family],
    })),
    method: "Absolute-Dominance v6（多视角去混杂 + 绝对概率显著优势拒绝）",
  };
}
