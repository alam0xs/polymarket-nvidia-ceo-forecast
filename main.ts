// polymarket-nvidia-ceo-forecast-01 entry v2.
//
// Model (all fitted on the practice cases only, by local backtest on the
// five overlapping cases p00..p04; see analysis in the build session):
//  - Median: persistence, drifting toward the cross-sectional
//    sum-to-one target. The 24 Yes prices should sum to ~1; when the
//    checkpoint sum S deviates, each market's median drifts toward
//    x * (1/S), ramping in with time constant 120 min
//    (no shift at h=1, ~half the shift by h~80). No long-run mean
//    reversion: in the practice window that hurt badly (prices were
//    trending, e.g. Jensen Huang (No Change) collapsing 0.24 -> 0.06),
//    and empirical forward-delta quantiles were poisoned by the Nov 10-11
//    flip regime (markets flipping 0.03 <-> 0.45) and also hurt.
//  - Spread: normal quantiles around the median with per-market sigma
//    estimated from the last 120 one-minute diffs, winsorised at +-0.02
//    (the raw sd is dominated by rare huge jumps; the winsorised recent
//    sd tracks the current regime), x0.85, floored at 0.0003 and capped
//    at 0.005, growing as sqrt(h). Markets with <30 diffs shrink their
//    sigma toward the pooled cross-market recent sigma. Markets priced
//    >= 0.10 get a sigma floor of 0.0015: the big markets move in bursts
//    and their last-2h realised vol understates their 4h moves.
//    The spread is asymmetric: the downside z is x1.2, the upside x0.85
//    (in the practice, 4h falls were larger than rises).
//  - Local backtest (overlapping p00..p04, 104 market-cases):
//    house baseline sigma 0.0025: 0.3925; this model: ~0.290.
//    Full-practice house baseline is 0.363589 (includes quiet p05).
//    Rolling-origin backtest (622 market-cases, Nov 10-12): baseline
//    1.502, this model 1.449; Nov 12-13 subset: baseline 0.411, this 0.325.
//
// Deterministic, stdlib only, offline. Reads every *.in in the input
// directory and writes one *.out per case.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Normal quantiles, with the downside widened x1.2 and the upside x0.85. */
const Z = [-1.644854 * 1.2, -0.67449 * 1.2, 0, 0.67449 * 0.85, 1.644854 * 0.85];

const WIN = 120;          // minutes of history used for the sigma estimate
const WINSOR = 0.02;      // winsorise 1-min diffs at +-0.02
const SIGMA_SCALE = 0.85; // shrink the raw recent sd slightly (backtest)
const SIGMA_FLOOR = 0.0003;
const SIGMA_CAP = 0.005;
const SIGMA_TOP_FLOOR = 0.0015; // for markets priced >= 0.10
const RENORM_A = 1.0;     // fraction of the sum-to-one correction applied
const RENORM_TAU = 120;   // minutes over which the median correction ramps in

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  quantiles: number[];
  markets: CaseMarket[];
}

const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** Winsorised sd of the 1-min diffs in the last WIN minutes of `prices`. */
function recentSigma(prices: number[]): number {
  const n = Math.min(WIN, prices.length - 1);
  if (n < 2) return 0.0015;
  let sum = 0;
  let sumSq = 0;
  for (let i = prices.length - n; i < prices.length; i++) {
    const d = prices[i] - prices[i - 1];
    const w = d > WINSOR ? WINSOR : d < -WINSOR ? -WINSOR : d;
    sum += w;
    sumSq += w * w;
  }
  const mean = sum / n;
  const variance = Math.max(0, sumSq / n - mean * mean);
  return Math.sqrt(variance);
}

function forecastAll(c: Case): { id: string; q: number[][] }[] {
  const horizon = c.horizon_minutes;
  const sumLast = c.markets.reduce((s, m) => s + (m.prices.length ? m.prices[m.prices.length - 1] : 0), 0);

  // Pooled recent sigma across markets, for shrinking very short histories.
  let pooledSigma = 0.0015;
  {
    let sum = 0;
    let sumSq = 0;
    let n = 0;
    for (const m of c.markets) {
      const k = Math.min(WIN, m.prices.length - 1);
      for (let i = m.prices.length - k; i < m.prices.length; i++) {
        const d = m.prices[i] - m.prices[i - 1];
        const w = d > WINSOR ? WINSOR : d < -WINSOR ? -WINSOR : d;
        sum += w;
        sumSq += w * w;
        n++;
      }
    }
    if (n > 10) {
      const mean = sum / n;
      pooledSigma = Math.sqrt(Math.max(0, sumSq / n - mean * mean));
    }
  }

  return c.markets.map((m) => {
    if (m.prices.length === 0) throw new Error(`market ${m.id} has no prices`);
    const x = m.prices[m.prices.length - 1] as number;
    let sigma = recentSigma(m.prices) * SIGMA_SCALE;
    const nDiffs = m.prices.length - 1;
    if (nDiffs < 30) {
      // Short history: shrink toward the pooled sigma.
      const w = nDiffs / 30;
      sigma = w * sigma + (1 - w) * pooledSigma * SIGMA_SCALE;
    }
    sigma = Math.min(SIGMA_CAP, Math.max(SIGMA_FLOOR, sigma));
    if (x >= 0.1) sigma = Math.max(sigma, SIGMA_TOP_FLOOR);

    // Median target from the sum-to-one correction; ramp it in over time.
    const target = sumLast > 0 ? x * (1 + RENORM_A * (1 / sumLast - 1)) : x;
    const rows: number[][] = [];
    for (let h = 1; h <= horizon; h++) {
      const ramp = 1 - Math.exp(-h / RENORM_TAU);
      const med = x + (target - x) * ramp;
      const sd = sigma * Math.sqrt(h);
      const row = Z.map((z) => price6(med + z * sd));
      // Enforce non-decreasing after rounding/clipping.
      for (let j = 1; j < row.length; j++) {
        if (row[j] < row[j - 1]) row[j] = row[j - 1];
      }
      rows.push(row);
    }
    return { id: m.id, q: rows };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.quantiles.length !== QUANTILES.length || c.quantiles.some((t, j) => t !== QUANTILES[j])) throw new Error(`${file} asks for quantiles ${JSON.stringify(c.quantiles)}, not ${JSON.stringify(QUANTILES)}`);
  const forecasts = forecastAll(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
