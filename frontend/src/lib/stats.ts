export function mean(xs: readonly number[]): number {
  if (xs.length === 0) return 0
  let s = 0
  for (const x of xs) s += x
  return s / xs.length
}

/** Sample standard deviation (n-1). Returns 0 for n < 2. */
export function sd(xs: readonly number[]): number {
  if (xs.length < 2) return 0
  const m = mean(xs)
  let s = 0
  for (const x of xs) s += (x - m) * (x - m)
  return Math.sqrt(s / (xs.length - 1))
}

export function rollingMean(xs: readonly number[], window: number): number[] {
  const out: number[] = []
  let acc = 0
  for (let i = 0; i < xs.length; i++) {
    acc += xs[i]
    if (i >= window) acc -= xs[i - window]
    out.push(acc / Math.min(i + 1, window))
  }
  return out
}

export function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x
}

export function softmax(scores: readonly number[], temperature: number): number[] {
  const t = Math.max(temperature, 1e-6)
  const m = Math.max(...scores)
  const ex = scores.map((s) => Math.exp((s - m) / t))
  const z = ex.reduce((a, b) => a + b, 0)
  return ex.map((e) => e / z)
}

export function cosine(a: readonly number[], b: readonly number[]): number {
  let dot = 0
  let na = 0
  let nb = 0
  const n = Math.min(a.length, b.length)
  for (let i = 0; i < n; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  if (na === 0 || nb === 0) return 0
  return dot / Math.sqrt(na * nb)
}

export function pearson(a: readonly number[], b: readonly number[]): number {
  const n = Math.min(a.length, b.length)
  if (n < 2) return 0
  const ma = mean(a.slice(0, n))
  const mb = mean(b.slice(0, n))
  let num = 0
  let da = 0
  let db = 0
  for (let i = 0; i < n; i++) {
    num += (a[i] - ma) * (b[i] - mb)
    da += (a[i] - ma) ** 2
    db += (b[i] - mb) ** 2
  }
  if (da === 0 || db === 0) return 0
  return num / Math.sqrt(da * db)
}

/** Split into `blocks` contiguous blocks and return each block's mean. */
export function blockMeans(xs: readonly number[], blocks: number): number[] {
  if (xs.length === 0) return Array(blocks).fill(0)
  const out: number[] = []
  for (let b = 0; b < blocks; b++) {
    const lo = Math.floor((b * xs.length) / blocks)
    const hi = Math.max(lo + 1, Math.floor(((b + 1) * xs.length) / blocks))
    out.push(mean(xs.slice(lo, Math.min(hi, xs.length))))
  }
  return out
}

export function kl(p: readonly number[], q: readonly number[]): number {
  let s = 0
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 0) s += p[i] * Math.log(p[i] / Math.max(q[i], 1e-12))
  }
  return s
}
