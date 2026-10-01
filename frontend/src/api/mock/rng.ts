/** Deterministic PRNG streams for the mock (mulberry32, seeded via splitmix32). */

function splitmix32(a: number): number {
  a |= 0
  a = (a + 0x9e3779b9) | 0
  let t = a ^ (a >>> 16)
  t = Math.imul(t, 0x21f0aaad)
  t = t ^ (t >>> 15)
  t = Math.imul(t, 0x735a2d97)
  return (t ^ (t >>> 15)) >>> 0
}

export class Rng {
  private s: number
  constructor(seed: number) {
    this.s = splitmix32(seed) || 1
  }
  /** Uniform in [0, 1). */
  next(): number {
    this.s = (this.s + 0x6d2b79f5) | 0
    let t = this.s
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  int(n: number): number {
    return Math.floor(this.next() * n)
  }
  normal(): number {
    const u = Math.max(this.next(), 1e-12)
    const v = this.next()
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }
  /** Sample an index from a probability vector; returns [index, draw]. */
  categorical(p: readonly number[]): [number, number] {
    const r = this.next()
    let acc = 0
    for (let i = 0; i < p.length; i++) {
      acc += p[i]
      if (r < acc) return [i, r]
    }
    return [p.length - 1, r]
  }
  shuffle<T>(xs: T[]): T[] {
    const a = [...xs]
    for (let i = a.length - 1; i > 0; i--) {
      const j = this.int(i + 1)
      ;[a[i], a[j]] = [a[j], a[i]]
    }
    return a
  }
}

/** Mirrors the backend's stream layout: env, policy_A, policy_B, gen, perturb. */
export function spawnStreams(seed: number): { env: Rng; policyA: Rng; policyB: Rng; gen: Rng; perturb: Rng } {
  const base = splitmix32(seed * 7919 + 13)
  return {
    env: new Rng(base ^ 0x1001),
    policyA: new Rng(base ^ 0x2002),
    policyB: new Rng(base ^ 0x3003),
    gen: new Rng(base ^ 0x4004),
    perturb: new Rng(base ^ 0x5005),
  }
}
