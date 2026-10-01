import { defineConfig } from 'vitest/config'

// Unit tests only; the Playwright smoke test lives in tests/ and is run with `npx playwright test`.
export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    environment: 'node',
  },
})
