import { defineConfig } from '@playwright/test'

/**
 * Smoke test in MOCK mode (?mock=1) against `vite preview` of the production build.
 * Uses the pre-installed Chromium under /opt/pw-browsers (do not run `playwright install`).
 * Override with PW_CHROMIUM=/path/to/chrome or SMOKE_BASE_URL=http://host:port (skips the web server).
 */
const executablePath = process.env.PW_CHROMIUM ?? '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'
const port = Number(process.env.SMOKE_PORT ?? 4179)
const external = process.env.SMOKE_BASE_URL

export default defineConfig({
  testDir: './tests',
  outputDir: './test-results/artifacts',
  timeout: 90_000,
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  use: {
    browserName: 'chromium',
    baseURL: external ?? `http://127.0.0.1:${port}`,
    viewport: { width: 1440, height: 900 },
    launchOptions: {
      executablePath,
      args: ['--autoplay-policy=no-user-gesture-required', '--use-fake-device-for-media-stream'],
    },
    trace: 'off',
  },
  webServer: external
    ? undefined
    : {
        command: `npx vite build && npx vite preview --port ${port} --strictPort --host 127.0.0.1`,
        url: `http://127.0.0.1:${port}`,
        reuseExistingServer: false,
        timeout: 120_000,
      },
})
