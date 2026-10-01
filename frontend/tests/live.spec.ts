import { expect, test } from '@playwright/test'

/**
 * Opt-in: the same flow as smoke.spec.ts but against the REAL backend.
 *   VITE_API_TARGET=http://127.0.0.1:8000 LIVE_SMOKE=1 npx playwright test tests/live.spec.ts
 */
test.skip(!process.env.LIVE_SMOKE, 'set LIVE_SMOKE=1 (and VITE_API_TARGET) to run against a live backend')

test('lab against the live API', async ({ page }) => {
  const errors: string[] = []
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text())
  })
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`))

  await page.goto('/')
  await expect(page.getByTestId('data-source-badge')).toContainText('LIVE SIMULATION')
  await page.getByTestId('enable-audio').click()
  await page.getByTestId('start-first_encounter').click()
  await expect(page.getByTestId('start-menu')).toBeHidden()
  for (let i = 1; i <= 3; i++) {
    await page.getByTestId('step').click()
    await expect(page.getByTestId('episode-counter')).toHaveText(new RegExp(`^\\s*${i}\\s*/`))
  }
  await expect(page.getByTestId('timeline-row')).toHaveCount(3)
  await page.screenshot({ path: 'test-results/live-01-lab.png' })

  // fast run of 30 more episodes
  await page.getByRole('button', { name: 'no audio / fast' }).click()
  await page.getByRole('spinbutton').first().fill('33')
  await page.getByTestId('start').click()
  await expect(page.getByTestId('episode-counter')).toHaveText(/^\s*33\s*\//, { timeout: 20_000 })
  await expect(page.getByTestId('start')).toBeVisible()

  // interventions
  await page.getByRole('button', { name: 'Clear memory' }).click()
  await page.getByRole('checkbox', { name: 'disable coupling' }).first().check()
  await page.getByTestId('step').click()
  await expect(page.getByTestId('timeline')).toContainText('effective from step 33')

  await page.getByTestId('timeline-row').nth(1).click()
  await page.getByTestId('inspect-A').click()
  await expect(page.getByTestId('inspector')).toContainText('Learned associations')
  await page.screenshot({ path: 'test-results/live-02-inspector.png' })
  await page.getByRole('button', { name: 'Close inspector' }).click()

  // replay a motif from the timeline (posts replay_motif)
  await page.getByTestId('timeline-row').first().getByRole('button').click()
  await page.getByTestId('step').click()

  // composer -> human phrase
  await page.getByRole('tab', { name: 'Compose' }).click()
  for (const k of ['C4', 'E4', 'G4']) await page.getByRole('button', { name: k, exact: true }).click()
  await page.getByRole('button', { name: 'Send as human' }).click()
  await expect(page.getByTestId('timeline-row').first()).toContainText('human')

  // WAV export of the shown phrase (POST /api/phrases/render)
  const [wav] = await Promise.all([page.waitForEvent('download'), page.getByRole('button', { name: '⤓ WAV' }).click()])
  const wavPath = `test-results/live-phrase.wav`
  await wav.saveAs(wavPath)
  const bytes = (await import('node:fs')).readFileSync(wavPath)
  expect(bytes.subarray(0, 4).toString('latin1')).toBe('RIFF')
  expect(bytes.length).toBeGreaterThan(10_000)

  await page.getByTestId('open-experiments').click()
  await page.getByRole('spinbutton', { name: /episodes per run/ }).fill('30')
  await page.getByTestId('run-experiment').click()
  await expect(page.getByTestId('experiment-table')).toBeVisible({ timeout: 60_000 })
  await page.screenshot({ path: 'test-results/live-03-experiments.png' })
  await page.getByRole('button', { name: 'Close experiments' }).click()

  // export -> import (replay)
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByTestId('export').click()])
  await download.saveAs('test-results/live-run.json')
  await page.locator('input[type=file]').first().setInputFiles('test-results/live-run.json')
  await expect(page.getByTestId('data-source-badge')).toContainText('REPLAY')
  await page.getByTestId('step').click()
  await expect(page.getByTestId('episode-counter')).toHaveText(/^\s*1\s*\//)
  await page.screenshot({ path: 'test-results/live-04-replay.png' })

  expect(errors, `console errors:\n${errors.join('\n')}`).toEqual([])
})
