import { expect, test, type Page } from '@playwright/test'

const SHOTS = 'test-results'

function collectErrors(page: Page): string[] {
  const errors: string[] = []
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text())
  })
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`))
  return errors
}

test('run loop, interventions, composer, preset comparison, export/import replay (mock)', async ({ page }) => {
  const errors = collectErrors(page)
  await page.goto('/?mock=1')
  await page.getByTestId('enable-audio').click()

  // preset comparison table
  await page.getByRole('button', { name: 'Run preset comparison' }).first().click()
  await expect(page.getByTestId('preset-comparison')).toContainText('Mean score, first 20% of episodes')

  await page.getByTestId('start-shared_history').click()
  await expect(page.getByTestId('start-menu')).toBeHidden()

  // 1x run for a moment: waits for audio, so only a few episodes complete
  await page.getByTestId('start').click()
  await expect(page.getByTestId('pause')).toBeVisible()
  await page.waitForTimeout(1500)
  await page.getByTestId('pause').click()
  await expect(page.getByTestId('start')).toBeVisible()
  const afterSlow = Number((await page.getByTestId('episode-counter').innerText()).split('/')[0].trim())
  expect(afterSlow).toBeGreaterThanOrEqual(1)
  expect(afterSlow).toBeLessThanOrEqual(2)

  // fast mode: n=10 per request, no audio
  await page.getByRole('button', { name: 'no audio / fast' }).click()
  await page.getByTestId('start').click()
  await expect(page.getByTestId('episode-counter')).toContainText(/^\s*(4\d|5\d|6\d|7\d|8\d|9\d|1\d\d)/, { timeout: 15_000 })
  await page.getByTestId('pause').click()

  // interventions
  await page.getByRole('button', { name: 'Clear memory' }).click()
  await page.getByRole('checkbox', { name: 'freeze state' }).first().check()
  await page.getByRole('combobox', { name: 'Transform' }).selectOption('rhythm_shuffle')
  await page.getByRole('button', { name: 'Apply' }).click()
  await page.getByTestId('step').click()
  await expect(page.getByTestId('timeline')).toContainText('effective from step')
  await page.getByRole('tab', { name: /Log/ }).click()
  await expect(page.getByTestId('event-log')).toContainText('Swap musical characteristic')
  await page.getByRole('tab', { name: 'Interventions' }).click()

  // replay a motif from the timeline (plays + queues replay_motif)
  await page.getByTestId('timeline-row').nth(2).getByRole('button').click()
  await page.getByTestId('step').click()
  await expect(page.getByTestId('timeline-row').first()).toContainText('Replay')

  // composer: add notes and send as human
  await page.getByRole('tab', { name: 'Compose' }).click()
  for (const k of ['C4', 'E4', 'G4', 'C5']) await page.getByRole('button', { name: k, exact: true }).click()
  await page.getByRole('button', { name: 'Save motif' }).click()
  await page.getByRole('button', { name: 'Measure features' }).click()
  await expect(page.locator('.features')).toContainText('4 notes')
  const before = await page.getByTestId('timeline-row').count()
  await page.getByRole('button', { name: 'Send as human' }).click()
  await expect(page.getByTestId('timeline-row')).toHaveCount(before + 1)
  await page.screenshot({ path: `${SHOTS}/05-after-interventions.png` })

  // config popover
  await page.getByTestId('config-button').click()
  await expect(page.getByRole('dialog', { name: 'Configuration summary' })).toContainText('model input modality')
  await page.getByTestId('config-button').click()

  // export -> import => REPLAY badge, Step re-emits recorded events
  const [download] = await Promise.all([page.waitForEvent('download'), page.getByTestId('export').click()])
  const file = `${SHOTS}/exported-run.json`
  await download.saveAs(file)
  await page.locator('input[type=file]').first().setInputFiles(file)
  await expect(page.getByTestId('data-source-badge')).toContainText('REPLAY')
  await page.getByTestId('step').click()
  await page.getByTestId('step').click()
  await expect(page.getByTestId('episode-counter')).toContainText('2')
  await page.screenshot({ path: `${SHOTS}/06-replay.png` })

  expect(errors, `console errors:\n${errors.join('\n')}`).toEqual([])
})

test('usable at 1024 px wide', async ({ page }) => {
  const errors = collectErrors(page)
  await page.setViewportSize({ width: 1024, height: 768 })
  await page.goto('/?mock=1')
  await page.getByTestId('start-first_encounter').click()
  for (let i = 0; i < 3; i++) await page.getByTestId('step').click()
  await expect(page.getByTestId('timeline-row')).toHaveCount(3)
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
  expect(overflow).toBeLessThanOrEqual(0)
  await page.screenshot({ path: `${SHOTS}/07-1024.png`, fullPage: true })
  expect(errors).toEqual([])
})
