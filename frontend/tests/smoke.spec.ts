import { expect, test } from '@playwright/test'

const SHOTS = 'test-results'

test('lab smoke (mock mode)', async ({ page }) => {
  const errors: string[] = []
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(m.text())
  })
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`))

  await page.goto('/?mock=1')
  await expect(page.getByTestId('data-source-badge')).toContainText('DEMO DATA (mock)')
  await expect(page.getByTestId('footer-note')).toContainText('Engineered agent simulation')
  await expect(page.getByTestId('start-menu')).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/01-start-menu.png` })

  // audio: enabled only from a user gesture
  await page.getByTestId('enable-audio').click()
  await expect(page.getByTestId('audio-status')).toHaveText(/audio on|audio paused/)

  // start a preset
  await page.getByTestId('start-first_encounter').click()
  await expect(page.getByTestId('start-menu')).toBeHidden()
  await expect(page.getByTestId('agent-panel-A')).toBeVisible()
  await expect(page.getByTestId('agent-panel-B')).toBeVisible()

  // step 3 times
  for (let i = 1; i <= 3; i++) {
    await page.getByTestId('step').click()
    await expect(page.getByTestId('episode-counter')).toContainText(String(i))
  }
  await expect(page.getByTestId('timeline-row')).toHaveCount(3)
  await expect(page.getByTestId('piano-roll')).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/02-lab-after-3-steps.png` })

  // select a timeline row and open the inspector
  await page.getByTestId('timeline-row').nth(1).click()
  await page.getByTestId('inspect-B').click()
  await expect(page.getByTestId('inspector')).toBeVisible()
  await expect(page.getByTestId('inspector')).toContainText('Learned associations')
  await page.screenshot({ path: `${SHOTS}/03-inspector.png` })
  await page.getByRole('button', { name: 'Close inspector' }).click()
  await expect(page.getByTestId('inspector')).toBeHidden()

  // experiments
  await page.getByTestId('open-experiments').click()
  await expect(page.getByTestId('experiments-view')).toBeVisible()
  await expect(page.getByTestId('experiment-caveat')).toContainText('Runs (seeds) are the unit of analysis')
  await page.getByTestId('run-experiment').click()
  await expect(page.getByTestId('experiment-table')).toBeVisible({ timeout: 30_000 })
  await page.screenshot({ path: `${SHOTS}/04-experiments.png`, fullPage: true })

  expect(errors, `console errors:\n${errors.join('\n')}`).toEqual([])
})
