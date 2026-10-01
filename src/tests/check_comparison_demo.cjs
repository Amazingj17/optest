const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const {pathToFileURL} = require('node:url');
const fs = require('node:fs');
(async () => {
  const browser = await chromium.launch({headless: true, channel: 'msedge'});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(pathToFileURL(process.argv[2]).href);
    assert.equal(await page.locator('#scene option').count(), 6);
    assert.equal(await page.locator('#size option[value="1000"]').count(), 0);
    assert.equal(await page.getByText('各分支独立完成调度', {exact: false}).count(), 0);
    for (let i = 0; i < 6; i++) {
      await page.selectOption('#scene', String(i));
      assert.deepEqual(await page.locator('#methods tr').evaluateAll(rows => rows.map(r => r.dataset.method)),
        ['heft', 'graph_ppo', 'tier_mappo', 'hybrid']);
      for (const method of ['heft', 'graph_ppo', 'tier_mappo', 'hybrid']) {
        await page.locator(`[data-method="${method}"]`).click();
        const actual = await page.evaluate(({i, method}) => {
          const d = JSON.parse(document.getElementById('data').textContent).scenes[i];
          const b = method === 'hybrid' ? d.candidates.find(c => c.name === d.selected_source) : d.comparison_candidates.find(c => c.name === method);
          return {count: document.querySelectorAll('#gantt rect.task').length,
            expected: d.scenario.tasks.length, starts: [...document.querySelectorAll('#gantt rect.task')].map(e => +e.dataset.start),
            expectedStarts: b.entries.map(e => e.start)};
        }, {i, method});
        assert.equal(actual.count, actual.expected);
        assert.deepEqual(actual.starts, actual.expectedStarts);
      }
      await page.locator('#gantt rect.task').first().click();
      assert.match(await page.locator('#detail').innerText(), /工作量/);
    }
    await page.locator('#play').click();
    assert.equal(await page.locator('#play').innerText(), '暂停');
    await page.locator('#play').click();
    await page.evaluate(() => window.scrollTo(0, 0));
    assert.deepEqual(errors, []);
    if (process.argv[3]) await page.screenshot({path: process.argv[3]});
    const result = {passed: true, scenes: 6, strategies: 4, timeline_checks: 24, page_errors: errors};
    if (process.argv[4]) fs.writeFileSync(process.argv[4], JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(e => {console.error(e); process.exit(1);});
