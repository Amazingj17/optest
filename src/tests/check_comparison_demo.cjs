const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const {pathToFileURL} = require('node:url');
const fs = require('node:fs');
const data = JSON.parse(fs.readFileSync(process.argv[2].replace(/index\.html$/, 'results.json'), 'utf8'));
(async () => {
  const browser = await chromium.launch({headless: true, channel: 'msedge'});
  try {
    const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
    const errors = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto(pathToFileURL(process.argv[2]).href);
    assert.equal(await page.locator('#scene option').count(), data.scenes.length);
    assert.equal(await page.locator('#size option[value="1000"]').count(), 0);
    assert.equal(await page.getByText('各分支独立完成调度', {exact: false}).count(), 0);
    let timelineChecks = 0;
    for (let i = 0; i < data.scenes.length; i++) {
      await page.selectOption('#scene', String(i));
      const scene = data.scenes[i];
      const candidates = scene.comparison_candidates ?? scene.candidates;
      const methods = [...candidates.map(c => c.name), 'hybrid'];
      assert.deepEqual(await page.locator('#methods tr').evaluateAll(rows => rows.map(r => r.dataset.method)),
        methods);
      assert.equal(await page.locator('#valid').innerText(),
        `${candidates.filter(c => c.valid).length + Number(scene.record.valid_schedule)} / ${methods.length}`);
      assert.ok(await page.locator('#ratio').innerText());
      for (const method of methods) {
        await page.locator(`[data-method="${method}"]`).click();
        const expected = method === 'hybrid'
          ? scene.candidates.find(c => c.name === scene.selected_source)
          : candidates.find(c => c.name === method);
        const actual = await page.evaluate(() => {
          return {count: document.querySelectorAll('#gantt rect.task').length,
            dagTasks: document.querySelectorAll('#dag g.task').length,
            starts: [...document.querySelectorAll('#gantt rect.task')].map(e => +e.dataset.start)};
        });
        assert.equal(actual.count, scene.scenario.tasks.length);
        assert.equal(actual.dagTasks, scene.scenario.tasks.length);
        assert.deepEqual(actual.starts, expected.entries.map(e => e.start));
        timelineChecks++;
      }
      await page.locator('#gantt rect.task').first().click();
      assert.match(await page.locator('#detail').innerText(), /工作量/);
    }
    await page.locator('#play').click();
    assert.equal(await page.locator('#play').innerText(), '暂停');
    await page.locator('#play').click();
    await page.locator('#progress').fill('500');
    await page.locator('#progress').dispatchEvent('input');
    assert.equal(await page.locator('#play').innerText(), '播放调度');
    assert.match(await page.locator('#time').innerText(), /^t = \d/);
    for (const size of [...new Set(data.scenes.map(s => s.scenario.tasks.length))]) {
      await page.selectOption('#size', String(size));
      assert.equal(await page.locator('#scene option').count(),
        data.scenes.filter(s => s.scenario.tasks.length === size).length);
    }
    await page.selectOption('#size', 'all');
    for (const variant of ['heterogeneous', 'homogeneous']) {
      const count = data.scenes.filter(s => s.scenario.scenario_id.endsWith(`:${variant}`)).length;
      if (!count) continue;
      await page.selectOption('#variant', variant);
      assert.equal(await page.locator('#scene option').count(), count);
    }
    await page.selectOption('#variant', 'all');
    await page.evaluate(() => window.scrollTo(0, 0));
    assert.deepEqual(errors, []);
    if (process.argv[3]) await page.screenshot({path: process.argv[3]});
    const result = {passed: true, scenes: data.scenes.length, timeline_checks: timelineChecks, page_errors: errors};
    if (process.argv[4]) fs.writeFileSync(process.argv[4], JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result));
  } finally { await browser.close(); }
})().catch(e => {console.error(e); process.exit(1);});
