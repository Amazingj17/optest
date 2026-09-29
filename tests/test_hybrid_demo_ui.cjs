// DOM-level interaction test. Requires jsdom; no browser or network is used.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {JSDOM, VirtualConsole} = require('jsdom');
const errors = [], checks = [];
let tick = null;
const virtualConsole = new VirtualConsole();
virtualConsole.on('jsdomError', e => errors.push(String(e)));
const dom = new JSDOM(fs.readFileSync(process.argv[2], 'utf8'), {
  runScripts: 'dangerously', virtualConsole,
  beforeParse(window) {
    window.setInterval = callback => {tick = callback; return 1;};
    window.clearInterval = () => {tick = null;};
  }
});
const win = dom.window, doc = win.document, $ = id => doc.getElementById(id);
const payload = JSON.parse($('data').textContent);
assert.equal($('scene').options.length, payload.scenes.length);
checks.push('all bundled scene options render');
for (let i = 0; i < payload.scenes.length; i++) {
  $('scene').value = String(i);
  $('scene').dispatchEvent(new win.Event('change'));
  const scene = payload.scenes[i];
  assert.equal($('ratio').textContent, scene.record.ratio.toFixed(6));
  assert.equal($('gantt').querySelectorAll('rect.task').length, scene.scenario.tasks.length);
  assert.equal($('dag').querySelectorAll('g.task').length, scene.scenario.tasks.length);
  for (const method of ['heft', 'search_blocks', 'residual_hrl', 'hybrid']) {
    doc.querySelector(`[data-method="${method}"]`).click();
    assert.equal(doc.querySelector('.method.active').dataset.method, method);
    $('gantt').querySelector('rect.task').dispatchEvent(new win.MouseEvent('click'));
    assert.match($('detail').textContent, /工作量/);
  }
  checks.push(`scene ${i + 1}: metrics, DAG, four strategy timelines and task details`);
}
for (const size of ['50', '100', '300']) {
  $('size').value = size;
  $('size').dispatchEvent(new win.Event('change'));
  assert.equal($('scene').options.length, 2);
  $('variant').value = 'heterogeneous';
  $('variant').dispatchEvent(new win.Event('change'));
  assert.equal($('scene').options.length, 1);
  $('variant').value = 'all';
  $('variant').dispatchEvent(new win.Event('change'));
}
checks.push('all task-size and resource-type filters');
$('play').click();
assert.equal($('play').textContent, '暂停');
assert.equal(typeof tick, 'function');
tick();
assert.equal($('progress').value, '5');
$('play').click();
assert.equal(tick, null);
$('progress').value = '500';
$('progress').dispatchEvent(new win.Event('input'));
assert.notEqual($('cursor').getAttribute('x1'), '0');
checks.push('play, pause, slider and timeline cursor');
assert.equal(errors.length, 0, errors.join('\n'));
const result = {passed: true, checks, js_errors: errors,
  scope: 'jsdom interaction verification; browser visual rendering not checked'};
if (process.argv[3]) fs.writeFileSync(process.argv[3], JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
dom.window.close();
