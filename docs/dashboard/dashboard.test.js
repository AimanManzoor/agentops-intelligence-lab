const assert = require('node:assert/strict');
const { test } = require('node:test');
require('./data.js');
const { selection, failures, mean, csv } = require('./dashboard.js');
const data = globalThis.AGENTOPS_DATA;
test('seeded sample totals', () => {
  assert.equal(data.length, 5000);
  assert.equal(failures(data), 1886);
  assert.equal(new Set(data.map(r => r.run_id)).size, 5000);
});
test('filter combinations and empty selections', () => {
  assert.equal(selection(data, {status:'1'}).length, 1886);
  assert.equal(selection(data, {status:'0'}).length, 3114);
  assert.equal(selection(data, {status:'0',cause:'CPU saturation'}).length, 0);
  const filtered = selection(data, {pipeline:'payment_etl',from:'2026-02-01',to:'2026-02-28'});
  assert.ok(filtered.length > 0);
  assert.ok(filtered.every(r => r.pipeline_name === 'payment_etl' && r.started_at.startsWith('2026-02')));
  assert.equal(mean([], 'cpu_usage_pct'), null);
});
test('CSV includes exactly the selected rows', () => {
  const filtered = selection(data, {status:'1'});
  assert.equal(csv(filtered).split('\n').length, 1887);
  assert.equal(csv([]).split('\n').length, 1);
});
