const assert = require('node:assert/strict');
const {test} = require('node:test');
const {csv, csvCell} = require('./dashboard.js');

test('formula prefixes are treated as text', () => {
  for (const prefix of ['=', '+', '-', '@', '＝', '＋', '－', '＠']) {
    assert.equal(csvCell(prefix + '1'), '"\'' + prefix + '1"');
  }
});

test('leading whitespace and control characters are handled', () => {
  for (const text of [' =1', '\t=1', '\r=1', '\n=1', '\uFEFF=1', '\tplain']) {
    assert.equal(csvCell(text), '"\'' + text + '"');
  }
});

test('ordinary values retain their representation', () => {
  for (const [value, expected] of [
    ['payment_etl', '"payment_etl"'], [-2, '"-2"'],
    [0, '"0"'], [false, '"false"'], [null, '""'], [undefined, '""']
  ]) {
    assert.equal(csvCell(value), expected);
  }
});

test('quotes separators and newlines remain inside a quoted field', () => {
  assert.equal(csvCell('a,"b";c\nd'), '"a,""b"";c\nd"');
  assert.equal(csvCell('=1,"next"'), '"\'=1,""next"""');
});

test('full export applies the protection without mutating source data', () => {
  const row = {run_id: 1, pipeline_name: '=1+1', root_cause: '@SUM(1)'};
  const before = JSON.stringify(row);
  const output = csv([row]);
  assert.ok(output.includes('"\'=1+1"'));
  assert.ok(output.includes('"\'@SUM(1)"'));
  assert.equal(JSON.stringify(row), before);
});

test('empty export contains only the fixed header', () => {
  assert.equal(csv([]), 'run_id,pipeline_name,started_at,expected_workload,cpu_usage_pct,memory_usage_pct,recent_failure_rate,failure_probability,target_failed,root_cause');
});
