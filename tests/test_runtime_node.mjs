// Fast zero-GPU Node.js unit tests for FrameFlow JS runtime loader, graph compiler & pacer.
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { f16ToF32, parseModel, compilePlan, evalFilmMlp, align32 } from '../runtime/loader.js';
import { FrameFlowPacer } from '../runtime/pacer.js';

test('f16ToF32 decodes exact special and normal values', () => {
  assert.equal(f16ToF32(0x0000), 0);
  assert.equal(f16ToF32(0x3c00), 1.0);
  assert.equal(f16ToF32(0xbc00), -1.0);
  assert.equal(f16ToF32(0x3800), 0.5);
  assert.equal(f16ToF32(0x4000), 2.0);
  assert.equal(align32(272), 288);
  assert.equal(align32(544), 544);
});

test('FrameFlowPacer schedules 2x and 3x frames without stalls', () => {
  const p2 = new FrameFlowPacer({ factor: 2 });
  assert.equal(p2.onSourceFrame(0, 0).length, 0);
  assert.deepEqual(p2.tick(0), { source: 0 });
  const tasks = p2.onSourceFrame(1, 33.33);
  assert.equal(tasks.length, 1);
  assert.equal(tasks[0].t, 0.5);
  tasks[0].ready = true;
  const f1 = p2.tick(45);
  assert.equal(f1?.mid, true);
  assert.equal(f1?.t, 0.5);
  const f2 = p2.tick(70);
  assert.deepEqual(f2, { source: 1 });

  const p3 = new FrameFlowPacer({ factor: 3 });
  p3.onSourceFrame(0, 0);
  p3.tick(0);
  const t3 = p3.onSourceFrame(1, 50);
  assert.equal(t3.length, 2);
  assert.ok(Math.abs(t3[0].t - 1 / 3) < 1e-6);
  assert.ok(Math.abs(t3[1].t - 2 / 3) < 1e-6);
});

test('runtime loader parses and compiles real exported Std and Lite graphs', () => {
  const preExported = process.env.FRAMEFLOW_TEST_EXPORT_DIR;
  const tmp = preExported || fs.mkdtempSync(path.join(os.tmpdir(), 'ff-node-test-'));
  try {
    if (!preExported) {
      const py = fs.existsSync('.venv/bin/python') ? '.venv/bin/python' : 'python3';
      execFileSync(py, [
        '-c',
        `import sys; sys.path.insert(0,'.'); from frameflow import FrameFlowNet, export_model; ` +
        `export_model(FrameFlowNet('std'), '${tmp}/std', 10); ` +
        `export_model(FrameFlowNet('lite'), '${tmp}/lite', 20)`,
      ]);
    }
    for (const preset of ['std', 'lite']) {
      const dir = path.join(tmp, preset);
      const graph = fs.readFileSync(path.join(dir, 'graph.json'), 'utf8');
      const man = fs.readFileSync(path.join(dir, 'model.json'), 'utf8');
      const bin = fs.readFileSync(path.join(dir, 'model.bin'));
      const model = parseModel(graph, man, bin);
      assert.equal(model.info().formatVersion, 1);
      assert.equal(model.info().name, preset === 'std' ? 'GameFI-Std' : 'GameFI-Lite');

      const plan = compilePlan(model, 480, 288, 1);
      assert.ok(plan.pairSteps.length > 20);
      assert.ok(plan.midSteps.length > 10);
      assert.ok(plan.finalIds.flow_tq && plan.finalIds.vis_tq && plan.finalIds.R_t);

      const gVec = evalFilmMlp(model, ['film.film.0.weight', 'film.film.0.bias', 'film.film.2.weight', 'film.film.2.bias'], 0.5);
      assert.equal(gVec.length, model.meta.cfg.cr * 2);
    }

    if (!preExported) {
      const py = fs.existsSync('.venv/bin/python') ? '.venv/bin/python' : 'python3';
      execFileSync(py, [
        '-c',
        `import sys; sys.path.insert(0,'.'); from frameflow import GameSRNet, export_sr_model; ` +
        `export_sr_model(GameSRNet('sr_std'), '${tmp}/sr_std', 10); ` +
        `export_sr_model(GameSRNet('sr_lite'), '${tmp}/sr_lite', 20)`,
      ]);
      for (const preset of ['sr_std', 'sr_lite']) {
        const dir = path.join(tmp, preset);
        const model = parseModel(
          fs.readFileSync(path.join(dir, 'graph.json'), 'utf8'),
          fs.readFileSync(path.join(dir, 'model.json'), 'utf8'),
          fs.readFileSync(path.join(dir, 'model.bin')),
        );
        const info = model.info();
        assert.equal(info.task, 'sr');
        assert.equal(info.scale, 2);
        const plan = compilePlan(model, 480, 272, 1);
        assert.equal(plan.task, 'sr');
        assert.ok(plan.finalIds.render);
        const outMeta = plan.tensorMeta.get(plan.finalIds.render);
        assert.deepEqual(outMeta, { c: 3, h: 544, w: 960 });
        assert.ok(plan.pairSteps.length >= 6);
        assert.equal(plan.midSteps.length, 0);
        assert.equal(plan.pairSteps[plan.pairSteps.length - 1].op, 'render_sr');
      }
    }
  } finally {
    if (!preExported) {
      fs.rmSync(tmp, { recursive: true, force: true });
    }
  }
});
