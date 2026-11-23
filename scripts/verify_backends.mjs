// SPDX-License-Identifier: AGPL-3.0-only
// Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
// Headless Edge 157 runner for verifying BOTH FrameFlow WebGPU and WebGL2 backends.
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';

const jobsFile = path.resolve(process.argv[2]);
const spec = JSON.parse(await fs.readFile(jobsFile, 'utf8'));
const ffRoot = path.resolve(path.dirname(jobsFile), '../..');
const pvRoot = path.resolve(ffRoot, '../pvsspw');
const requireFromPv = createRequire(path.join(pvRoot, 'package.json'));
const { chromium } = requireFromPv('playwright-core');

const mime = { '.html': 'text/html', '.js': 'application/javascript', '.json': 'application/json', '.bin': 'application/octet-stream' };

const html = `<!doctype html><html><body><script type="module">
import { createFrameFlow, loadModelFromUrl, probeCapabilities } from '/runtime/index.js';

async function decodePng(b64, w, h) {
  const blob = await fetch('data:image/png;base64,' + b64).then(r => r.blob());
  const bmp = await createImageBitmap(blob);
  const cv = new OffscreenCanvas(w, h);
  const ctx = cv.getContext('2d', { alpha: false });
  ctx.drawImage(bmp, 0, 0, w, h);
  bmp.close();
  const d = new Uint8Array(ctx.getImageData(0, 0, w, h).data.buffer);
  for (let i = 3; i < d.length; i += 4) d[i] = 255;
  return d;
}

async function rgbaToPngB64(rgba, w, h) {
  const cv = new OffscreenCanvas(w, h);
  const ctx = cv.getContext('2d');
  const clamped = new Uint8ClampedArray(rgba);
  for (let i = 3; i < clamped.length; i += 4) clamped[i] = 255;
  ctx.putImageData(new ImageData(clamped, w, h), 0, 0);
  const blob = await cv.convertToBlob({ type: 'image/png' });
  const bytes = new Uint8Array(await blob.arrayBuffer());
  let bin = '';
  for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return btoa(bin);
}

const med = (arr) => { const s = [...arr].sort((a, b) => a - b); return Number(s[Math.floor(s.length / 2)].toFixed(2)); };

async function syncGpu(engine, session) {
  if (engine.backend === 'webgpu') {
    await session.finish();
    return;
  }
  // In Chromium ANGLE, gl.finish() alone does not always drain the GPU command queue;
  // a 1x1 readPixels on the output FBO forces a true GPU completion fence.
  const gl = engine.gl;
  const px = new Uint8Array(4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, engine.fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, session.outputTexture(0), 0);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

async function benchSession(engine, model, w, h, internal, k, aBytes, bBytes) {
  const s = engine.createSession(model, { width: w, height: h, internal, k, stillGuard: true, sampling: 'bilinear' });
  s.pushFrame(0, aBytes);
  s.pushFrame(1, bBytes);
  for (let i = 0; i < 4; i++) { s.pair(0, 1); s.mid(0.5, 0); }
  await syncGpu(engine, s);

  const totMs = [], pMs = [], mMs = [];
  const iters = 10;
  for (let i = 0; i < iters; i++) {
    const t0 = performance.now();
    s.pair(0, 1);
    if (engine.backend === 'webgpu') {
      await syncGpu(engine, s);
      const t1 = performance.now();
      pMs.push(t1 - t0);
      s.mid(0.5, 0);
      await syncGpu(engine, s);
      const t2 = performance.now();
      mMs.push(t2 - t1);
      totMs.push(t2 - t0);
    } else {
      // Measure full pair+mid with a single end-of-frame GPU fence (avoids double readPixels stall)
      s.mid(0.5, 0);
      await syncGpu(engine, s);
      const dt = performance.now() - t0;
      totMs.push(dt);
    }
  }
  s.destroy();
  const tm = med(totMs);
  return {
    pairMs: pMs.length ? med(pMs) : '-',
    midMs: mMs.length ? med(mMs) : '-',
    totalMs: tm,
  };
}

window.runVerify = async (aB64, bB64, jobs) => {
  const capabilities = await probeCapabilities();
  const a960 = await decodePng(aB64, 960, 544);
  const b960 = await decodePng(bB64, 960, 544);
  const a480 = await decodePng(aB64, 480, 272);
  const b480 = await decodePng(bB64, 480, 272);

  const engines = {};
  const models = {};
  const results = [];

  for (const job of jobs) {
    if (!engines[job.backend]) {
      engines[job.backend] = await createFrameFlow({ backend: job.backend });
    }
    const engine = engines[job.backend];
    const mKey = job.backend + ':' + job.preset;
    if (!models[mKey]) {
      models[mKey] = await loadModelFromUrl(engine, '/models/' + job.preset);
    }
    const model = models[mKey];

    // Exact parity session (internal: 'frame', full 960x544)
    const s = engine.createSession(model, {
      width: 960,
      height: 544,
      internal: 'frame',
      k: job.k,
      stillGuard: true,
      sampling: job.sampling,
    });
    s.pushFrame(0, a960);
    s.pushFrame(1, b960);
    s.pair(0, 1);
    s.mid(job.t, 0);
    await s.finish();
    const outRgba = await s.readRgba(0);
    const pngB64 = await rgbaToPngB64(outRgba, 960, 544);
    s.destroy();

    // Also measure practical speeds once per (backend, preset) on t050
    let bench480 = null, benchPsp = null;
    if (job.t_name === 't050') {
      bench480 = await benchSession(engine, model, 960, 544, [480, 288], job.k, a960, b960);
      if (job.preset === 'lite') {
        benchPsp = await benchSession(engine, model, 480, 272, 'frame', job.k, a480, b480);
      }
    }

    results.push({
      backend: job.backend,
      preset: job.preset,
      t_name: job.t_name,
      engineInfo: engine.info(),
      bench480,
      benchPsp,
      pngB64,
    });
  }

  for (const e of Object.values(engines)) e.destroy();
  return { capabilities, results };
};
</script></body></html>`;

const verifyDir = path.dirname(jobsFile);
const server = http.createServer(async (req, res) => {
  const u = new URL(req.url, 'http://127.0.0.1').pathname;
  const h = {
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cross-Origin-Resource-Policy': 'same-origin',
  };
  if (u === '/') { res.writeHead(200, { ...h, 'Content-Type': 'text/html' }); return res.end(html); }
  try {
    let file;
    if (u.startsWith('/runtime/')) file = path.join(ffRoot, u);
    else if (u.startsWith('/models/')) file = path.join(verifyDir, u.slice('/models/'.length));
    else { res.writeHead(404); return res.end('not found'); }
    const data = await fs.readFile(file);
    res.writeHead(200, { ...h, 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  } catch {
    res.writeHead(404);
    res.end('not found');
  }
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

const aB64 = (await fs.readFile(spec.frameA)).toString('base64');
const bB64 = (await fs.readFile(spec.frameB)).toString('base64');

const browser = await chromium.launch({
  channel: 'msedge',
  headless: true,
  args: ['--enable-features=Vulkan', '--use-angle=gl'],
});
try {
  const page = await browser.newPage();
  page.setDefaultTimeout(180000);
  page.on('console', (m) => { if (m.type() === 'error') console.error('[page:err]', m.text()); });
  page.on('pageerror', (e) => console.error('[pageerror]', e));
  await page.goto(`http://127.0.0.1:${port}/`);
  const rep = await page.evaluate(([a, b, jobs]) => window.runVerify(a, b, jobs), [aB64, bB64, spec.jobs]);
  for (let i = 0; i < spec.jobs.length; i++) {
    await fs.writeFile(spec.jobs[i].out_png, Buffer.from(rep.results[i].pngB64, 'base64'));
    delete rep.results[i].pngB64;
  }
  await fs.writeFile(path.join(verifyDir, 'report.json'), JSON.stringify(rep, null, 2) + '\n');
} finally {
  await browser.close();
  server.close();
}
