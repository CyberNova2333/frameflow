// SPDX-License-Identifier: AGPL-3.0-only
// Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
// Headless Edge 157 runner for verifying GameSR-2x WebGPU + WebGL2 parity & latency.
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
import { createFrameFlow, loadModelFromUrl } from '/runtime/index.js';

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
  const gl = engine.gl;
  const px = new Uint8Array(4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, engine.fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, session.outputTexture(0), 0);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
}

async function benchSr(engine, model, w, h, bytes) {
  const s = engine.createSession(model, { width: w, height: h });
  s.pushFrame(0, bytes);
  for (let i = 0; i < 8; i++) { s.upscale(0, 0); }
  await syncGpu(engine, s);
  const times = [];
  for (let i = 0; i < 16; i++) {
    const t0 = performance.now();
    s.upscale(0, 0);
    await syncGpu(engine, s);
    times.push(performance.now() - t0);
  }
  s.destroy();
  return med(times);
}

window.runVerify = async (b64_480, b64_960, jobs) => {
  const bytes480 = await decodePng(b64_480, 480, 272);
  const bytes960 = await decodePng(b64_960, 960, 544);
  const engines = {};
  const models = {};
  const results = [];

  for (const job of jobs) {
    if (!engines[job.backend]) engines[job.backend] = await createFrameFlow({ backend: job.backend });
    const engine = engines[job.backend];
    const mKey = job.backend + ':' + job.preset;
    if (!models[mKey]) models[mKey] = await loadModelFromUrl(engine, '/models/' + job.preset);
    const model = models[mKey];

    const s = engine.createSession(model, { width: 480, height: 272 });
    s.pushFrame(0, bytes480);
    s.upscale(0, 0);
    await syncGpu(engine, s);
    const outRgba = await s.readRgba(0);
    const pngB64 = await rgbaToPngB64(outRgba, 960, 544);
    s.destroy();

    const ms480 = await benchSr(engine, model, 480, 272, bytes480);
    const ms960 = await benchSr(engine, model, 960, 544, bytes960);

    results.push({
      backend: job.backend,
      preset: job.preset,
      engineInfo: engine.info(),
      ms480,
      ms960,
      pngB64,
    });
  }
  for (const e of Object.values(engines)) e.destroy();
  return { results };
};
</script></body></html>`;

const server = http.createServer(async (req, res) => {
  const url = req.url.split('?')[0];
  if (url === '/') {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    return res.end(html);
  }
  if (url === '/favicon.ico') { res.writeHead(204); return res.end(); }
  let filePath = null;
  if (url.startsWith('/runtime/')) filePath = path.join(ffRoot, url);
  else if (url.startsWith('/models/')) filePath = path.join(path.dirname(jobsFile), url.slice('/models/'.length));
  if (!filePath) { res.writeHead(404); return res.end('not found'); }
  try {
    const data = await fs.readFile(filePath);
    res.writeHead(200, { 'Content-Type': mime[path.extname(filePath)] || 'application/octet-stream' });
    res.end(data);
  } catch {
    res.writeHead(404); res.end('missing');
  }
});

await new Promise(r => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

const browser = await chromium.launch({
  channel: 'msedge',
  headless: true,
  args: ['--enable-features=Vulkan', '--use-angle=gl'],
});

try {
  const page = await browser.newPage();
  page.on('console', m => { if (m.type() === 'error') console.error('[browser error]', m.text()); });
  await page.goto(`http://127.0.0.1:${port}/`);
  const b480 = (await fs.readFile(spec.frame480)).toString('base64');
  const b960 = (await fs.readFile(spec.frame960)).toString('base64');
  const out = await page.evaluate(async ([a, b, jobs]) => window.runVerify(a, b, jobs), [b480, b960, spec.jobs]);
  for (let i = 0; i < spec.jobs.length; i++) {
    const r = out.results[i];
    await fs.writeFile(spec.jobs[i].out_png, Buffer.from(r.pngB64, 'base64'));
    delete r.pngB64;
  }
  await fs.writeFile(path.join(path.dirname(jobsFile), 'report.json'), JSON.stringify(out, null, 2));
} finally {
  await browser.close();
  server.close();
}
