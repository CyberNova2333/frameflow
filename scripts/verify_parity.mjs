// SPDX-License-Identifier: AGPL-3.0-only
// Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
// Headless Edge 157 WebGPU runner for Milestone 1 PyTorch <-> WebGPU parity verification.
import http from 'node:http';
import fs from 'node:fs/promises';
import path from 'node:path';
import { createRequire } from 'node:module';

const jobsFile = path.resolve(process.argv[2]);
const spec = JSON.parse(await fs.readFile(jobsFile, 'utf8'));
const pvRoot = path.resolve(path.dirname(jobsFile), '../../../pvsspw');
const fhPkgDir = path.join(pvRoot, '.cache/flyhigh-engine/examples/web/pkg');
const requireFromPv = createRequire(path.join(pvRoot, 'package.json'));
const { chromium } = requireFromPv('playwright-core');

const mime = { '.html': 'text/html', '.js': 'application/javascript', '.wasm': 'application/wasm', '.json': 'application/json', '.bin': 'application/octet-stream' };

const html = `<!doctype html><html><body><script type="module">
import init, * as fh from '/pkg/flyhigh_web.js';
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
window.runParity = async (aB64, bB64, jobs) => {
  await init();
  const engine = await fh.FhEngine.create({ precision: 'auto', autotune: false });
  const device = engine.device();
  const aRgba = await decodePng(aB64, 960, 544);
  const bRgba = await decodePng(bB64, 960, 544);

  const modelCache = new Map();
  const results = [];
  for (const job of jobs) {
    let model = modelCache.get(job.preset);
    if (!model) {
      const [graph, man, bin] = await Promise.all([
        fetch('/models/' + job.preset + '/graph.json').then(r => r.text()),
        fetch('/models/' + job.preset + '/model.json').then(r => r.text()),
        fetch('/models/' + job.preset + '/model.bin').then(r => r.arrayBuffer()),
      ]);
      model = engine.loadModel(graph, man, new Uint8Array(bin));
      modelCache.set(job.preset, model);
    }
    // Exact parity session: repair=0, staticProtection=false, fastRender=false, linearT=false, internal='frame'
    // Note: decisive=0 is achieved when repair=0 and render='model'? Wait: check if flyhigh-engine enables decisive by default!
    const s = engine.session(model, {
      width: 960,
      height: 544,
      internal: 'frame',
      k: job.k,
      repair: 0,
      render: job.sampling,
      staticProtection: false,
      fastRender: false,
      linearT: false,
    });
    const sa = s.addInput(), sb = s.addInput(), so = s.addOutput();
    s.writeRgba(sa, aRgba);
    s.writeRgba(sb, bRgba);
    s.warmUp();
    await device.queue.onSubmittedWorkDone();
    for (let i = 0; i < 4; i++) { s.pair(sa, sb); s.mid(job.t, so); }
    await device.queue.onSubmittedWorkDone();
    const pairFullMs = await s.pairTimed(sa, sb);
    const midFullMs = await s.midTimed(job.t, so);
    const outRgba = new Uint8Array(await s.readRgba(so));
    const pngB64 = await rgbaToPngB64(outRgba, 960, 544);
    s.free();

    // Quick 480x288 internal speed probe (fastRender+linearT) on t050
    let pair480Ms = null, mid480Ms = null;
    if (job.t_name === 't050') {
      const s480 = engine.session(model, {
        width: 960, height: 544, internal: [480, 288], k: job.k, repair: 0, fastRender: true, linearT: true,
      });
      const ia = s480.addInput(), ib = s480.addInput(), io = s480.addOutput();
      s480.writeRgba(ia, aRgba); s480.writeRgba(ib, bRgba);
      s480.warmUp(); await device.queue.onSubmittedWorkDone();
      for (let i = 0; i < 6; i++) { s480.pair(ia, ib); s480.mid(0.5, io); }
      await device.queue.onSubmittedWorkDone();
      const ps = [], ms = [];
      for (let i = 0; i < 12; i++) {
        ps.push(await s480.pairTimed(ia, ib));
        await device.queue.onSubmittedWorkDone();
        ms.push(await s480.midTimed(0.5, io));
        await device.queue.onSubmittedWorkDone();
      }
      const med = (arr) => { const x = arr.filter(v => typeof v === 'number').sort((u,v)=>u-v); return x.length ? Number(x[Math.floor(x.length/2)].toFixed(3)) : null; };
      pair480Ms = med(ps); mid480Ms = med(ms);
      s480.free();
    }

    results.push({
      preset: job.preset,
      t_name: job.t_name,
      pairFullMs: pairFullMs != null ? Number(pairFullMs.toFixed(3)) : null,
      midFullMs: midFullMs != null ? Number(midFullMs.toFixed(3)) : null,
      pair480Ms,
      mid480Ms,
      pngB64,
    });
  }
  return { engineInfo: engine.info(), results };
};
</script></body></html>`;

const parityBase = path.dirname(jobsFile);
const server = http.createServer(async (req, res) => {
  const u = new URL(req.url, 'http://127.0.0.1').pathname;
  const h = { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp', 'Cross-Origin-Resource-Policy': 'same-origin' };
  if (u === '/') { res.writeHead(200, { ...h, 'Content-Type': 'text/html' }); return res.end(html); }
  try {
    let file;
    if (u.startsWith('/pkg/')) file = path.join(fhPkgDir, u.slice('/pkg/'.length));
    else if (u.startsWith('/models/')) file = path.join(parityBase, u.slice('/models/'.length));
    else { res.writeHead(404); return res.end('not found'); }
    const data = await fs.readFile(file);
    res.writeHead(200, { ...h, 'Content-Type': mime[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  } catch { res.writeHead(404); res.end('not found'); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

const aB64 = (await fs.readFile(spec.frameA)).toString('base64');
const bB64 = (await fs.readFile(spec.frameB)).toString('base64');

const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--enable-features=Vulkan', '--use-angle=gl'] });
try {
  const page = await browser.newPage();
  page.setDefaultTimeout(180000);
  page.on('console', m => { if (m.type() === 'error') console.error('[page:err]', m.text()); });
  page.on('pageerror', e => console.error('[pageerror]', e));
  await page.goto(`http://127.0.0.1:${port}/`);
  const rep = await page.evaluate(([a, b, jobs]) => window.runParity(a, b, jobs), [aB64, bB64, spec.jobs]);
  for (let i = 0; i < spec.jobs.length; i++) {
    await fs.writeFile(spec.jobs[i].gpu_out, Buffer.from(rep.results[i].pngB64, 'base64'));
    delete rep.results[i].pngB64;
  }
  await fs.writeFile(path.join(parityBase, 'webgpu_report.json'), JSON.stringify(rep, null, 2) + '\n');
  console.log('[webgpu] Engine:', rep.engineInfo);
} finally {
  await browser.close();
  server.close();
}
