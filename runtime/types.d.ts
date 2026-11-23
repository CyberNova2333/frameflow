// SPDX-License-Identifier: AGPL-3.0-only
// Additional Terms: see LICENSE-ADDITIONAL_TERMS-zh-CN.md (v1.6.7-r4)
export type BackendKind = 'webgpu' | 'webgl2';

export interface FrameFlowCreateOptions {
  backend?: 'auto' | BackendKind;
  device?: GPUDevice;
  gl?: WebGL2RenderingContext;
  canvas?: HTMLCanvasElement | OffscreenCanvas;
  precision?: 'auto' | 'f16' | 'f32';
  powerPreference?: 'high-performance' | 'low-power';
}

export interface FrameFlowSessionOptions {
  width: number;
  height: number;
  internal?: 'frame' | 'auto' | [number, number];
  k?: number;
  stillGuard?: boolean;
  sampling?: 'model' | 'bilinear' | 'bicubic';
  useFilm?: boolean;
  useRefine?: boolean;
  flipSrcY?: boolean;
}

export interface FrameFlowModelInfo {
  name: string;
  formatVersion: number;
  trainStep: number;
  params: number;
  tensors: number;
  renderSampling: string;
  defaultK: number;
  cfg: Record<string, number> | null;
}

export interface FrameFlowModel {
  info(): FrameFlowModelInfo;
  destroy(): void;
}

export interface FrameFlowSession {
  readonly width: number;
  readonly height: number;
  readonly workW: number;
  readonly workH: number;
  addInput(): number;
  addOutput(): number;
  inputTexture(slot: number): GPUTexture | WebGLTexture;
  outputTexture(slot?: number): GPUTexture | WebGLTexture;
  setExternalInputTexture(slot: number, texture: GPUTexture | WebGLTexture): void;
  pushFrame(slot: number, source: Uint8Array | Uint8ClampedArray | HTMLCanvasElement | OffscreenCanvas | ImageBitmap | HTMLVideoElement): void;
  pair(slotA: number, slotB: number): void;
  mid(t?: number, outSlot?: number): void;
  readRgba(outSlot?: number): Promise<Uint8Array>;
  finish(): Promise<void> | void;
  destroy(): void;
}

export interface FrameFlowEngine {
  readonly backend: BackendKind;
  info(): Record<string, unknown>;
  loadModel(graph: string | object, manifest: string | object, bin: ArrayBuffer | Uint8Array): FrameFlowModel;
  createSession(model: FrameFlowModel, options: FrameFlowSessionOptions): FrameFlowSession;
  destroy(): void;
}

export interface FrameFlowCapabilities {
  supported: boolean;
  recommended: BackendKind | null;
  webgpu: { supported: boolean; reason: string | null; vendor?: string; architecture?: string; f16?: boolean; timestamps?: boolean };
  webgl2: { supported: boolean; reason: string | null; colorBufferFloat?: boolean; colorBufferHalfFloat?: boolean; maxDrawBuffers?: number; renderer?: string };
}

export function probeCapabilities(options?: { powerPreference?: 'high-performance' | 'low-power'; gl?: WebGL2RenderingContext; canvas?: HTMLCanvasElement | OffscreenCanvas }): Promise<FrameFlowCapabilities>;
export function createFrameFlow(options?: FrameFlowCreateOptions): Promise<FrameFlowEngine>;
export function loadModelFromUrl(engine: FrameFlowEngine, baseUrl: string, stem?: string): Promise<FrameFlowModel>;

export class FrameFlowPacer {
  constructor(options?: { factor?: 2 | 3; maxBufferMs?: number });
  reset(): void;
  onSourceFrame(slot: number, nowMs: number): Array<{ kind: 'mid'; t: number; outSlot: number; dueMs: number; ready: boolean }>;
  tick(nowMs: number): { source: number } | { output: number; mid: true; t: number } | null;
  report(): Record<string, unknown>;
}
