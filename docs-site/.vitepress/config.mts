import { defineConfig } from 'vitepress'
import { guideLinks } from './guide-links.mts'

const guideRouteMap: Record<string, string> = {
  'README.md': 'docs/architecture',
  'walkthroughs/one-real-run.md': 'docs/walkthrough',
  'code-map.md': 'docs/code-map',
  'modules/dual-backend-runtime.md': 'docs/backends',
  'modules/gamefi-and-gamesr-models.md': 'docs/models',
  'modules/training-and-release-pipeline.md': 'docs/training',
  'glossary.md': 'docs/architecture',
  'references/source-evidence.md': 'docs/architecture',
  'references/quality-review.md': 'docs/architecture',
  'change-log.md': 'docs/architecture',
}

export default defineConfig({
  title: 'FrameFlow',
  description: '跨浏览器光流插帧（GameFI）与 2× 超分（GameSR）训练、导出及 WebGPU/WebGL2 运行时套件。',
  cleanUrls: true,
  lastUpdated: false,
  markdown: {
    config(md) {
      md.use(guideLinks(guideRouteMap))
    },
  },
  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN',
      themeConfig: {
        nav: [
          { text: '首页', link: '/' },
          { text: '起步与接入', link: '/docs/' },
          { text: '模型与训练', link: '/docs/models' },
          { text: '架构与源码', link: '/docs/architecture' },
          { text: 'API 与格式', link: '/api-reference/' },
        ],
        sidebar: {
          '/docs/': [
            {
              text: '起步与模拟器集成',
              items: [
                { text: '概览与能力边界', link: '/docs/' },
                { text: '浏览器与模拟器接入', link: '/docs/integration' },
                { text: '训练、蒸馏与双端发版', link: '/docs/training' },
                { text: 'API 与模型格式参考', link: '/api-reference/' },
              ],
            },
            {
              text: '模型结构与双后端',
              items: [
                { text: 'GameFI 与 GameSR 模型及格式', link: '/docs/models' },
                { text: 'WebGPU 与 WebGL2 双后端运行时', link: '/docs/backends' },
              ],
            },
            {
              text: '原理与源码导览',
              items: [
                { text: '架构全景与源码指南总览', link: '/docs/architecture' },
                { text: '一次真实运行走查（源码）', link: '/docs/walkthrough' },
                { text: '代码地图（按职责定位）', link: '/docs/code-map' },
              ],
            },
          ],
        },
        outline: { label: '本页目录', level: [2, 3] },
        docFooter: { prev: '上一页', next: '下一页' },
      },
    },
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/en/' },
          { text: 'Getting Started', link: '/en/docs/' },
          { text: 'Models & Training', link: '/en/docs/models' },
          { text: 'Architecture', link: '/en/docs/architecture' },
          { text: 'API & Format', link: '/en/api-reference/' },
        ],
        sidebar: {
          '/en/docs/': [
            {
              text: 'Getting Started & Integration',
              items: [
                { text: 'Overview & Scope Boundaries', link: '/en/docs/' },
                { text: 'Browser & Emulator Integration', link: '/en/docs/integration' },
                { text: 'Training, Distillation & Releases', link: '/en/docs/training' },
                { text: 'API & Format Reference', link: '/en/api-reference/' },
              ],
            },
            {
              text: 'Models & Dual Backends',
              items: [
                { text: 'GameFI & GameSR Specifications', link: '/en/docs/models' },
                { text: 'WebGPU & WebGL2 Runtime', link: '/en/docs/backends' },
              ],
            },
            {
              text: 'Architecture & Source Guide',
              items: [
                { text: 'Architecture & Source Guide Index', link: '/en/docs/architecture' },
                { text: 'End-to-End Walkthrough', link: '/en/docs/walkthrough' },
                { text: 'Code Map by Responsibility', link: '/en/docs/code-map' },
              ],
            },
          ],
        },
        outline: { label: 'On this page', level: [2, 3] },
      },
    },
  },
  themeConfig: {
    search: { provider: 'local' },
    socialLinks: [{ icon: 'github', link: 'https://github.com/CyberNova2333/frameflow' }],
  },
})
