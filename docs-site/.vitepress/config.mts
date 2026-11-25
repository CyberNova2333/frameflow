import { defineConfig } from 'vitepress'
import { guideLinks } from './guide-links.mts'

const guideRouteMap: Record<string, string> = {
  'README.md': 'docs/architecture',
  'walkthroughs/one-real-run.md': 'docs/walkthrough',
  'code-map.md': 'docs/code-map',
  'modules/gamefi-and-gamesr-models.md': 'docs/contracts',
  'modules/dual-backend-runtime.md': 'docs/performance',
  'modules/training-and-release-pipeline.md': 'docs/training',
  'glossary.md': 'docs/architecture',
  'references/source-evidence.md': 'docs/architecture',
  'references/quality-review.md': 'docs/architecture',
  'change-log.md': 'docs/decisions',
}

export default defineConfig({
  title: 'FrameFlow',
  description: '跨浏览器 WebGPU/WebGL2 光流插帧（GameFI）与 2× 超分（GameSR）训练推理套件。',
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
          { text: '契约与调优', link: '/docs/contracts' },
          { text: '架构与源码', link: '/docs/architecture' },
          { text: 'API 与组件', link: '/api-reference/' },
        ],
        sidebar: {
          '/docs/': [
            {
              text: '起步与集成训练',
              items: [
                { text: '概览与能力边界', link: '/docs/' },
                { text: '浏览器与模拟器接入', link: '/docs/integration' },
                { text: '训练、蒸馏与双端发版', link: '/docs/training' },
                { text: 'API 与组件参考（Apifox 视图）', link: '/api-reference/' },
                { text: '接入与运行排错 FAQ', link: '/docs/troubleshooting' },
              ],
            },
            {
              text: '契约与性能调优',
              items: [
                { text: 'GameFI / GameSR 契约与格式', link: '/docs/contracts' },
                { text: '双后端计算与节拍器调优', link: '/docs/performance' },
              ],
            },
            {
              text: '原理与源码导览',
              items: [
                { text: '架构全景与交互图谱', link: '/docs/architecture' },
                { text: '一次真实运行走查（源码）', link: '/docs/walkthrough' },
                { text: '代码地图（按职责定位）', link: '/docs/code-map' },
                { text: '工程决策与合规边界', link: '/docs/decisions' },
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
          { text: 'Contracts & Perf', link: '/en/docs/contracts' },
          { text: 'Architecture', link: '/en/docs/architecture' },
          { text: 'API & Components', link: '/en/api-reference/' },
        ],
        sidebar: {
          '/en/docs/': [
            {
              text: 'Getting Started & Training',
              items: [
                { text: 'Overview & Scope Boundaries', link: '/en/docs/' },
                { text: 'Browser & Emulator Integration', link: '/en/docs/integration' },
                { text: 'Training, Distillation & Releases', link: '/en/docs/training' },
                { text: 'API & Component Reference', link: '/en/api-reference/' },
                { text: 'Integration & Runtime FAQ', link: '/en/docs/troubleshooting' },
              ],
            },
            {
              text: 'Contracts & Performance',
              items: [
                { text: 'GameFI / GameSR Contracts & Format', link: '/en/docs/contracts' },
                { text: 'Dual-Backend & Pacer Tuning', link: '/en/docs/performance' },
              ],
            },
            {
              text: 'Architecture & Decisions',
              items: [
                { text: 'Architecture & Interactive Diagram', link: '/en/docs/architecture' },
                { text: 'End-to-End Walkthrough', link: '/en/docs/walkthrough' },
                { text: 'Code Map by Responsibility', link: '/en/docs/code-map' },
                { text: 'Decisions & Compliance ADRs', link: '/en/docs/decisions' },
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
