import { describe, expect, it } from 'vitest'
import { pickVulkanAsset } from '../src/core/runtimes/llamacpp'

const a = (name: string) => ({ name, browser_download_url: `https://x/${name}`, size: 1 })

describe('pickVulkanAsset', () => {
  it('skips asset-less "latest" style releases and picks the newest win-vulkan-x64 zip', () => {
    const picked = pickVulkanAsset([
      { tag_name: 'v0.5.0', draft: false, assets: [a('nightly-tag.txt')] },
      { tag_name: 'b2', draft: true, assets: [a('llama-b2-bin-win-vulkan-x64.zip')] },
      { tag_name: 'b1', draft: false, assets: [a('llama-b1-bin-win-cpu-x64.zip'), a('llama-b1-bin-win-vulkan-x64.zip')] }
    ])
    expect(picked).toMatchObject({ tag: 'b1', name: 'llama-b1-bin-win-vulkan-x64.zip' })
  })
  it('returns null when nothing matches', () => {
    expect(pickVulkanAsset([{ tag_name: 'b1', draft: false, assets: [a('llama-b1-bin-win-cuda-12.4-x64.zip')] }])).toBeNull()
  })
})
