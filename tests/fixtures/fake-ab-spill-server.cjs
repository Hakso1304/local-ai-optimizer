// JSON HTTP stand-in for scripts/ab-spill.ts launch tests. Runs under process.execPath only.
const http = require('node:http')

const args = process.argv.slice(2)
const arg = (name) => args[args.indexOf(name) + 1]
const mode = arg('--fake-mode')

if (mode === 'streams') {
  process.stdout.write('load_tensors: offloaded 25/25 layers to GPU\nVulkan0 model buffer size = 800.50 MiB\n')
  process.stderr.write('Vulkan0 KV buffer size = 256.25 MiB\nVulkan0 compute buffer size = 64.00 MiB\n')
} else {
  process.stdout.write('model load complete; no buffer declaration\n')
  process.stderr.write('diagnostic without buffer numbers\n')
}

http.createServer((req, res) => {
  res.setHeader('content-type', 'application/json')
  if (req.url === '/health') return res.end(JSON.stringify({ status: 'ok' }))
  if (req.url === '/tokenize') return res.end(JSON.stringify({ tokens: Array(16).fill(1) }))
  if (req.url === '/completion') {
    req.resume()
    return res.end(JSON.stringify(mode === 'no-timings' ? { content: 'done' } : {
      content: 'done', timings: { prompt_n: 16, prompt_ms: 12.5, prompt_per_second: 1280, predicted_per_second: 42 }
    }))
  }
  res.statusCode = 404
  res.end('{}')
}).listen(Number(arg('--port')), '127.0.0.1')
