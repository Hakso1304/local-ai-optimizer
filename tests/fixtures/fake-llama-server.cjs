// Stand-in for llama-server.exe in process tests. FAKE_MODE: ok | wrong | die-oom
const http = require('node:http')
const args = process.argv.slice(2)
const arg = (k) => args[args.indexOf(k) + 1]
const mode = process.env.FAKE_MODE

if (mode === 'die-oom') {
  console.error('ggml_vulkan: Device memory allocation of size 9000000000 failed.')
  console.error('ggml_vulkan: vk::Device::allocateMemory: ErrorOutOfDeviceMemory')
  process.exit(3)
}

console.error('load_tensors: offloaded 25/25 layers to GPU')
console.error('load_tensors:      Vulkan0 model buffer size =   500.79 MiB')
const modelPath = mode === 'wrong' ? 'C:\\somewhere\\other.gguf' : arg('-m')
http
  .createServer((req, res) => {
    if (req.url === '/health') return mode === 'slow' ? res.writeHead(503).end('{"error":"loading model"}') : res.end(JSON.stringify({ status: 'ok' }))
    if (req.url === '/props') {
      const nCtx = mode === 'drift' ? Number(arg('-c')) / 2 : Number(arg('-c'))
      return res.end(JSON.stringify({ model_path: modelPath, default_generation_settings: { n_ctx: nCtx } }))
    }
    if (req.url === '/completion' && mode === 'http400') {
      return res.writeHead(400).end(JSON.stringify({ error: { code: 400, type: 'exceed_context_size_error', message: 'request (40000 tokens) exceeds the available context size (32768 tokens)' } }))
    }
    if (req.url === '/completion') {
      res.writeHead(200, { 'content-type': 'text/event-stream' })
      res.write('data: {"content":"Hi","stop":false}\n\n')
      res.end('data: {"content":"","stop":true,"stop_type":"limit","timings":{"prompt_n":3,"prompt_ms":1.5,"prompt_per_second":2000,"predicted_n":1,"predicted_ms":2,"predicted_per_second":500}}\n\n')
      return
    }
    if (req.url === '/apply-template') {
      let body = ''
      req.on('data', (d) => (body += d))
      req.on('end', () => { const b = JSON.parse(body); res.end(JSON.stringify({ prompt: `[${JSON.stringify(b.chat_template_kwargs)}] ${b.messages.map((m) => m.content).join('|')}` })) })
      return
    }
    if (req.url === '/tokenize') {
      let body = ''
      req.on('data', (d) => (body += d))
      req.on('end', () => res.end(JSON.stringify({ tokens: JSON.parse(body).content.split(/\s+/).filter(Boolean).map((_, i) => i) })))
      return
    }
    res.writeHead(404).end()
  })
  .listen(Number(arg('--port')), '127.0.0.1')
