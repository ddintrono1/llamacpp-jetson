# Docker-containerized llama.cpp for Jetson Orin

Run [llama.cpp](https://github.com/ggml-org/llama.cpp) on an NVIDIA Jetson AGX Orin inside Docker, serve models over an OpenAI-compatible HTTP API, and benchmark them for speed and energy efficiency.

- **Image:** a two-stage build of llama.cpp `v0.6.0` with the CUDA backend, compiled only for Orin's GPU (`sm_87`). Target platform: JetPack 7.2 (L4T R39.2, Ubuntu 24.04, CUDA 13.2). See [llamacpp.Dockerfile](llamacpp.Dockerfile).
- **Serving:** `llama-server` with full GPU offload, flash attention and API-key authentication, with optional speculative decoding through a draft model.
- **Benchmarking:** [bench.py](bench.py) measures generation speed, draft-token acceptance and **tokens per joule** using the board's power sensors.

All commands are `make` targets defined in the [makefile](makefile).

## Usage

### Requirements

- Jetson AGX Orin with JetPack 7.2
- Docker with the NVIDIA container runtime (`--gpus all`)
- On the host: `curl` and `jq` (for `make infer`), and `python3` (for `make bench`; ships with JetPack, standard library only)

### Setup

1. Put GGUF model files in `models/`. The folder is mounted read-only into the container at `/models`.
2. Create a `.env` file with the API key the server requires. The makefile loads it automatically.

   ```
   LLAMA_API_KEY=<your key>
   ```

3. Build the image:

   ```bash
   make build
   ```

### Serving

```bash
# Any model, on port 8080
make serve MODEL=/models/<model>.gguf

# Qwen3.8-27B with the DFlash2 draft model (speculative decoding)
make serve-qwen3.8_27B_sd

# Stop the server
make clear
```

### Querying

```bash
make infer PROMPT="your question"
```

This sends a chat request to the running server and prints the JSON response. Any OpenAI-compatible client also works against `http://<jetson-ip>:8080/v1`, with the API key as a Bearer token.

### Shell

```bash
make shell
```

This opens a shell in the running container, or in a new one if no server is running.

### Benchmarking

Start a server, then:

```bash
make bench
```

This runs every prompt in [bench_prompts.txt](resources/bench_prompts.txt) 5 times, prints each run and a summary, and writes a CSV to `results/<date>_<time>.csv`. Settings can be overridden on the command line:

| Variable | Default | Meaning |
|---|---|---|
| `BENCH_PROMPTS` | `resources/bench_prompts.txt` | Prompts file, one prompt per line; blank lines and `#` comments are skipped |
| `BENCH_RUNS` | `5` | Runs per prompt |
| `BENCH_N_PREDICT` | `128` | Tokens generated per run |
| `POWER_INTERVAL` | `100` | Power sampling period (ms) |
| `POWER_RAILS` | `VDD_GPU_SOC\|VDD_CPU_CV\|VIN_SYS_5V0` | Power rails summed as module power |

For example: `make bench BENCH_RUNS=10 BENCH_N_PREDICT=256`. Run `python3 bench.py --help` for all options.

## Benchmarking methodology

### Requests

Each run sends one request to the `/completion` endpoint:

- **Fixed length:** generation stops after `BENCH_N_PREDICT` tokens, so all runs do the same amount of work.
- **`temperature: 0`:** greedy decoding makes the output repeatable. Runs of the same prompt produce the same text, so run-to-run differences reflect speed, not different output.
- **`cache_prompt: false`:** the prompt is processed again on every run instead of being reused from the cache, so later runs are not artificially faster.
- **Raw prompt:** `/completion` takes the prompt as plain text and does not apply the model's chat template. The model simply continues the text.

### Speed and acceptance

These come from the server's `timings`:

- **Prompt processing (`pp_tok_s`)** and **generation (`tg_tok_s`)** in tokens per second.
- **Acceptance rate** = `draft_n_accepted / draft_n`: the share of tokens proposed by the draft model that the target model accepted. A higher rate means a larger speedup.

Acceptance depends on content. Predictable text such as code is accepted more often than free prose, so the prompts file should reflect the real workload. The example file has one code prompt and one prose prompt.

### Energy and tokens per joule

A power rail is a supply line that feeds one group of components at a set voltage. The AGX Orin has INA3221 sensors that measure the voltage and current of four rails:

| Rail | Bus voltage | Covers | Summed |
|---|---|---|---|
| `VDD_GPU_SOC` | 12 V | GPU and SoC | yes |
| `VDD_CPU_CV` | 12 V | CPU and computer-vision accelerators | yes |
| `VIN_SYS_5V0` | 5 V | 5 V system supply of the module (memory, I/O, ...) | yes |
| `VDDQ_VDD2_1V8AO` | 5 V | Memory supply | no |

`VDDQ_VDD2_1V8AO` is left out because it is measured on the same 5 V bus as `VIN_SYS_5V0`, which suggests its power is already included there; adding it would likely count memory power twice. This is inferred from the bus voltages, not confirmed by NVIDIA documentation. jtop's "All" figure sums all four rails, so it reads slightly higher than `power_w`.

While each request runs, a background thread reads the sensors directly from sysfs (`/sys/bus/i2c/drivers/ina3221/`) every `POWER_INTERVAL` ms. These are the same sensors `tegrastats` and jtop report. The sensors update every 1 ms, so a 100 ms period is well within their resolution. For each sample the script computes each rail's power as voltage x current and sums the selected rails. Then:

```
power_w   = mean over samples of (sum of rails)
energy_j  = power_w x (prompt time + generation time)
tok_per_j = generated tokens / energy_j
```

Request times come from the server's `timings`, not from the sampling window, so the sampler's start and stop do not count.

### Limitations

- **Idle power is included.** The module draws about 7 W at rest, which is part of every measurement. `tok_per_j` is the energy cost of a request on this board, not the extra energy the model alone uses.
- **Prompt processing is included.** Energy covers the whole request, but only generated tokens are counted. With short prompts the difference is small; long prompts lower `tok_per_j`.
- **Module power only.** The rails cover the Orin module, not the carrier board, fan or power supply losses. A meter at the wall would read higher.
- **Rail names are board-specific.** Other Jetson modules have different rails. If `POWER_RAILS` names a rail that doesn't exist, the script stops and lists the available ones.
- **Short runs are coarse.** With a 100 ms period, a run of a few seconds has only a few dozen samples. More tokens or more runs give steadier numbers.

### Output

Each row of the CSV is one run:

| Column | Meaning |
|---|---|
| `timestamp`, `model` | When the run happened and the model path reported by the server |
| `prompt_id`, `run` | Prompt number in the prompts file and repetition number |
| `n_prompt`, `n_gen` | Prompt and generated token counts |
| `pp_tok_s`, `tg_tok_s` | Prompt processing and generation speed |
| `draft_n`, `draft_n_accepted`, `accept_rate` | Draft tokens proposed, accepted, and their ratio (empty without a draft model) |
| `power_w`, `energy_j`, `tok_per_j` | Mean module power, request energy, and generated tokens per joule |

The summary at the end shows the mean ± standard deviation of speed, acceptance, power and tokens per joule for each prompt.
