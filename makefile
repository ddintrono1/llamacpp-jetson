# Include .env vars
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

#############################################
# Vars
#############################################

IMAGE_NAME:=llama-cpp-orin:0.6.0
CONTAINER_NAME:=llamacpp-container
MAX_TOKENS:=2048

# benchmarking
BENCH_PROMPTS:=resources/bench_prompts.txt
BENCH_RUNS:=5
BENCH_N_PREDICT:=128
POWER_INTERVAL:=100
POWER_RAILS:=VDD_GPU_SOC|VDD_CPU_CV|VIN_SYS_5V0 # AGX Orin power rails summed as module power

# Server address, shared by the serve targets and the client targets (infer, bench-sd)
# 0.0.0.0 means the server will listen on whatever (still constrained by container's available network)
HOST:=0.0.0.0
PORT:=8080
API_URL:=http://$(HOST):$(PORT)

# Common `docker run` command with flags: GPU access, local models volume (read-only), port.
DOCKER_RUN:=docker run --rm --gpus all -v $(CURDIR)/models:/models:ro -p $(PORT):$(PORT)

# Common `llama-server` command with flags
SERVER:=llama-server --host $(HOST) --port $(PORT) --api-key $(LLAMA_API_KEY)

# Abort unless variable $(1) is set; $(2) is the value hint shown in the usage message.
# Recursive `=` so $@ resolves to the calling target.
require=if [ -z "$($(1))" ]; then \
		echo "No argument supplied"; \
		echo "Usage: make $@ $(1)=$(2)"; \
		exit 1; \
	fi

.PHONY: serve serve-qwen3.8_27B_sd infer shell build bench 


#############################################
# Targets
#############################################

build:
	docker build -t $(IMAGE_NAME) --build-arg JOBS=4 -f llamacpp.Dockerfile .

# Prompt the loaded model (local only)
infer:
	@$(call require,PROMPT,<your question>); \
	$(call require,LLAMA_API_KEY,<key> (or set it in .env)); \
	jq -n --arg p "$$PROMPT" --argjson m $(MAX_TOKENS) '{messages: [{role: "system", content: "You are a concise assistant."}, {role: "user", content: $$p}], max_tokens: $$m}' | \
	curl -s $(API_URL)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer $(LLAMA_API_KEY)" \
		-d @-

shell:
	@docker exec -it $(CONTAINER_NAME) bash 2>/dev/null \
	|| \
	(echo "$(IMAGE_NAME)"; \
	$(DOCKER_RUN) -it $(IMAGE_NAME) bash)

# Benchmark the running server (speed, draft acceptance, tokens per joule) with bench.py.
# Results go to results/<timestamp>.csv; see `python3 bench.py --help`.
bench:
	python3 bench.py --url $(API_URL) --prompts $(BENCH_PROMPTS) --runs $(BENCH_RUNS) \
		--n-predict $(BENCH_N_PREDICT) --interval $(POWER_INTERVAL) --rails "$(POWER_RAILS)"

# Load the specified model and serve it on port 8080.
serve:
	@$(call require,MODEL,/models/<model>.gguf); \
	$(DOCKER_RUN) \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) \
		$(SERVER) -m $(MODEL) -ngl 99 -c 8192 -np 1 -fa on

# Load Qwen3.8-27B with speculative decoding
serve-qwen3.8_27B_sd:
	$(DOCKER_RUN) \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) \
		$(SERVER) -m models/Qwen3.8-27B-Q4_K_M.gguf -md models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
		--spec-type draft-dflash --spec-draft-n-max 15 --jinja -ngl 99 -c 8192 -np 1 -fa on

clear:
	@docker kill $(CONTAINER_NAME)