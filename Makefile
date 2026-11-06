# ===== config=====
USER     := ## <your-dockerhub-username>
REPO     := qgpm
HF_VER   := v1.0

HF_IMAGE   := $(USER)/$(REPO):hf-$(HF_VER)
HF_LATEST   := $(USER)/$(REPO):hf-latest

# ===== help =====
.PHONY: help
help:
	@echo "Targets:"
	@echo "  build-hf       Build HF image"
	@echo "  test-hf        Smoke test HF image"
	@echo "  push-hf        Push HF image"
	@echo "  clean          Remove dangling images"

# ===== HF =====
.PHONY: build-hf
build-hf:
	docker build -f Dockerfile.hf \
		-t $(HF_IMAGE) \
		-t $(HF_LATEST) \
		.

.PHONY: test-hf
test-hf:
	@echo "=== Smoke test: $(HF_IMAGE) ==="
	docker run --rm --gpus all $(HF_IMAGE) python -c "\
import torch; assert torch.cuda.is_available(), 'CUDA not available'; \
from flash_attn import flash_attn_func; \
import transformers; \
print(f'torch={torch.__version__} cuda={torch.version.cuda} transformers={transformers.__version__}'); \
print('HF smoke test OK')"

.PHONY: push-hf
push-hf:
	docker push $(HF_IMAGE)
	docker push $(HF_LATEST)

# ===== release: build + test + push =====
.PHONY: release-hf
release-hf: build-hf test-hf push-hf
	@echo "HF released: $(HF_IMAGE)"

# ===== clean =====
.PHONY: clean
clean:
	docker image prune -f
	docker builder prune -f