# Documentation

Detailed user and technical guides live in this directory. The repository root
keeps only entry points, contribution policy, security policy, and agent
operating guides.

Start here:

- [Flag and workflow guide](advanced-gguf-quantizer-flags.md): recipes,
  commands, artifacts, memory-bounded runs, and validation.
- [Imatrix and KLD guide](advanced-gguf-quantizer-imatrix-kld.md): calibration,
  saved-logit KLD base creation, and evidence reuse rules.
- [Layer policy guide](advanced-gguf-quantizer-layer-policy.md): fused tensor
  groups, output/head policy, MTP/NextN handling, and mixed precision policy.
- [CUDA acceleration guide](advanced-gguf-quantizer-cuda-acceleration.md): where
  CUDA is used and what should stay CPU-side.
- [NVFP4 GGUF contract](advanced-gguf-quantizer-nvfp4-contract.md): required
  NVFP4 tensors, scale meaning, runtime attachment, and validation.
- [MXFP8 GGUF contract](advanced-gguf-quantizer-mxfp8-contract.md): OCP E4M3
  block layout, GGUF identifiers, quantization policy, and CPU/CUDA runtime
  support.

Root documents:

- [README](../README.md): human quick start and feature overview.
- [NOTICE](../NOTICE.md): license notices, attribution, citation guidance, and
  MXFP6 compatibility warning.
- [SKILLS](../SKILLS.md): agent-driven quantization workflow.
- [AGENTS](../AGENTS.md): coding-agent operating guide for this repository.
- [CONTRIBUTING](../CONTRIBUTING.md): contribution scope and review evidence.
- [SECURITY](../SECURITY.md): security reporting and safe-use notes.
