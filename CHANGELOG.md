# Changelog

## Unreleased

- Add the MiniMax LLM Protocol V1 adapter with MiniMax-M3.1-Flash as the default
  and MiniMax-M3 as the other current catalog choice. Keep explicit MiniMax-M2.7
  selections compatible.
- Support Flash thinking effort, preserve full assistant reasoning/tool history
  and streaming deltas, and accept image/video message parts in the MiniMax
  adapter (LLM specification §8.3 operations and §8.4 model discovery).
- Use the documented global and China direct API endpoints and current standard
  pricing metadata, including the separate M3 long-context tier.
