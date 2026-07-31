# Stateless GraphRAG retrieval

Covers the currently maintained non-trained graph method.

## GraphRAG

Dense ranks all candidates first. Document inputs derive typed title/body entity evidence. If a request has no title groups (for example ISETrace ToolOutput candidates), the method falls back to bounded entities shared in candidate text, including capitalized spans and structured URLs/paths/filenames. It uses the same frozen encoder to resolve candidates, then applies local stable insertion of bridge partners. If nothing moves, the ranking is exactly Dense. Entity structure is method-private, never an EvidenceGraph, and never reads native provenance edges.

The legacy EPGM implementation and its label-conditioned contracts remain removed. ISETrace uses the new `provenance_path` request/method over the query-independent trajectory graph rather than restoring a compatibility layer. See [`isetrace-nontrain-retrieval.md`](isetrace-nontrain-retrieval.md).
