# Stateless GraphRAG retrieval

Covers the currently maintained non-trained graph method.

## GraphRAG

Dense ranks all candidates first. The method derives typed title/body entity evidence with the same frozen encoder, then applies local stable insertion of bridge partners. If nothing moves, the ranking is exactly Dense. Entity structure is method-private and never an EvidenceGraph.

The legacy EPGM implementation and its execution-provenance contracts were removed. ISETrace integration will introduce a new trajectory-native design rather than restoring the deleted method as a compatibility layer.
