"""Phase 1 of the client-call -> requirements pipeline.

Extracts cited statements (verbatim quote + speaker + timestamp) from a
client call transcript into auto-discovered per-feature knowledge docs
under ``knowledge/``, merging a new call's statements into existing docs
for the same feature rather than overwriting them.

The rule underneath all of it, enforced structurally rather than merely
prompted for: NO CITATION, NO STATEMENT. The model is used for exactly
one thing -- pulling quoted statements out of one transcript -- and the
doc writing is plain Python, so a statement with no real quote to point
at cannot reach a file.
"""

__version__ = "0.1.0"
