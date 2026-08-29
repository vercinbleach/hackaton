---
name: cala-query
description: Translate a natural-language request into the smallest valid Cala query plan.
---

# Cala query planning

Produce one plan that preserves the user's filters and requested projection.

- Treat the root as the collection being requested.
- Represent every stated restriction as a filter. Keep `mention` verbatim and normalize only `value`.
- Put only explicitly requested output properties in `return`.
- For a collection query with no explicit output property, return only `name` as the minimum identifier.
- Do not add relationship properties merely to substantiate a filter.
- Use `retrieve_entity` for properties of one named entity.
- Use `unsupported` only when the catalog cannot represent the request.
- Set unused nullable properties to `null` and unused arrays to `[]`.
