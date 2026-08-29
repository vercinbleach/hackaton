---
name: cala-query
description: Translate a natural-language request into the smallest valid Cala query plan.
---

# Cala query planning

Produce one plan that preserves the user's filters and requested projection.

- Treat the root as the collection being requested.
- Represent every stated restriction as a filter. Keep `mention` verbatim and normalize only `value`.
- Put only explicitly requested output properties in `return`.
- When the user asks for entities through a relationship, return the entity name plus the relationship property needed to substantiate the match. For companies founded by former employees of an organization, return `name` and `founder`.
- Use `retrieve_entity` for properties of one named entity.
- Use `unsupported` only when the catalog cannot represent the request.
- Set unused nullable properties to `null` and unused arrays to `[]`.

Examples:

`compañías que han sido fundadas por ex Google employees` maps to root `companies`, filter `previous_job_eq` with mention and value `Google`, and return fields `name` and `founder`.

`nombre y año de fundación de empresas creadas por exempleados de Google` uses the same filter and returns only `name` and `founded_year`.
