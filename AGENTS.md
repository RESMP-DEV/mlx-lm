# Instructions for the RESMP-DEV mlx-lm fork

## Agent rules

- This checkout is the RESMP-DEV research fork. Follow the user's current
  instructions for pushes and pull requests.
- Preserve upstream history and third-party authorship when importing work.

## Upstream policy

We never upstream to `ml-explore/mlx-lm`. The maintainers do not want our
contributions and have not asked for them. Publish our work ourselves:

- Push to the `origin` remote (`RESMP-DEV/mlx-lm`). Pull requests there are
  allowed when the user asks for one.
- Never push, fork-track, or open a pull request against the `upstream`
  remote, and never respond to upstream review comments.
- Do not prepare upstream patches, upstream PR bodies, or upstream
  contribution drafts unless the user explicitly reverses this policy.
- Keep upstream remotes fetch-only so an accidental push fails.

## Code standards

- Keep code comments concise (usually 1-2 lines)
- Avoid redundant or excessive inline commentary
- Use ASD-STE100 Simplified Technical English, simple wordings

### Examples

```python
  # Good (explains reason)

  # The schema requires "content" field to be present.
  choice[key_name]["content"] = text if text else None

  # Bad (excessive comment for explicit code)

  # `content` stays present and nullable, the way the schema has
  # it. A model that stops while still inside a reasoning block
  # leaves `text` empty, and dropping the key makes a client raise
  # KeyError instead of reading an empty answer. Streaming deltas
  # are left alone: omitting fields between chunks is normal there.
```
