# Instructions for the private mlx-lm fork

## Agent rules

- This checkout is a private research fork. Follow the user's current
  instructions for private pushes and pull requests.
- Preserve upstream history and third-party authorship when importing work.
- If preparing an upstream new-model contribution, include the upstream pull
  request URL with `?template=new_model.md` in the handoff.

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
