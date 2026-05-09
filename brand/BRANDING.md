# validate-llm — brand assets

Primary mark: **Shield A** (outlined shield, green check). The cursor wordmark and `pip install` lockup are also approved alternates — use whichever fits the surface best.

## Files

### Mark
- `mark.svg` — primary mark, light background
- `mark-dark.svg` — primary mark, dark background
- `favicon.svg` — favicon-optimized (thicker strokes for 16–32px)

### Combination
- `lockup.svg` / `lockup-dark.svg` — horizontal mark + wordmark

### Wordmark (cursor)
- `wordmark-cursor.svg` — `validate-llm` + animated blinking caret, light
- `wordmark-cursor-dark.svg` — same, dark
- `wordmark-cursor-static.svg` — non-animated version for surfaces that don't support SMIL (PyPI, some markdown renderers)

### Install lockup
- `pip-install.svg` — `$ pip install validate-llm` with green underline, light
- `pip-install-dark.svg` — same, dark

## When to use which

| Surface | Recommended |
|---|---|
| Favicon, app icon | `favicon.svg` |
| Docs header | `lockup.svg` / `lockup-dark.svg` |
| README hero (GitHub) | `wordmark-cursor.svg` (animation works on GitHub) **or** `pip-install.svg` |
| PyPI page | `wordmark-cursor-static.svg` or `pip-install.svg` (PyPI strips animation) |
| Social card / OG image | `lockup.svg` |
| Conference slides | `lockup-dark.svg` on dark, `lockup.svg` on light |

## Tokens

| Token  | Value                       | Use                       |
|--------|-----------------------------|---------------------------|
| Ink    | `#0E0F12`                   | Mark stroke, body text    |
| Bone   | `#FAFAF7`                   | Light surfaces            |
| Green  | `oklch(0.71 0.15 162)`      | Accent — checkmarks, cursors, underlines |
| Mute   | `#7A7972`                   | Captions, muted text      |

## Type pairing

- **Wordmark / display:** Space Grotesk 700, letter-spacing −0.025em
- **Mono / cursor / install:** JetBrains Mono 500–700

## Notes

- Shield path: `M60 8 L106 22 V60 C106 86 86 104 60 114 C34 104 14 86 14 60 V22 Z`
- Favicon uses 8px stroke (vs 6px hero) for visibility at 16–32px
- Cursor blink: SMIL `<animate>` on the caret rect, `dur="1.1s"`. Falls back to a solid caret if SMIL is unsupported (use the `-static` variant explicitly when you know the renderer strips animation)
- All SVGs use `oklch()` for the green — if you need a fallback hex for legacy renderers, substitute `#1FB48A`
