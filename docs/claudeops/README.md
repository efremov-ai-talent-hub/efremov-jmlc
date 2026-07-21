# ClaudeOps

Качество обеспечивают два субагента, их определения приложены в [`agents/`](./agents):

- [`code-reviewer`](./agents/code-reviewer.md) — ревью незакоммиченных изменений, read-only, находки по severity.
- [`autodoc-maintainer`](./agents/autodoc-maintainer.md) — ведёт `.autodoc/changelog.md`: одно изменение — одна запись.

Копии здесь — документация, не конфигурация: агенты регистрируются из `.claude/agents/`.

Основные конвенции — в [`CLAUDE.md`](../../CLAUDE.md) в корне репозитория, он же входная точка.
