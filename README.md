# automatic_github_copilot_Poisson

GitHub Copilot CLI を使って、**実装 → ローカルチェック → レビュー** を反復し、レビューが `PASS` になったら自動コミットまで行うためのスクリプト集です。
例として流体のCavity流れを実装させています。（Poisson分布）

主なファイル:

- `copilot-review-loop.zsh`: 実装・レビューのループ本体
- `run.sh`: 実行例
- `task.md`: 実装対象タスクのサンプル

## 仕組み

`copilot-review-loop.zsh` は次の流れで動作します。

1. タスクファイルをもとに実装エージェントを実行する
2. lint / typecheck / test をローカルで実行する
3. 差分とチェック結果をもとにレビューエージェントを実行する
4. レビュー結果が `PASS` かつ issue 0 件になるまで繰り返す
5. 成功時は必要に応じて自動コミットする

デフォルト構成では以下のエージェントを利用します。

- 実装: `implementer` (`.github/agents/implementer.agent.md` 上の `Claude Implementer`, model: `claude-sonnet-4.6`)
- レビュー: `reviewer` (`.github/agents/reviewer.agent.md` 上の `GPT Reviewer`, model: `gpt-5.4`)

## 必要条件

- Git リポジトリ内で実行すること
- 以下のコマンドが使えること
  - `git`
  - `jq`
  - `cp`
  - `copilot`
  - `python3`
  - `sed`
  - `tr`
  - `grep`
  - `date`
  - `wc`

また、`copilot` CLI 側で `implementer` / `reviewer` エージェントが使える状態である必要があります。

## 使い方

最小例:

```bash
./copilot-review-loop.zsh --task-file task.md
```

このリポジトリの `run.sh` は、Python プロジェクト向けに lint / typecheck / test とコミットメッセージを指定した実行例です。

```bash
./run.sh
```

内容:

```bash
./copilot-review-loop.zsh \
  --task-file task.md \
  --lint-cmd "ruff check ." \
  --typecheck-cmd "mypy ." \
  --test-cmd "pytest -q" \
  --commit-message "feat: strengthen registration validation"
```

## オプション

```text
./copilot-review-loop.zsh [options]

-t, --task-file FILE            Task description file (default: task.md)
-n, --max-iters NUM             Max review loop iterations (default: 5)
--test-cmd CMD                  Test command (simple argv-style only)
--lint-cmd CMD                  Lint command (simple argv-style only)
--typecheck-cmd CMD             Typecheck command (simple argv-style only)
--log-dir DIR                   Log directory (default: .copilot-loop-logs)
--implementer-agent NAME        Custom agent name for implementation
--reviewer-agent NAME           Custom agent name for review
--commit-message MSG            Commit message after PASS
--no-auto-commit                Do not auto-commit even if PASS
--include-task-file             Include task file in commit
-h, --help                      Show help
```

## コマンド指定の制約

`--test-cmd` / `--lint-cmd` / `--typecheck-cmd` には、**単純な argv 形式のコマンド**だけを渡せます。

許可される例:

```bash
pytest -q
pnpm test -- --runInBand
```

拒否される例:

```bash
echo hi | sed ...
FOO=bar pytest
python -c "..."
cmd > out.txt
$(subcommand)
$HOME/bin/tool
```

シェル機能が必要な場合は、スクリプトファイルに切り出してそのスクリプトを呼び出してください。

## ログ

各イテレーションの生成物は既定で `.copilot-loop-logs/` に保存されます。主に以下が出力されます。

- 実装用プロンプト
- 実装エージェントの出力
- ローカルチェック結果
- Git 差分ログ
- レビュープロンプト
- レビュー JSON

レビューが `PASS` しない場合は、このディレクトリを見ると原因を追いやすくなります。

## 補足

- `task.md` は実装対象の指示書です
- `--include-task-file` を付けない限り、`task.md` は自動コミット対象から除外されます
- `.copilot-loop-logs` も自動コミット対象から除外されます
