---
name: kda-git-push-setup
description: KDA 仓库 push 到 GitHub 的 remote/凭证约定——多账号 store helper 会串号导致私有仓库报 Repository not found
metadata:
  type: reference
---

KernelDesignAgent 仓库推送到 `github.com/Rainchar9119/KernelDesignAgent`（**私有仓库**，owner
账号 `Rainchar9119` 是用户本人）。

**坑（2026-08-10 踩过）**：本机 `~/.git-credentials` 用 `store` helper，存了**多个** `github.com`
账号(`yudaohai666`、`200554918`…)。store 对同一 host 多行时会取到**非目标账号**的 token，
而它们无权访问 `Rainchar9119` 的私有库 → GitHub 统一回 **`remote: Repository not found`**
（私有库权限不足时不报 403、伪装成 not found）。表现为「上次能推、这次突然 not found」，
极易误判成网络/仓库被删。

**排查顺序（只读，先定位再动手）**：
1. `git ls-remote https://github.com/git/git.git HEAD` 验公网连通（通=排除网络/代理）。
2. `env | grep -i proxy` + `git config --get http.proxy` 查代理。
3. `grep -c "@github.com" ~/.git-credentials` 看是否多账号串号（本坑的根因）。

**修法（已落地，B 方案）**：
- remote 改带用户名形式,让 git 优先匹配对应凭证行:
  `git remote set-url origin https://Rainchar9119@github.com/Rainchar9119/KernelDesignAgent.git`
- 在 `~/.git-credentials` 追加一行 `https://Rainchar9119:<PAT>@github.com`（`umask 077`,chmod 600,
  写时不回显 token）。PAT 需 `repo` 权限。
- 之后 `git push origin main` 正常（2026-08-10 已验证 `dff0f62..adc78e5`）。

**安全**：写 token 进凭证文件/贴进对话后,提醒用户去 GitHub revoke 并重生成;换 token 时更新
`~/.git-credentials` 里 `Rainchar9119` 那一行即可。记忆里**绝不存 token 本体**。

git 常规约定见仓库层规则：不推 main 除非明确许可（本仓库用户已明确要求推 main）。
