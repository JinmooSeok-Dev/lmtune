<!--
PR 작성 지침은 plan (`/home/jinmoo/.claude/plans/async-cooking-cat.md`) 의
"산출물 표준" 절을 따른다. PR 1개 = 1 study 또는 1 phase 단위.
-->

## Summary

<!-- 1-3 문장. 무엇을 / 왜 변경했는지. plan 의 어떤 phase·acceptance 항목을 채우는지 명시. -->

## Plan Context

- Phase: <!-- W / B0 / B1 / B2 / S6 / 기타 -->
- Contract item filled: <!-- input #N or output #X (plan § User Contract 참조) -->
- Related study_id (있으면): <!-- st-XXXX -->

## Changes

<!-- 핵심 파일·디렉토리 단위. 추가/삭제/수정 표시. -->

- `src/bench/...` —
- `b200/...` —
- `tests/...` —

## Test Plan

- [ ] `ruff check src tests` 통과
- [ ] `pytest -q` 전체 통과
- [ ] (UI/dashboard 변경 시) 브라우저에서 실제 렌더 확인
- [ ] (helmfile/k8s 변경 시) `helmfile ... --dry-run` 종료코드 0
- [ ] (study 결과 PR) `b200/studies/<id>/ANALYSIS.md` 작성됨

## Risk / Rollback

<!-- 롤백 한 줄 명령. 영향 범위. -->

## Notes

<!-- 후속 작업 후보, 연관 PR 링크, 회귀 알림 등. -->
