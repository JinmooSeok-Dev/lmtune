# well-lit path — pd-disaggregation (B200, placeholder)

Phase B1 에서 채움. 본 디렉토리는 Phase B0 단계에서 트리만 잡아두기 위한 placeholder.

## 다음 단계 (B1 진입 시)

- peer repo 의 해당 path 템플릿을 fork 하여 `helmfile.yaml.gotmpl` 작성
- 모델별 `values-<model>.yaml` 작성 (16-GPU 토폴로지 인지)
- B0 에서 검증된 base overlay (`../base/values-b200-common.yaml.gotmpl`) 상속
