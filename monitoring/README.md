# monitoring

Endpoint(vLLM)는 외부에서 이미 떠 있어야 합니다. 본 스택은 **Prometheus + Grafana 만** 띄워서 해당 endpoint `/metrics` 를 scrape 합니다.

## 실행

```bash
cd monitoring
# Prometheus 타겟 수정 (host/endpoint 주소)
vim prometheus.yml
# Grafana admin 비밀번호 덮어쓰기 (선택)
export GRAFANA_PASSWORD=my-pass
docker compose up -d
```

- Prometheus: http://localhost:9090
- Grafana:    http://localhost:3000 (admin / admin)

## 대시보드

`../dashboards/grafana/endpoint_live.json` 가 프로비저닝 provider 를 통해 자동 로딩됩니다.
신규 대시보드를 추가하려면 JSON 파일을 `../dashboards/grafana/` 에 둡니다.

## Prometheus 타겟 동적 추가

여러 endpoint 를 측정하려면 `prometheus.yml` 의 `scrape_configs` 에 `endpoint_slug` label 을 부여한 job 을 추가합니다. PromQL 쿼리 시 `{endpoint_slug="..."}` 로 분리하여 동시에 비교 가능합니다.
