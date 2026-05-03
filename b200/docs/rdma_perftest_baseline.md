# RDMA Perftest Baseline — B200 인터노드 fabric

> Phase B6 의 host-level baseline. B0 의 `b200/scripts/fabric_test.yaml` (k8s pod / NCCL all-reduce) 와 별개로, **호스트에서 직접 `ib_write_bw` / `ib_read_bw` / `ib_send_bw` 를 측정**해 두 노드 사이 RDMA fabric 의 raw bandwidth 를 확인한다. NHN Cloud B200 환경에서 Non-privileged Pod RDMA Write 363.98 Gbps 의 host 기준 재현 절차.

## 0. 사전 준비

| 항목 | 확인 명령 | 기대 |
|:---|:---|:---|
| RDMA NIC | `ibv_devices` | `mlx5_0` 등 device 1+ 개 |
| 링크 상태 | `ibstat \| grep -E '(State\|Rate)'` | `State: Active`, `Rate: 200/400 Gb/sec` |
| GID | `show_gids` (mft 또는 `ibv_devinfo -d <dev> -v`) | RoCE v2 의 IPv4 GID index 확인 (보통 3) |
| Perftest | `which ib_write_bw ib_read_bw ib_send_bw` | 모두 PATH 에 |
| 방화벽 | `nc -zv <peer> 18515` | 18515-18517/tcp 열림 |
| 권한 | `getcap $(which ib_write_bw)` 또는 root | `cap_net_raw,cap_ipc_lock` 또는 sudo |

설치 (Ubuntu/Debian):
```bash
sudo apt-get install -y perftest libibverbs-utils ibutils infiniband-diags
```

## 1. 표준 측정 (한 번에 3-test)

본 repo 의 자동화 스크립트:

```bash
# 노드 A (server)
bash b200/scripts/rdma_bench.sh server
# → 출력: data/raw/rdma/<TS>/ib_{write,read,send}_bw.server.txt

# 노드 B (client) — 노드 A 의 RDMA NIC IP 를 인자로
bash b200/scripts/rdma_bench.sh client 10.x.x.x
# → 출력: data/raw/rdma/<TS>/ib_{write,read,send}_bw.client.txt + summary.json
```

기본 옵션 (env 로 override 가능):
- `RDMA_DEVICE=mlx5_0`
- `GID_INDEX=3` (RoCE v2)
- `MSG_SIZE=65536` (64 KiB)
- `QP_COUNT=2`
- `DURATION=30` (초)

출력 `summary.json` 예:
```json
{
  "ts": "20260429T123000Z",
  "tests": {
    "ib_write_bw": {"avg_gbps": 363.98, "peak_gbps": 365.20, "samples": 30},
    "ib_read_bw":  {"avg_gbps": 358.10, "peak_gbps": 360.85, "samples": 30},
    "ib_send_bw":  {"avg_gbps": 361.42, "peak_gbps": 363.05, "samples": 30}
  }
}
```

## 2. 수동 단일 명령 (디버깅용)

`b200/scripts/rdma_bench.sh` 가 실패할 때 직접:

```bash
# server
ib_write_bw -F -d mlx5_0 -x 3 -s 65536 -q 2 -D 30 --report_gbits

# client
ib_write_bw -F -d mlx5_0 -x 3 -s 65536 -q 2 -D 30 --report_gbits <SERVER_IP>
```

핵심 플래그:
- `-F` : CPU frequency 검증 skip (가상화/대형 시스템에서 실패 회피)
- `-x` : GID index — RoCE v2 에서 IPv4 GID 의 index. 잘못 잡으면 "Couldn't read remote address"
- `-s` : message size (64 KiB 가 throughput peak 근방)
- `-q` : QP 개수 — multi-QP 로 link rate 도달
- `-D` : duration 초 (vs `-n iter`). 30 초 권장
- `--report_gbits` : 단위를 Gbps 로

## 3. 결과 해석

| Bandwidth | 의미 |
|:---|:---|
| ≥ 360 Gbps (400 GbE/NDR 의 ~91%) | 정상. 라인 레이트 실효치 (Encoding overhead 차감) |
| 200-350 Gbps | 1개 PF 만 활성 / GID 잘못 / NUMA cross-socket. `nvidia-smi topo -m` + `lstopo` 로 토폴로지 확인 |
| < 200 Gbps | RDMA 미활성, TCP fallback, 케이블/스위치 문제. `dmesg \| grep mlx5` |
| 측정 실패 | `RDMA_NIC_PCIE_RELAXED_ORDERING=1` 환경 변수 또는 firmware 업데이트 필요 가능 |

NHN Cloud B200 reference: **Non-priv Pod 363.98 Gbps RDMA Write** (mlx5 mlx_5_x, ConnectX-7 400 GbE NDR).

## 4. 변수 sweep (B6 axis 입력)

axis 효과 측정용:

```bash
# msg size sweep — write 대역폭 vs message size 곡선
for SZ in 1024 4096 16384 65536 1048576; do
  MSG_SIZE=$SZ DURATION=10 bash b200/scripts/rdma_bench.sh client <SERVER_IP> 2>&1 | tee log.${SZ}.txt
done

# QP 개수 sweep
for Q in 1 2 4 8; do
  QP_COUNT=$Q DURATION=10 bash b200/scripts/rdma_bench.sh client <SERVER_IP>
done
```

결과는 `b200/search-spaces/b6_lowlevel.yaml` 의 `nccl_*` axis 와 결합되어 application-level 영향력 추정에 쓰인다.

## 5. NCCL 전달

본 host-level baseline 이 OK 면 다음으로 NCCL all-reduce (B0 fabric_test.yaml) 측정:

```bash
kubectl logs -n bench-fabric-test job/nccl-test-allreduce | tail -30
# busbw_GB/s 값이 위 ib_write_bw 의 ~80-90% 면 stack OK
```

NCCL bus_bw < 70% × ib_write_bw → NCCL plugin / ENV(NCCL_IB_HCA, NCCL_IB_GID_INDEX, NCCL_IB_SL) 점검.

## 6. 결과 archive

`b200/studies/B0_smoke/` 또는 `b200/studies/B6_<TS>/` 에 다음을 보관:
- `summary.json`
- `ib_write_bw.client.txt` (raw, debug 용)
- `nvidia-smi topo -m` 출력
- `lspci -tv` 출력
- `cat /proc/cmdline` (kernel param)

이 archive 는 B6 의 PCIe ACS / IOMMU pt / NUMA pinning axis 변경 전·후 비교 baseline 이 된다.

## References

- NVIDIA Mellanox Perftest: <https://github.com/linux-rdma/perftest>
- RoCE v2 GID 가이드: <https://enterprise-support.nvidia.com/s/article/understanding-show-gids-script>
- 사용자 이력서 v5 — NHN Cloud B200 Non-privileged Pod RDMA Write 363.98 Gbps
- B0 fabric_test (k8s 측정): `b200/scripts/fabric_test.yaml`
