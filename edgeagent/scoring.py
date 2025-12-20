"""
Score 기반 스케줄링을 위한 비용 계산 모듈

Score(i, u, v) = ExecCost(i, u) + TransCost(v -> u)

- i: 실행하고자 하는 tool
- u: task i가 실행되는 노드 {DEVICE, EDGE, CLOUD}
- v: tool chain에서 task i 이전 task가 실행된 노드

ExecCost(i, u) = α * (P_comp[i][u] + β * P_net[u])
- α: Tool i의 연산 중요도 (0 ~ 1)
- β: 외부 인터넷 사용 여부 (data_locality가 external_data일 때 1)
- P_comp[i][u]: Tool i의 노드 u에서의 연산 비용
- P_net[u]: 노드 u의 외부 네트워크 비용

TransCost(v -> u) = (1-α) * 통신비용
- 데이터 이동 비용은 미들웨어를 경유하느냐에 따라 결정
- 미들웨어는 DEVICE에 위치

통신비용 계산:
- Job 시작: P^{in}(u) - 미들웨어 → 실행노드 업로드
- 노드 변경 (v≠u): P^{out}(v) + P^{in}(u) - 이전노드 → 미들웨어 → 현재노드
- 노드 유지 (v==u): 0 - 미들웨어 경유 안 함
- Job 종료: + P^{out}(u) - 최종 결과 반환

P^{in}(u) = P_comm[(DEVICE, u)]  # 미들웨어 → 노드 업로드 비용
P^{out}(v) = P_comm[(v, DEVICE)]  # 노드 → 미들웨어 다운로드 비용
"""

from pathlib import Path
from typing import Optional
import yaml

from .types import Location, SchedulingResult, LOCATIONS
from .profiles import NetworkMeasurements


class ScoringEngine:
    """
    Score(i, u, v) = ExecCost(i, u) + TransCost(v -> u)

    ExecCost(i, u) = α * (P_comp[i][u] + β * P_net[u])
    TransCost(v -> u) = (1-α) * 통신비용

    통신비용 (미들웨어 경유):
        - Job 시작: P^{in}(u)
        - 노드 변경 (v≠u): P^{out}(v) + P^{in}(u)
        - 노드 유지 (v==u): 0
        - Job 종료: + P^{out}(u)
    """

    # 노드 약어 ↔ 전체 이름 매핑
    SHORT_TO_FULL = {"D": "DEVICE", "E": "EDGE", "C": "CLOUD"}
    FULL_TO_SHORT = {"DEVICE": "D", "EDGE": "E", "CLOUD": "C"}
    LOCATION_TO_IDX = {"DEVICE": 0, "EDGE": 1, "CLOUD": 2}

    def __init__(self, system_config_path: str | Path, registry):
        """
        Args:
            system_config_path: system.yaml 경로
            registry: ToolRegistry 인스턴스
        """
        self.registry = registry
        self.system_config = self._load_system_config(system_config_path)
        self.p_net = self._load_p_net()
        self.p_comm = self._load_p_comm()
        self.p_comm_in = self._load_p_comm_in()
        self.p_comm_out = self._load_p_comm_out()
        # 실측 기반 동적 계산용
        self.network = NetworkMeasurements.from_config(self.system_config)
        self.hardware_mapping = self.system_config.get("hardware_mapping", {})

    def _load_system_config(self, path: str | Path) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_p_net(self) -> dict[Location, float]:
        """P_net 계산 (노드별 외부 네트워크 비용)

        external_bandwidth_mbps에서 역정규화:
        - 높은 대역폭 → 낮은 비용
        - 낮은 대역폭 → 높은 비용
        - p_net[u] = min_bw / bw[u]
        """
        ext_bw = self.system_config.get("external_bandwidth_mbps", {})
        if not ext_bw:
            # fallback: 기본값
            return {loc: 0.5 for loc in LOCATIONS}

        # 최소 대역폭 기준으로 역정규화
        min_bw = min(ext_bw.values())
        return {loc: min_bw / ext_bw.get(loc, min_bw) for loc in LOCATIONS}

    def _load_p_comm(self) -> dict[tuple[Location, Location], float]:
        """P_comm 로드 (노드 쌍 간 통신 비용)"""
        raw = self.system_config.get("p_comm", {})
        result = {}

        for key, value in raw.items():
            # "D_E" → ("DEVICE", "EDGE")
            parts = key.split("_")
            if len(parts) == 2:
                v_short, u_short = parts
                v_full = self.SHORT_TO_FULL.get(v_short, v_short)
                u_full = self.SHORT_TO_FULL.get(u_short, u_short)
                result[(v_full, u_full)] = float(value)

        # 기본값 설정 (missing pairs)
        for v in LOCATIONS:
            for u in LOCATIONS:
                if (v, u) not in result:
                    if v == u:
                        result[(v, u)] = 0.0
                    else:
                        result[(v, u)] = 0.5

        return result

    def _load_p_comm_in(self) -> dict[Location, float]:
        """P^{in}(u) = P_comm[(D, u)] (middleware → node 업로드 비용)

        middleware는 DEVICE에 위치하므로 D → u 비용 사용
        """
        return {loc: self.p_comm[("DEVICE", loc)] for loc in LOCATIONS}

    def _load_p_comm_out(self) -> dict[Location, float]:
        """P^{out}(v) = P_comm[(v, D)] (node → middleware 다운로드 비용)

        middleware는 DEVICE에 위치하므로 v → D 비용 사용
        """
        return {loc: self.p_comm[(loc, "DEVICE")] for loc in LOCATIONS}

    def _get_p_exec(self, tool_name: str, u: Location) -> float:
        """Tool별 P_exec[u] 가져오기 (실측 기반 또는 fallback)"""
        profile = self.registry.get_profile(tool_name)
        idx = self.LOCATION_TO_IDX[u]

        # 1. 실측 기반 measurements가 있으면 동적 계산
        if profile and hasattr(profile, 'measurements') and profile.measurements:
            p_exec_list = profile.measurements.calculate_p_exec(self.hardware_mapping)
            return float(p_exec_list[idx])

        # 2. Fallback: profile의 P_exec
        if profile and hasattr(profile, 'P_exec') and profile.P_exec:
            return float(profile.P_exec[idx])

        # 3. 기본값
        return 0.5

    def compute_cost(
        self,
        tool_name: str,
        u: Location,
        v: Optional[Location],
        is_first: bool = False,
        is_last: bool = False,
    ) -> tuple[float, float, float]:
        """
        단일 Tool의 비용 계산

        Cost_i(u, v) = α * (P_comp[i][u] + β * P_net[u]) + (1-α) * P_comm[(v,u)]

        Args:
            tool_name: Tool 이름
            u: 현재 노드
            v: 이전 노드 (None이면 Job 시작)
            is_first: 첫 번째 Tool 여부 (Job 시작 비용 추가)
            is_last: 마지막 Tool 여부 (Job 종료 비용 추가)

        Returns:
            (total_cost, comp_cost, comm_cost)
        """
        profile = self.registry.get_profile(tool_name)

        # β는 data_locality가 external_data일 때 자동으로 1
        data_locality = getattr(profile, 'data_locality', 'args_only') if profile else 'args_only'
        beta = 1 if data_locality == "external_data" else 0

        # α 결정: 실측 기반 또는 fallback
        if profile and hasattr(profile, 'measurements') and profile.measurements:
            # 실측 기반 동적 계산: α = T_exec / (T_exec + T_comm_in + T_comm_out)
            alpha = profile.measurements.calculate_alpha(
                self.network, self.hardware_mapping
            )
        else:
            # Fallback: profile의 alpha
            alpha = getattr(profile, 'alpha', 0.5) if profile else 0.5

        # 연산 비용: P_exec[i][u] + β * P_net[u]
        p_exec_u = self._get_p_exec(tool_name, u)
        comp_cost = p_exec_u + beta * self.p_net.get(u, 0.5)

        # === 통신 비용: 서브에이전트 모드 (노드 간 직접 전송) ===
        # comm_cost = 0.0
        #
        # if is_first:
        #     # Job 시작: P^{in}(u)
        #     comm_cost = self.p_comm_in[u]
        # elif v is not None:
        #     if v == u:
        #         # 노드 유지: 0 (middleware 경유 안 함)
        #         comm_cost = 0.0
        #     else:
        #         # 노드 변경: P^{out}(v) + P^{in}(u)
        #         comm_cost = self.p_comm_out[v] + self.p_comm_in[u]
        #
        # if is_last:
        #     # Job 종료: + P^{out}(u)
        #     comm_cost += self.p_comm_out[u]

        # === 통신 비용: 미들웨어 경유 모드 ===
        # 모든 tool 실행이 미들웨어를 경유: in + out
        comm_cost = self.p_comm_in[u] + self.p_comm_out[u]

        # 총 비용: α * comp_cost + (1-α) * comm_cost
        total_cost = alpha * comp_cost + (1 - alpha) * comm_cost

        return total_cost, comp_cost, comm_cost

    def calculate_chain_cost(
        self,
        tool_names: list[str],
        node_assignments: list[Location],
    ) -> tuple[float, list[SchedulingResult]]:
        """
        Tool Chain 전체의 총 비용 계산

        Args:
            tool_names: Tool 이름 리스트
            node_assignments: 각 Tool의 노드 배치

        Returns:
            (total_cost, placements)
        """
        total_cost = 0.0
        placements = []
        n = len(tool_names)

        for k, (tool_name, u) in enumerate(zip(tool_names, node_assignments)):
            v = node_assignments[k - 1] if k > 0 else None
            is_first = (k == 0)
            is_last = (k == n - 1)

            cost, comp_cost, comm_cost = self.compute_cost(
                tool_name, u, v, is_first, is_last
            )

            placements.append(SchedulingResult(
                tool_name=tool_name,
                location=u,
                reason="brute_force_optimal",
                score=cost,
                exec_cost=comp_cost,
                trans_cost=comm_cost,
            ))

            total_cost += cost

        return total_cost, placements

