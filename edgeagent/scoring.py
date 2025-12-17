"""
Score 기반 스케줄링을 위한 비용 계산 모듈

Cost_i(u, v) = α_i * (P_comp[i][u] + β_i * P_net[u]) + (1 - α_i) * P_comm[(v, u)]

- i: Tool 인덱스
- u: 현재 노드 {DEVICE, EDGE, CLOUD}
- v: 이전 노드 {DEVICE, EDGE, CLOUD}
- α_i: Tool i의 연산 중요도 (0 ~ 1)
- β_i: Tool i의 외부 인터넷 사용 여부 (0 or 1)
- P_comp[i][u]: Tool i의 노드 u에서의 연산 비용 (tools_scenario*.yaml)
- P_net[u]: 노드 u의 외부 네트워크 비용 (system.yaml)
- P_comm[(v,u)]: 노드 v→u 통신 비용 (system.yaml)

Job 시작/종료 비용 (is_first, is_last 플래그로 처리):
- Job 시작: + (1-α) * P_comm[(D, u)]
- Job 종료: + (1-α) * P_comm[(u, D)]
"""

from pathlib import Path
from typing import Optional
import yaml

from .types import Location, ToolPlacement, LOCATIONS


class ScoringEngine:
    """
    Cost_i(u, v) = α * (P_comp[i][u] + β * P_net[u]) + (1-α) * TransCost

    TransCost 계산 방식:
      subagent_mode=True (기본값, SubAgent 간 직접 통신):
        - P_comm[(v, u)] 사용
      subagent_mode=False (Direct 실행, middleware 경유):
        - Job 시작: P^{in}(u)
        - 노드 변경 (v≠u): P^{out}(v) + P^{in}(u)
        - 노드 유지 (v==u): 0
        - Job 종료: + P^{out}(u)
    """

    # 노드 약어 ↔ 전체 이름 매핑
    SHORT_TO_FULL = {"D": "DEVICE", "E": "EDGE", "C": "CLOUD"}
    FULL_TO_SHORT = {"DEVICE": "D", "EDGE": "E", "CLOUD": "C"}
    LOCATION_TO_IDX = {"DEVICE": 0, "EDGE": 1, "CLOUD": 2}

    def __init__(self, system_config_path: str | Path, registry, subagent_mode: bool = True):
        """
        Args:
            system_config_path: system.yaml 경로
            registry: ToolRegistry 인스턴스
            subagent_mode: True면 SubAgent 직접 통신 (기본값), False면 middleware 경유 모델
        """
        self.registry = registry
        self.subagent_mode = subagent_mode
        self.system_config = self._load_system_config(system_config_path)
        self.p_net = self._load_p_net()
        self.p_comm = self._load_p_comm()

        # Middleware In/Out 비용 로드 (subagent_mode=False일 때)
        if not subagent_mode:
            self.p_comm_in = self._load_p_comm_in()
            self.p_comm_out = self._load_p_comm_out()

    def _load_system_config(self, path: str | Path) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _load_p_net(self) -> dict[Location, float]:
        """P_net 로드 (노드별 외부 네트워크 비용)"""
        raw = self.system_config.get("p_net", {})
        return {loc: float(raw.get(loc, 0.5)) for loc in LOCATIONS}

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

    def _get_p_comp(self, tool_name: str, u: Location) -> float:
        """Tool별 P_comp[u] 가져오기"""
        profile = self.registry.get_profile(tool_name)
        if profile and hasattr(profile, 'P_comp') and profile.P_comp:
            idx = self.LOCATION_TO_IDX[u]
            return float(profile.P_comp[idx])
        # 기본값: system.yaml의 p_comp 또는 0.5
        default_p_comp = self.system_config.get("p_comp", {})
        return float(default_p_comp.get(u, 0.5))

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

        # Tool Profile에서 α 가져오기 (기본값: α=0.5)
        alpha = getattr(profile, 'alpha', 0.5) if profile else 0.5

        # β는 data_locality가 external_data일 때 자동으로 1
        data_locality = getattr(profile, 'data_locality', 'args_only') if profile else 'args_only'
        beta = 1 if data_locality == "external_data" else 0

        # 연산 비용: P_comp[i][u] + β * P_net[u]
        p_comp_u = self._get_p_comp(tool_name, u)
        comp_cost = p_comp_u + beta * self.p_net.get(u, 0.5)

        # 통신 비용 계산
        comm_cost = 0.0

        if self.subagent_mode:
            # === SubAgent 직접 통신 모드 (기존 방식) ===
            if is_first:
                # Job 시작: P_comm[(D, u)]
                comm_cost = self.p_comm[("DEVICE", u)]
            elif v is not None:
                if v == u:
                    # 노드 유지: 0
                    comm_cost = 0.0
                else:
                    # 노드 변경: P_comm[(v, u)]
                    comm_cost = self.p_comm[(v, u)]

            if is_last:
                # Job 종료: + P_comm[(u, D)]
                comm_cost += self.p_comm[(u, "DEVICE")]
        else:
            # === Middleware 경유 모드 (Direct 실행) ===
            if is_first:
                # Job 시작: P^{in}(u)
                comm_cost = self.p_comm_in[u]
            elif v is not None:
                if v == u:
                    # 노드 유지: 0 (middleware 경유 안 함)
                    comm_cost = 0.0
                else:
                    # 노드 변경: P^{out}(v) + P^{in}(u)
                    comm_cost = self.p_comm_out[v] + self.p_comm_in[u]

            if is_last:
                # Job 종료: + P^{out}(u)
                comm_cost += self.p_comm_out[u]

        # 총 비용: α * comp_cost + (1-α) * comm_cost
        total_cost = alpha * comp_cost + (1 - alpha) * comm_cost

        return total_cost, comp_cost, comm_cost

    def calculate_chain_cost(
        self,
        tool_names: list[str],
        node_assignments: list[Location],
    ) -> tuple[float, list[ToolPlacement]]:
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

            placements.append(ToolPlacement(
                tool_name=tool_name,
                location=u,
                score=cost,
                exec_cost=comp_cost,
                trans_cost=comm_cost,
            ))

            total_cost += cost

        return total_cost, placements

