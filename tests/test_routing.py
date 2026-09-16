"""
test_routing.py — Supervisor 动态路由的单元测试（全程离线，不调 LLM）

校验 _allowed_next 计算出的合法 handoff 集合，以及
decide_next 在"唯一合法路径"下不依赖 LLM 即可确定下一跳（保证安全默认）。
"""
from agents import _allowed_next, decide_next, normalize_sources


def _task(tid, status="pending"):
    return {"id": tid, "description": f"任务 {tid}", "status": status}


def _finding(tid):
    return {"task_id": tid, "summary": "结论", "sources": [],
            "confidence": "medium", "revision": 1}


class TestAllowedNext:
    def test_start_without_tasks_goes_to_planner(self):
        self._assert_only({}, "planner")

    def test_pending_no_critique_can_research_or_critic_choice(self):
        state = {
            "tasks": [_task("t1", "done"), _task("t2", "pending")],
            "findings": {"t1": _finding("t1")},
            "iteration": 0,
            "max_iterations": 4,
        }
        assert set(_allowed_next(state)) == {"researcher", "critic"}

    def test_all_done_without_critique_goes_to_critic(self):
        self._assert_only({
            "tasks": [_task("t1", "done")],
            "findings": {"t1": _finding("t1")},
        }, "critic")

    def test_pass_goes_to_writer(self):
        self._assert_only({
            "tasks": [_task("t1", "done")],
            "findings": {"t1": _finding("t1")},
            "critique": {"decision": "PASS"},
            "iteration": 1,
        }, "writer")

    def test_replan_goes_to_planner(self):
        self._assert_only({
            "tasks": [_task("t1", "done")],
            "findings": {"t1": _finding("t1")},
            "critique": {"decision": "REPLAN", "missing_topics": ["新方向"]},
            "iteration": 1,
        }, "planner")

    def test_revise_with_pending_forces_researcher(self):
        self._assert_only({
            "tasks": [_task("t1", "pending")],
            "findings": {},
            "critique": {"decision": "REVISE",
                         "issues": [{"task_id": "t1", "issue": "来源不足"}]},
            "iteration": 1,
        }, "researcher")

    def test_budget_exhausted_forces_writer_only_when_findings_exist(self):
        # 有产出 + 轮次耗尽 → 强制收尾
        self._assert_only({
            "tasks": [_task("t1", "done")],
            "findings": {"t1": _finding("t1")},
            "critique": {"decision": "REVISE"},
            "iteration": 4,
            "max_iterations": 4,
        }, "writer")

        # 没产出时不能"强制通过"，仍需先规划
        self._assert_only({
            "tasks": [],
            "findings": {},
            "iteration": 4,
            "max_iterations": 4,
        }, "planner")

    def test_all_done_old_revise_without_pending_goes_back_to_critic(self):
        # 返工已做完（没有 pending），即便 critique 还停留在 REVISE，
        # 也必须回到 Critic 复审，而不是反复派出没有任务可做的 Researcher 空转
        state = {
            "tasks": [_task("t1", "done")],
            "findings": {"t1": _finding("t1")},
            "critique": {"decision": "REVISE", "issues": []},
            "iteration": 1,
        }
        self._assert_only(state, "critic")

    def test_review_marker_routes_to_critic(self):
        # Researcher 返工完成 / Planner 补题完成后写入 REVIEW 标记
        state = {
            "tasks": [_task("t1", "done")],
            "findings": {"t1": _finding("t1")},
            "critique": {"decision": "REVIEW"},
            "iteration": 1,
        }
        self._assert_only(state, "critic")

    def test_review_with_pending_allows_choice(self):
        state = {
            "tasks": [_task("t1", "done"), _task("t2", "pending")],
            "findings": {"t1": _finding("t1")},
            "critique": {"decision": "REVIEW"},
            "iteration": 1,
        }
        assert set(_allowed_next(state)) == {"researcher", "critic"}

    def _assert_only(self, state, expected):
        assert _allowed_next(state) == [expected]
        # 唯一合法路径时不需要也不应该调用 LLM
        target, _ = decide_next(state)
        assert target == expected


class TestNormalizeSources:
    def test_string_urls(self):
        srcs = normalize_sources(["https://a.com/x", "https://b.com/y"])
        assert [s["url"] for s in srcs] == [
            "https://a.com/x", "https://b.com/y"]

    def test_dict_items(self):
        srcs = normalize_sources([
            {"title": "A", "url": "https://a.com"},
            {"title": "B", "url": "https://b.com"},
        ])
        assert srcs[0]["title"] == "A"

    def test_dedupe_and_drop_invalid(self):
        srcs = normalize_sources([
            "https://a.com",
            "https://a.com",
            "not-a-url",
            {"url": "ftp://bad", "title": "x"},
        ])
        assert len(srcs) == 1
        assert srcs[0]["url"] == "https://a.com"

    def test_strip_trailing_punctuation(self):
        srcs = normalize_sources(["https://a.com/p,"])
        assert srcs[0]["url"] == "https://a.com/p"

    def test_urls_embedded_in_text(self):
        srcs = normalize_sources("见 https://arxiv.org/abs/1234 这篇论文")
        assert len(srcs) == 1
        assert srcs[0]["url"] == "https://arxiv.org/abs/1234"

    def test_empty(self):
        assert normalize_sources(None) == []
        assert normalize_sources([]) == []
