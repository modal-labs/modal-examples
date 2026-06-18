from pathlib import Path

import internal.link_check as link_check


def extract(tmp_path: Path, filename: str, text: str) -> list[link_check.LinkRecord]:
    path = tmp_path / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return list(
        link_check.extract_links_from_file(
            path,
            root=tmp_path,
            base_url="https://modal.com",
        )
    )


def test_markdown_links_are_not_double_counted(tmp_path):
    records = extract(
        tmp_path,
        "example.py",
        "# See [Modal Apps](https://modal.com/docs/guide/apps#ephemeral-apps).\n",
    )

    assert records == [
        link_check.LinkRecord(
            url="https://modal.com/docs/guide/apps#ephemeral-apps",
            target="https://modal.com/docs/guide/apps",
            kind="external",
            file="example.py",
            line=1,
            source="markdown",
        )
    ]


def test_markdown_links_can_contain_parentheses(tmp_path):
    records = extract(
        tmp_path,
        "example.py",
        "# See [Ligand](https://en.wikipedia.org/wiki/Ligand_(biochemistry)).\n",
    )

    assert records[0].target == "https://en.wikipedia.org/wiki/Ligand_(biochemistry)"


def test_yaml_brackets_are_not_markdown_links(tmp_path):
    records = extract(
        tmp_path,
        "data/boltz_affinity.yaml",
        "smiles: 'N[C@@H](Cc1ccc(O)cc1)C(=O)O'\n",
    )

    assert records == []


def test_collects_site_absolute_local_and_skipped_links(tmp_path):
    records = extract(
        tmp_path,
        "example.py",
        "\n".join(
            [
                "# See [Images](/docs/guide/images#base-images).",
                "# ![screenshot](./screenshot.png)",
                "# Runtime docs [here]({modal_example_url}).",
                'url = "http://localhost:8000/health"',
            ]
        ),
    )

    assert records[0].kind == "external"
    assert records[0].target == "https://modal.com/docs/guide/images"
    assert records[1].kind == "local"
    assert records[1].target == str(tmp_path / "screenshot.png")
    assert records[2].kind == "skip"
    assert records[2].reason == "templated URL"
    assert len(records) == 3


def test_ignores_links_outside_prose_comment_lines(tmp_path):
    records = extract(
        tmp_path,
        "example.py",
        "\n".join(
            [
                'url = "https://api.example.test/private"',
                "Plain markdown [link](https://example.test/plain).",
                "# Comment prose https://example.test/prose",
            ]
        ),
    )

    assert [record.url for record in records] == ["https://example.test/prose"]


def test_modal_app_urls_are_skipped(tmp_path):
    records = extract(
        tmp_path,
        "example.py",
        "\n".join(
            [
                "# Try the app at https://modal-labs--example-web.modal.run/.",
                "# Or visit https://u35iiiyqp5klbs.r3.modal.host/.",
            ]
        ),
    )

    assert [record.kind for record in records] == ["skip", "skip"]
    assert [record.reason for record in records] == ["Modal app URL", "Modal app URL"]


def test_normalizes_github_package_ref_urls(tmp_path):
    records = extract(
        tmp_path,
        "example.py",
        '# image = image.uv_pip_install("https://github.com/NVIDIA/NeMo.git@main")\n',
    )

    assert records[0].target == "https://github.com/NVIDIA/NeMo/tree/main"


def test_403_is_broken(monkeypatch):
    def forbidden(_target, *, timeout):
        return link_check.HttpCheckResponse(403, "Forbidden", {})

    monkeypatch.setattr(link_check, "_attempt_http_check", forbidden)

    result = link_check.check_external_target("https://example.test/private")

    assert result.status == "broken"
    assert result.http_status == 403


def test_anti_bot_403_is_skipped(monkeypatch):
    def challenge(_target, *, timeout):
        return link_check.HttpCheckResponse(
            403,
            "Forbidden",
            {"cf-mitigated": "challenge"},
        )

    monkeypatch.setattr(link_check, "_attempt_http_check", challenge)

    result = link_check.check_external_target("https://example.test/challenge")

    assert result.status == "skipped"
    assert result.http_status == 403
    assert result.reason == "blocked by anti-bot challenge"


def test_sec_uses_declared_user_agent():
    assert (
        link_check._user_agent_for_url("https://www.sec.gov/Archives/edgar/Feed/")
        == link_check.SEC_USER_AGENT
    )
    assert (
        link_check._user_agent_for_url("https://modal.com/docs")
        == link_check.USER_AGENT
    )


def test_baseline_allows_known_target_in_known_file():
    records = [
        link_check.LinkRecord(
            url="https://example.test/broken",
            target="https://example.test/broken",
            kind="external",
            file="example.py",
            line=10,
            source="markdown",
        )
    ]
    results = [
        link_check.CheckResult(
            target="https://example.test/broken",
            status="broken",
            reason="HTTP 404",
            http_status=404,
        )
    ]

    analysis = link_check.analyze_baseline(
        records,
        results,
        {"https://example.test/broken": {"example.py"}},
    )

    assert not analysis.has_failures
    assert analysis.known_targets == {"https://example.test/broken"}


def test_baseline_fails_known_target_in_new_file():
    records = [
        link_check.LinkRecord(
            url="https://example.test/broken",
            target="https://example.test/broken",
            kind="external",
            file="new_example.py",
            line=10,
            source="markdown",
        )
    ]
    results = [
        link_check.CheckResult(
            target="https://example.test/broken",
            status="broken",
            reason="HTTP 404",
            http_status=404,
        )
    ]

    analysis = link_check.analyze_baseline(
        records,
        results,
        {"https://example.test/broken": {"example.py"}},
    )

    assert analysis.has_failures
    assert list(analysis.new_locations) == ["https://example.test/broken"]


def test_baseline_fails_new_broken_target():
    records = [
        link_check.LinkRecord(
            url="https://example.test/new-broken",
            target="https://example.test/new-broken",
            kind="external",
            file="example.py",
            line=10,
            source="markdown",
        )
    ]
    results = [
        link_check.CheckResult(
            target="https://example.test/new-broken",
            status="broken",
            reason="HTTP 404",
            http_status=404,
        )
    ]

    analysis = link_check.analyze_baseline(records, results, {})

    assert analysis.has_failures
    assert list(analysis.new_targets) == ["https://example.test/new-broken"]


def test_baseline_skipped_target_is_not_fixed():
    results = [
        link_check.CheckResult(
            target="https://example.test/flaky",
            status="skipped",
            reason="rate-limited by target",
            http_status=429,
        )
    ]

    analysis = link_check.analyze_baseline(
        records=[],
        results=results,
        baseline={"https://example.test/flaky": {"example.py"}},
    )

    assert not analysis.has_failures
    assert analysis.fixed_targets == set()
    assert analysis.unchecked_targets == {"https://example.test/flaky"}
