"""Human-facing Streamlit shell for the frozen TWM station runtime.

Display logic only. Neural predictions still flow through:
runtime.ui_adapter -> runtime.engine -> canonical R12 weights / frozen R13 rollout.
Recorded futures are consulted only after prediction, for optional comparison.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import streamlit as st

from runtime.engine import SEEDS, verify_integrity
from runtime.schema import edge_pairs, observed_next, validate_scenario
from runtime.ui_adapter import predict_streamlit_payload
from runtime.weight_bootstrap import ensure_canonical_weights


ROOT = Path(__file__).resolve().parent
EXAMPLES = [ROOT / "examples" / f"station_{n}.json" for n in (7, 8, 9, 10)]
STATION_LABELS = {
    "station_7": "Станция A · 7 модулей",
    "station_8": "Станция B · 8 модулей",
    "station_9": "Станция C · 9 модулей",
    "station_10": "Станция D · 10 модулей",
}
MODES = ("Попробовать", "Как это работает", "Технический")
MODULE_NAMES = [f"Модуль {chr(65 + i)}" for i in range(10)]

st.set_page_config(
    page_title="TWM Lab · неизвестная станция",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1200px; padding-top: 2rem;}
        .hero {
            padding: 1.35rem 1.5rem;
            border: 1px solid rgba(35,106,123,.18);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(35,106,123,.08), rgba(35,106,123,.02));
            margin-bottom: 1rem;
        }
        .hero h1 {margin: 0 0 .35rem 0; font-size: 2rem;}
        .hero p {margin: 0; font-size: 1.05rem; line-height: 1.55;}
        .eyebrow {
            font-size: .78rem; font-weight: 700; text-transform: uppercase;
            letter-spacing: .08em; opacity: .66; margin-bottom: .35rem;
        }
        .device-grid {
            display:grid; grid-template-columns:repeat(auto-fit,minmax(135px,1fr));
            gap:.65rem; margin:.4rem 0 .9rem 0;
        }
        .device {
            border:1px solid rgba(0,0,0,.12); border-radius:14px;
            padding:.8rem .85rem; background:rgba(255,255,255,.68);
        }
        .device .name {font-weight:700; margin-bottom:.25rem;}
        .device .state {font-size:.88rem;}
        .on {color:#167244;} .off {color:#9b3d36;}
        .step-card {
            border-left:4px solid #236A7B; padding:.65rem .9rem;
            background:rgba(35,106,123,.05); border-radius:8px; margin:.4rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def trusted_hashes() -> dict[str, str]:
    ensure_canonical_weights()
    return verify_integrity()


@st.cache_data(show_spinner=False)
def predict_cached(history_json: str, actions: tuple[int, ...], seed: int) -> dict:
    return predict_streamlit_payload(history_json, actions, seed)


def module_name(index: int) -> str:
    return MODULE_NAMES[index]


def render_modules(nodes: list[int], title: str | None = None) -> None:
    if title:
        st.markdown(f"**{title}**")
    cards = []
    for i, value in enumerate(nodes):
        state = "активен" if value else "неактивен"
        cls = "on" if value else "off"
        dot = "●" if value else "○"
        cards.append(
            f'<div class="device"><div class="name">{module_name(i)}</div>'
            f'<div class="state {cls}">{dot} {state}</div></div>'
        )
    st.markdown(f'<div class="device-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def graph_dot(nodes: list[int], edges: list[int]) -> str:
    n = len(nodes)
    lines = [
        "digraph station {",
        'graph [bgcolor="transparent", rankdir=LR, pad="0.15", nodesep="0.45", ranksep="0.7"];',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=11, color="#7B8794"];',
        'edge [color="#91A3AE", arrowsize=0.65, penwidth=1.2];',
    ]
    for i, value in enumerate(nodes):
        fill = "#DFF3E7" if value else "#F4E9E7"
        lines.append(f'n{i} [label="{module_name(i)}", fillcolor="{fill}"];')
    for value, (i, j) in zip(edges, edge_pairs(n)):
        if value:
            lines.append(f"n{i} -> n{j};")
    lines.append("}")
    return "\n".join(lines)


def transition_summary(
    before_nodes: list[int],
    before_edges: list[int],
    after_nodes: list[int],
    after_edges: list[int],
) -> dict:
    changed_modules = [
        (i, before_nodes[i], after_nodes[i])
        for i in range(len(before_nodes))
        if before_nodes[i] != after_nodes[i]
    ]
    return {
        "modules": changed_modules,
        "edges": sum(a != b for a, b in zip(before_edges, after_edges)),
    }


def human_change_text(summary: dict) -> str:
    if not summary["modules"]:
        module_text = "модули не изменятся"
    else:
        module_text = "; ".join(
            f"{module_name(index)}: {'включится' if after else 'выключится'}"
            for index, _, after in summary["modules"]
        )
    edge_text = (
        "связи не изменятся"
        if summary["edges"] == 0
        else f"изменится связей: {summary['edges']}"
    )
    return f"{module_text}. {edge_text}."


def belief_story(output: dict) -> tuple[str, float]:
    joint = [float(x) for row in output["joint_belief"] for x in row]
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in joint if p > 0)
    concentration = max(0.0, min(1.0, 1.0 - entropy / math.log(len(joint))))
    if concentration >= 0.67:
        text = "Модель почти сосредоточилась на одной внутренней версии устройства станции."
    elif concentration >= 0.34:
        text = "Модель сузила круг вариантов, но всё ещё удерживает несколько внутренних версий."
    else:
        text = "Модель пока рассматривает много разных внутренних версий станции."
    return text, concentration


def load_scenario(choice: str, uploaded) -> tuple[dict, str]:
    if choice == "Свой сценарий · JSON":
        if uploaded is None:
            raise ValueError("Загрузите JSON, чтобы использовать свой сценарий.")
        raw = uploaded.getvalue()
        source_key = "upload:" + hashlib.sha256(raw).hexdigest()
    else:
        path = next(p for p in EXAMPLES if STATION_LABELS[p.stem] == choice)
        raw = path.read_bytes()
        source_key = path.stem
    scenario = validate_scenario(json.loads(raw.decode("utf-8")))
    return scenario, source_key


def clear_prediction() -> None:
    for key in ("prediction_output", "prediction_key", "fact_revealed"):
        st.session_state.pop(key, None)


def ensure_session(scenario: dict, source_key: str) -> None:
    if st.session_state.get("source_key") != source_key:
        st.session_state.source_key = source_key
        st.session_state.history = scenario["history"].copy()
        st.session_state.recorded_index = 0
        st.session_state.accepted_notice = False
        clear_prediction()


def build_actions(mode: str, action: int, horizon: int, n_nodes: int) -> tuple[int, ...]:
    if mode != "Технический":
        return (action,) * horizon
    tail_text = st.text_input(
        "Последующие действия",
        "",
        help="Через запятую. Если пусто, выбранное действие повторяется на всех шагах.",
    ).strip()
    if not tail_text:
        return (action,) * horizon
    try:
        tail = [int(x.strip()) for x in tail_text.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError("В последовательности действий должны быть только номера модулей.") from exc
    if len(tail) != horizon - 1:
        raise ValueError(f"Нужно указать ровно {horizon - 1} последующих действий.")
    if any(a < 0 or a >= n_nodes for a in tail):
        raise ValueError(f"Номер действия должен быть от 0 до {n_nodes - 1}.")
    return (action, *tail)


def show_hypotheses(output: dict, detailed: bool = False) -> None:
    story, concentration = belief_story(output)
    st.subheader("Что модель думает о скрытых правилах")
    st.write(story)
    st.progress(concentration)
    st.caption(
        "Шкала показывает только концентрацию внутренних весов, а не вероятность того, "
        "что модель права. На неоднозначных историях текущая версия может быть излишне уверенной."
    )
    if detailed:
        left, right = st.columns(2)
        with left:
            st.write("Фактор связей")
            st.dataframe(
                [{"контекст": i, "вес": round(v, 6)} for i, v in enumerate(output["edge_belief"])],
                hide_index=True,
                use_container_width=True,
            )
        with right:
            st.write("Фактор узлов")
            st.dataframe(
                [{"контекст": i, "вес": round(v, 6)} for i, v in enumerate(output["node_belief"])],
                hide_index=True,
                use_container_width=True,
            )


def show_rollout(output: dict, simple: bool) -> None:
    st.subheader("Прогноз во времени")
    st.caption(
        "После первого шага модель уже не получает факты: каждый следующий шаг строится "
        "из её собственного предыдущего прогноза."
    )
    for i, step in enumerate(output["steps"], start=1):
        summary = transition_summary(
            step["nodes_before"],
            step["edges_before"],
            step["nodes_after"],
            step["edges_after"],
        )
        note = "первый прогноз" if i == 1 else "симуляция из предыдущего прогноза"
        with st.expander(
            f"Шаг {i} · {module_name(step['action'])} · {note}",
            expanded=(i <= 2 and not simple),
        ):
            render_modules(step["nodes_after"])
            st.write(human_change_text(summary))


def technical_state(nodes: list[int], edges: list[int]) -> None:
    n = len(nodes)
    adjacency = {(i, j): value for value, (i, j) in zip(edges, edge_pairs(n))}
    st.write("**Узлы:** " + " · ".join(f"{i}: {v}" for i, v in enumerate(nodes)))
    st.dataframe(
        [
            {
                "из / в": str(i),
                **{
                    str(j): "—" if i == j else str(adjacency[(i, j)])
                    for j in range(n)
                },
            }
            for i in range(n)
        ],
        hide_index=True,
        use_container_width=True,
    )


def main() -> None:
    inject_css()
    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">TWM Lab · честная neural world model demo</div>
          <h1>Неизвестная станция</h1>
          <p>Модель видела несколько переходов, но ей не сообщили скрытые правила.
          Выберите воздействие и проверьте, сможет ли она предсказать последствия.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        with st.spinner("Проверяем канонические веса и целостность модели…"):
            hashes = trusted_hashes()
    except Exception as exc:
        st.error(f"Не удалось подготовить канонические веса или проверить целостность: {exc}")
        st.stop()

    mode = st.radio(
        "Режим",
        MODES,
        horizontal=True,
        label_visibility="collapsed",
        help="Во всех режимах работает один и тот же neural inference; меняется только количество показанных деталей.",
    )

    with st.sidebar:
        st.header("Эксперимент")
        choice = st.selectbox(
            "Станция",
            [STATION_LABELS[p.stem] for p in EXAMPLES] + ["Свой сценарий · JSON"],
        )
        uploaded = None
        if choice == "Свой сценарий · JSON":
            uploaded = st.file_uploader("JSON с 8 наблюдаемыми переходами", type="json")
        try:
            scenario, source_key = load_scenario(choice, uploaded)
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            st.info(str(exc))
            st.stop()
        ensure_session(scenario, source_key)

        if mode == "Технический":
            seed = st.selectbox("Checkpoint seed", SEEDS, index=0)
            st.caption("Это три отдельные проверенные пары весов, а не ensemble.")
        else:
            seed = 17
            st.caption("В обычном режиме используется одна фиксированная проверенная модель.")

        if st.button("↺ Начать сценарий заново", use_container_width=True):
            st.session_state.history = scenario["history"].copy()
            st.session_state.recorded_index = 0
            st.session_state.accepted_notice = False
            clear_prediction()
            st.rerun()

        with st.expander("Что здесь честно, а что условно"):
            st.write(
                "Названия «модулей» — только удобные ярлыки для абстрактных бинарных узлов. "
                "Модель не знает реальную физику станции."
            )
            st.write(
                "Сохранённый «факт» показывается только после neural prediction и никогда "
                "не передаётся модели заранее."
            )

    history = st.session_state.history
    cursor = st.session_state.recorded_index
    current = history[-1]
    nodes, edges = current["nodes_after"], current["edges_after"]
    recorded = scenario.get("recorded", [])
    next_recorded = recorded[cursor] if cursor < len(recorded) else None

    if st.session_state.pop("accepted_notice", False):
        st.success(
            "Новое фактическое наблюдение добавлено в историю. "
            "На следующем прогнозе модель пересчитает свои внутренние версии."
        )

    top_left, top_right = st.columns([1.05, 1.1])
    with top_left:
        st.subheader("Станция сейчас")
        render_modules(nodes)
        st.metric("Активных связей", sum(int(v) for v in edges))
        st.caption(
            "Зелёный модуль = 1, красный = 0. Это абстрактные состояния, "
            "а не реальные физические устройства."
        )
    with top_right:
        st.subheader("Схема активных связей")
        st.graphviz_chart(graph_dot(nodes, edges), use_container_width=True)
        st.caption("Стрелки показывают только наблюдаемое текущее состояние связей.")

    st.divider()
    st.subheader("1 · Выберите воздействие")
    default_action = next_recorded["action"] if next_recorded else 0
    action = st.selectbox(
        "Какой модуль затронуть?",
        list(range(scenario["n_nodes"])),
        index=default_action,
        format_func=module_name,
        key=f"action-{source_key}-{cursor}",
    )
    if next_recorded and action == next_recorded["action"]:
        st.caption(
            "Для этого действия в журнале есть сохранённый фактический исход — "
            "после прогноза его можно открыть и сравнить."
        )
    else:
        st.caption(
            "Модель всё равно сделает прогноз. Если в журнале нет такого исхода, "
            "интерфейс не будет изображать его как факт."
        )

    if mode == "Попробовать":
        look_ahead = st.checkbox("Заглянуть дальше одного шага", value=False)
        horizon = st.slider("Глубина прогноза", 2, 6, 3) if look_ahead else 1
        st.caption(
            "При многошаговом просмотре выбранное действие повторяется. "
            "После первого шага модель использует собственные прогнозы."
        )
    elif mode == "Как это работает":
        horizon = st.slider("Сколько шагов смоделировать", 1, 8, 3)
        st.caption(
            "Одинаковое выбранное действие повторяется на каждом шаге, "
            "чтобы легче увидеть накопление прогноза."
        )
    else:
        horizon = st.slider("Rollout horizon", 1, 16, 3)

    try:
        actions = build_actions(mode, action, horizon, scenario["n_nodes"])
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    history_json = json.dumps(history, ensure_ascii=False, separators=(",", ":"))
    prediction_key = hashlib.sha256(
        (history_json + "|" + ",".join(map(str, actions)) + f"|{seed}").encode("utf-8")
    ).hexdigest()

    if st.button("Сделать прогноз", type="primary", use_container_width=True):
        with st.spinner("Нейросеть анализирует историю и рассчитывает последствия…"):
            output = predict_cached(history_json, actions, seed)
        st.session_state.prediction_output = output
        st.session_state.prediction_key = prediction_key
        st.session_state.fact_revealed = False

    output = (
        st.session_state.get("prediction_output")
        if st.session_state.get("prediction_key") == prediction_key
        else None
    )
    if output is None:
        st.info("Выберите воздействие и нажмите «Сделать прогноз».")
        if mode == "Как это работает":
            st.markdown(
                """
                **Что произойдёт под капотом:**
                1. Neural system-ID перечитает последние 8 реальных наблюдений.
                2. Он сформирует веса внутренних гипотез о скрытой динамике.
                3. Neural transition model рассчитает следующее состояние для этих гипотез.
                4. Результаты будут смешаны согласно learned belief.
                5. UI только переведёт числа в эту визуальную форму.
                """
            )
        st.stop()

    first = output["steps"][0]
    summary = transition_summary(
        nodes, edges, first["nodes_after"], first["edges_after"]
    )

    st.divider()
    st.subheader("2 · Прогноз модели")
    st.markdown(
        f'<div class="step-card"><b>Если воздействовать на {module_name(action)}</b><br>'
        f'{human_change_text(summary)}</div>',
        unsafe_allow_html=True,
    )
    pred_left, pred_right = st.columns([1, 1])
    with pred_left:
        render_modules(first["nodes_after"], "Предсказанное следующее состояние")
    with pred_right:
        st.metric("Изменится модулей", len(summary["modules"]))
        st.metric("Изменится связей", summary["edges"])
        if summary["modules"]:
            st.write("**Ключевые изменения:**")
            for i, _, after in summary["modules"]:
                st.write(f"• {module_name(i)} → {'активен' if after else 'неактивен'}")
        else:
            st.write("Модель не ожидает изменений состояния модулей.")

    show_hypotheses(output, detailed=(mode != "Попробовать"))

    if len(output["steps"]) > 1:
        show_rollout(output, simple=(mode == "Попробовать"))

    st.divider()
    st.subheader("3 · Проверить прогноз фактом")
    observed = observed_next(scenario, cursor, nodes, edges, action)
    if observed is None:
        st.info(
            "Для выбранного действия в сохранённом журнале нет наблюдаемого исхода. "
            "Поэтому здесь есть только prediction — интерфейс не придумывает «правильный ответ»."
        )
    else:
        st.write(
            "Для этого действия существует сохранённый следующий переход "
            "из синтетического мира."
        )
        if not st.session_state.get("fact_revealed", False):
            if st.button("Показать сохранённый фактический исход", use_container_width=True):
                st.session_state.fact_revealed = True
                st.rerun()
        else:
            node_errors = sum(
                a != b for a, b in zip(first["nodes_after"], observed["nodes_after"])
            )
            edge_errors = sum(
                a != b for a, b in zip(first["edges_after"], observed["edges_after"])
            )
            if node_errors + edge_errors == 0:
                st.success("Модель полностью угадала сохранённое следующее состояние.")
            else:
                st.warning(
                    f"Модель ошиблась: не совпало модулей — {node_errors}, "
                    f"связей — {edge_errors}. Это реальная ошибка prediction; UI её не исправляет."
                )
            compare_left, compare_right = st.columns(2)
            with compare_left:
                render_modules(first["nodes_after"], "Прогноз")
            with compare_right:
                render_modules(observed["nodes_after"], "Наблюдаемый исход")
            if st.button(
                "Учесть этот факт и продолжить",
                type="primary",
                use_container_width=True,
            ):
                st.session_state.history = (history + [observed])[-8:]
                st.session_state.recorded_index = cursor + 1
                st.session_state.accepted_notice = True
                clear_prediction()
                st.rerun()

    if mode == "Как это работает":
        st.divider()
        st.subheader("Что именно было дано модели")
        st.write(
            "Модель получила только последние восемь наблюдаемых переходов: "
            "состояния до и после действий. Скрытые названия законов и сохранённое "
            "будущее ей не передаются."
        )
        st.dataframe(
            [
                {
                    "наблюдение": i + 1,
                    "действие": module_name(t["action"]),
                    "изменилось модулей": sum(
                        a != b for a, b in zip(t["nodes_before"], t["nodes_after"])
                    ),
                    "изменилось связей": sum(
                        a != b for a, b in zip(t["edges_before"], t["edges_after"])
                    ),
                }
                for i, t in enumerate(history)
            ],
            hide_index=True,
            use_container_width=True,
        )

    if mode == "Технический":
        st.divider()
        st.subheader("Техническая панель")
        tabs = st.tabs(["Состояние", "Belief", "Raw output", "Provenance"])
        with tabs[0]:
            technical_state(nodes, edges)
        with tabs[1]:
            show_hypotheses(output, detailed=True)
            st.json(
                {
                    "edge_logits": output["edge_logits"],
                    "node_logits": output["node_logits"],
                }
            )
        with tabs[2]:
            st.json(output)
        with tabs[3]:
            st.write("Checkpoint seed:", seed)
            st.json(
                {
                    k: v
                    for k, v in hashes.items()
                    if k.startswith("weights/") and f"seed{seed}" in k
                }
            )
            st.caption(
                "Canonical neural modules and examples are also SHA-checked "
                "by runtime.engine.verify_integrity()."
            )

    st.divider()
    st.caption(
        "TWM Lab: пользовательская оболочка не меняет neural inference. "
        "Станция — синтетическая абстрактная среда; это не модель реального промышленного объекта."
    )


if __name__ == "__main__":
    main()
