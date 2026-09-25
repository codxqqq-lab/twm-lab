# TWM Lab — интерактивная неизвестная станция

Пользовательская оболочка для **того же канонического neural runtime TWM**, который проверен в
[\`codxqqq-lab/twm\`](https://github.com/codxqqq-lab/twm).

Цель: дать человеку без технического контекста понятный цикл:

**наблюдаем станцию → выбираем воздействие → нейросеть прогнозирует → открываем сохранённый факт → сравниваем → даём модели новый опыт.**

## Три режима

- **Попробовать** — минимум терминов: модули, действие, прогноз, факт, следующий опыт.
- **Как это работает** — история наблюдений, внутренние гипотезы и объяснение recursive rollout.
- **Технический** — seed 17/29/43, raw belief/logits/output, матрица связей и provenance.

Режимы меняют только отображение. Neural inference один и тот же.

## Научная честность

Production path:

\`streamlit_app.py → runtime/ui_adapter.py → runtime/engine.py → canonical R12 weights\`

UI не:
- исправляет prediction;
- подставляет скрытый закон мира;
- передаёт recorded future в neural inference;
- превращает прогноз в «факт»;
- заменяет learned dynamics правилами.

Сохранённый будущий переход используется **только после prediction** — чтобы пользователь мог открыть фактический исход и сравнить его с прогнозом.

Названия «Модуль A», «Модуль B» и т. п. — только нейтральные визуальные ярлыки для абстрактных бинарных узлов. Мы намеренно не называем их реакторами, насосами или дверями: модель такой физической семантики не изучала.

## Канонический runtime

Следующие файлы скопированы без изменения из исходного проверенного репозитория:
- \`runtime/canonical/factor_model.py\`
- \`runtime/canonical/separated_system_id.py\`
- \`runtime/canonical/distilled_evidence_system_id.py\`
- \`runtime/engine.py\`
- \`runtime/schema.py\`
- \`runtime/ui_adapter.py\`
- четыре frozen R13 demonstration rows в \`examples/\`.

\`SOURCE_PROVENANCE.json\` сохраняет SHA-256 канонических файлов и checkpoints.

### Веса

Чтобы не размножать бинарные checkpoints между публичными репозиториями,
\`runtime/weight_bootstrap.py\` на первом старте скачивает шесть **точных** файлов из
\`codxqqq-lab/twm\`. Каждый файл сверяется по SHA-256 из \`SOURCE_PROVENANCE.json\`.
После этого \`runtime.engine\` повторно проверяет те же SHA до \`torch.load\`.

Bootstrap — только deployment plumbing; он не меняет tensors, model code или inference.

## Streamlit Community Cloud

- Repository: \`codxqqq-lab/twm-lab\`
- Branch: \`main\`
- Main file: \`streamlit_app.py\`
- Python: 3.12
- Secrets: не нужны

Первый cold start может быть дольше обычного: приложение скачивает около 18 MiB canonical checkpoints и проверяет их.

## Проверка

\`VERIFICATION.md\` содержит результаты исходной equivalence-проверки:
24/24 cases совпали с оригинальным frozen \`run_r13.py\` по belief, probabilities и hard rollout states.

CI в этом репозитории дополнительно проверяет синтаксис, schema tests, integrity SHA и прямой neural inference.
