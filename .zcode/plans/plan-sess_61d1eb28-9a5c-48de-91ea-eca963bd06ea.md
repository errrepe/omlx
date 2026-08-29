# Plano — Fase H: próximos levers do expert streaming (com toggles nas duas UIs)

Sequência em 5 fases independentes e commitáveis, na ordem que destrava as medições seguintes. Toda feature visível ganha toggle/config na WebUI (card `Advanced → Expert Streaming (SSD)` em `_modal_model_settings.html` + `dashboard.js` + i18n en/zh/zh-TW/fr) e no app macOS (`ModelsDTO.swift` → `ModelSettingsScreenVM.swift` → `ModelSettingsScreen.swift`), seguindo o padrão existente de `expert_streaming_enabled/budget/topk`. Padrão de knob interno: campo `Optional` em `model_settings.py` com default de env var, request+validação em `admin/routes.py:133-141/2402+`, exclusão de perfis (`model_profiles.py:110`), assinatura de runtime para reload (`engine_pool.py:714`).

## H1 — G4: per-layer `eval + clear_cache` para qwen4_exp (bit-exato)

O conversor já seta `layer._stream_eval = True` (`patches/expert_streaming/__init__.py:522`) mas só o decoder do GLM honra a flag (`vendor/.../glm5_next/language.py:863`); no qwen instalado ela é inerte. 

- Novo módulo `patches/expert_streaming/qwen35_stream_eval.py` seguindo o precedente do `adaptive_topk.py:112` (wrap de classe do mlx_vlm instalado): envolve `Qwen3_5MoeDecoderLayer.__call__`; quando `_stream_eval` ativo e a chamada é multi-token (prefill/chunk), faz `out = mx.eval(out)` + `mx.clear_cache()` por camada. Decode começa SEM o eval por camada (o custo de 48 syncs/token é real; o squeeze de decode já é coberto pelo G1) — ativar em decode só se o A/B pagar.
- Knob `expert_streaming_per_layer_eval` (default ON via env `OMLX_EXPERT_STREAMING_PER_LAYER_EVAL`), com UI avançada nas duas UIs (linha extra dentro do card de Expert Streaming) para manter a regra "feature entrou, tem toggle".
- Testes: bit-exatidão contra o caminho atual (padrão de `test_expert_streaming.py:713-853`), round-trip do knob, wrap não interfere com MTP/`target_verify` nem com o fused router.
- Bench A/B idle: TTFT 2k/8k cold, picos do pool (`mx.get_cache_memory()` no sampler), decode tok/s. Atualiza doc (seção Fase H).

## H2 — Learned pin store no servidor (bit-exato, ganho já medido)

`PinController` (`warmer.py:185-250`) já carrega perfil via env e tem `save_profile()` prontos; falta plugar no ciclo de vida do servidor.

- Settings novas: `expert_streaming_pins: Optional[bool]` (default env `OMLX_EXPERT_STREAMING_PIN`), `expert_streaming_pin_gib: Optional[float]` (default 1.25). Perfil aprendido sempre que pins ativos: caminho derivado por modelo (`<model_path>/.omlx/expert_pin_profile.json`) em vez do env estático de módulo.
- Hooks: carregar no convert (já existe, só plumb do caminho); salvar no unload do modelo (`engine_pool.unload_if_idle_unpinned` / teardown do BatchedEngine que já segura `_expert_streaming_backing`) — dump JSON barato.
- UI: toggle "Expert pins" + input de budget dentro do card, nas duas UIs (+4 arquivos de i18n). Zero mudança de output (mlock só afeta quais páginas ficam quentes).
- Testes: round-trip, save-on-unload, reload aquece desde o token 1 (semântica E3).

## H3 — SRP/SCH por modelo (offline, sem UI — ferramenta de análise)

- Trace de roteamento: env `OMLX_EXPERT_STREAMING_TRACE=<path>` faz o `StreamingSwitchGLU` anexar JSONL `{layer, uniq, positions}` por chamada (fora do caminho quente).
- `bench/lrc_analysis.py`: calcula SCH (hit rate ótimo dado futuro, por tamanho de cache) e SRP (cobertura por grupo fixo) do paper LRC; sumário por camada. Valida contra Qwen 23–32% / GLM 0% conhecidos e passa a orientar defaults de pins/seed/topk e o pre-flight de checkpoints novos.

## H4 — Harness de perplexidade (offline, sem UI — pré-requisito de qualidade)

- `bench/ppl_expert_streaming.py`: carrega o checkpoint via `mlx_lm` (o `evaluate.py` existe no venv como referência; implementação direta de NLL sobre arquivo de corpus local, sem dependência de datasets remotos). Como `StreamingSwitchGLU` é bit-exato vs residente (testes `:713-853`), medir no caminho residente é representativo — rápido e sem SSD. Saída: ppl por checkpoint, para comparar oQ4e vs variantes do tier frio.

## H5 — Tier dual-precision no SSD (oQ2.7 para experts frios) — near-lossless

- Ferramenta offline `tools/requant_cold_tier.py` (UX do `stripe_model.py`): gera `<model>/expert_cold/` com os bancos `switch_mlp` requantizados a oQ2.7 (níveis já no `oq.py`).
- Backing: `ExpertBackingStore` resolve a raiz fria; `load_expert_slice` escolhe tier pelo hot set (frequências do pin profile ∪ hotness de prefill; sem perfil = tudo frio). `StreamingQuantizedSwitchLinear` monta DOIS sub-banks por projeção (quente 4-bit, frio 2.7-bit — bits/gs distintos) → dois `mx.gather_qmm` por projeção, combinados no weighted-sum existente; `_RemapPlan` ganha tag de tier por expert.
- Setting `expert_streaming_cold_tier: Optional[str]` (None/"" = off; "2.7"/"3"), validação 400, capability flag `expert_streaming_cold_tier_present` (verificação de arquivos) nos admin flags; assinatura de runtime. UI: seletor dentro do card (visível com streaming ativo), nas duas UIs + i18n.
- Testes: exatidão por tier (expert frio = resultado do oQ2.7 puro), roteamento de tier, assembly misto, round-trip.
- Gate de qualidade (H4) ANTES de qualquer default: publicar delta de ppl oQ4e vs tier. Bench tok/s/TTFT (esperado ~2× no floor do GLM). Depois, re-testar G2/PILOT com o headroom liberado — decisão documentada, sem compromisso.

## Notas transversais

- Validação: janela idle, `cache_cool.py`, `--min-free-gb`/`--mem-ceiling-gib` dimensionados ao psutil available, `resource_sampler.py` ligado (lições pós-F/G).
- Docs: cada fase atualiza `docs/expert-streaming.md` (seção Fase H…); H3/H4 atualizam também o papers doc.
- Swift: usar o MCP do Xcode para build/teste do `apps/omlx-mac`; evitar compilações concorrentes.
- H3/H4 não têm UI por serem ferramentas offline de análise — não são features de runtime; todo o resto exposto acima tem toggle nas duas UIs.

Entrega recomendada: H1 primeiro (estabiliza máquina e medições), H2 na sequência (pequeno e independente), H3/H4 podem rodar em paralelo aos benches de H1, H5 por último por ser a mudança maior e depender do gate de qualidade.