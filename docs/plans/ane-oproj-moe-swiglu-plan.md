# Plano de Execução: Extensões ANE para o oMLX

**Branch:** `feat/ane-oproj-moe-swiglu` (já criado a partir de `main` @ `e008a66b`, v0.6.4, sincronizado com a remote em 2026-08-29)
**Status:** Planejamento completo. Nenhuma linha de implementação foi escrita. Este documento é o guia de handoff para a sessão de execução.
**Escopo aprovado pelo usuário:** Features A+B+C (abaixo), com o tuner ANE estendido já neste branch. Features D e E explicitamente excluídas.

---

## 0. Contexto e objetivo

O oMLX tem um recurso experimental de prefill híbrido ANE/GPU para Qwen3.5/3.6/3.8 densos (`qwen35_ane_prefill`). Este plano adiciona três extensões opt-in por modelo:

| # | Feature | O que faz |
|---|---------|-----------|
| **A** | **SwiGLU-in-ANE** | Move a ativação SwiGLU (silu·mul) da fatia ANE para *dentro* do programa ANE (mega-kernel), simplificando o merge Metal |
| **B** | **o_proj split** | Estende o split de canais ANE/GPU à projeção de saída de atenção (o_proj) — token-local, seguro para INT8 |
| **C** | **MoE shared expert** | Habilita o caminho ANE para o shared expert denso (always-on) de modelos MoE (ex.: Qwen3.6-35B-A3B) |

Todas com: default `False`, toggle/slider na UI do app Mac, fallback limpo para o caminho atual, testes, integração ao tuner/benchmark, e documentação atualizada.

### Decisões já tomadas (não reabrir)
- **Escopo A+B+C** — D (steering de prefill por request no scheduler) e E (draft/SpecPrefill no ANE) ficam para branch futuro.
- **Tuner estendido agora** — o_proj entra como dimensão calibrada no "Tune ANE Split" já neste branch (não só sliders manuais).
- **CPU-sharing em MoE fica não suportado** nesta iteração (forçado a 0, com log).
- **SwiGLU-in-ANE desabilita CPU gate/up sharing** nesta iteração (merge ANE-act + CPU-raw + GPU-raw fora de escopo).
- **QKV/k/v_proj NUNCA vão para o ANE** — KV cache é estado; aproximação INT8 acumularia erro (mesma política recurrent-safe do GDN qkv). **Router MoE nunca** — paridade bit-exata da seleção é requisito do repo.

### Descobertas-chave que sustentam o design (provenientes de exploração profunda + fontes Orion/ane-infer)
- O runtime privado do ANE **já aceita `silu`+`mul` em produção**: o programa fp16 fused-down (`fp16_swiglu_down_mil`, `qwen35_ane.mm:516`) os usa. Os programas gate+up INT8 atuais terminam num único `conv`; SwiGLU é aplicado nos kernels Metal de merge (`qwen35_ane_merge*_swiglu_output_*`).
- Kernels Metal plain (`qwen35_ane_merge*_output_*`) e `qwen35_ane_swiglu_suffix_` já existem.
- Os ops nativos `qwen35_ane_(dual_)affine_qmm_t` (usados pelo GDN z) fazem exatamente o que o o_proj precisa: fatia ANE + sufixo qmm GPU + merge plain concat.
- `shared_expert` (um `Qwen3NextMLP`/`Qwen3_5MoeMLP` com `gate/up/down_proj`) **já é alcançado pelo scan** `model.modules()` — hoje o pickup é *acidental* e passa a ser *governado* por toggle.
- Orion (arXiv 2603.06728) catalogou restrições do ANE: máx ~16 BLOBFILEs por programa (ops elementwise como `pow`/scalar consomem slots; `silu`/`mul` não constam peso-blob mas devem ser verificados), superfícies multi-I/O ordenadas alfabeticamente, mínimo ~49 KB por IOSurface. O design do slice A (2 convs + silu + mul por procedimento, ~4 chunks de peso) está bem abaixo do orçamento.

---

## 1. Mapa de wiring — os 11 registros para CADA novo setting

Estas são TODAS as pontas que um novo campo per-model setting toca. Cada slice de settings abaixo deve seguir esta lista completa. Âncoras de linha verificadas pós-sync (v0.6.4, HEAD `e008a66b`):

| Registro | Arquivo | Âncora atual | Papel |
|----------|---------|--------------|-------|
| 1 | `omlx/model_settings.py` | bloco ANE ~L113–140 (docstring), defaults ~L244+ | Declaração `@dataclass` + defaults + docs |
| 2 | `omlx/model_profiles.py` | `MODEL_SPECIFIC_PROFILE_FIELDS` L44–90 | Allowlist de profile per-model |
| 3 | `omlx/admin/routes.py` | `ModelSettingsRequest` L115+, blocos `if "x" in sent` ~L2369–2528 | Schema API + validação de ranges (400) |
| 4 | `omlx/engine_pool.py` | `_engine_runtime_signature` L545+, guard `qwen_ane_active` L604–606 | **Torna o setting reload-trigger** (assinatura de runtime) |
| 5 | `omlx/engine/batched.py` | `_enable_ane_prefill` ~L396–523 | Mapeamento setting → kwargs de `enable_qwen35_ane_prefill` (LLM) |
| 6 | `omlx/engine/vlm.py` | call site ~L1975 | Idem (VLM) |
| 7 | `omlx/admin/benchmark.py` | `_FEATURE_FLAG_SPECS` L237+, `_UPLOADED_SETTING_FIELDS` L327+ | Feature flags + upload leaderboard |
| 8 | `omlx/admin/static/js/dashboard.js` | lista ~L31–46, defaults ~L226–246 | Web editor (opcional, consistente) |
| 9 | `apps/.../Net/DTO/ModelsDTO.swift` | decode L112–127, patch `ModelSettingsPatch` L195–210 | Swift DTO |
| 10 | `apps/.../ViewModels/ModelSettingsScreenVM.swift` | `Field` enum, props, `load()`, `save()`, `currentSettingsDict()` | VM (⚠️ caminho NOVO pós-sync: `AppView/ViewModels/`, não `Screens/`) |
| 11 | `apps/.../Screens/ModelSettingsScreen.swift` | `ExperimentalSection` L1147, gate `isQwen35AnePrefillModel` L1158 | Linha de UI + localização via `defaultValue:` |

Extras: `ProfileSettingsKey` em `apps/.../Utils/ProfileWorkingState.swift` (L145–160); testes Swift `apps/omlx-mac/Tests/oMLXTests/ModelSettingsScreenVMTests.swift`.

### Como o reload funciona (não é preciso fazer nada extra além do registro 4)
`PUT /api/models/{id}/settings` → persiste → compara `_engine_runtime_signature` → se mudou, `requires_reload=True` → unload+reload automático se o modelo estiver pinned. O tuner nunca persiste settings; ele devolve `recommendation` que a UI aplica ao working profile.

---

## 2. Slices de implementação (ordem de execução)

### Slice 1 — Feature A: SwiGLU dentro do programa ANE

**Nativo** (`omlx/custom_kernels/qwen35_prefill/csrc/`):
1. Novo gerador MIL `int8_swiglu_bank_mil` em `qwen35_ane.mm` (próximo a `int8_linear_bank_mil` L441): por procedimento, **dois convs** (gate e up, cada um com seu par de chunks int8-data/int8-scale via `constexpr_blockwise_shift_scale`, offsets no `weight.bin` único usando `append_blob_chunk` L385), seguidos de:
   ```
   silu_out = silu(x=gate)
   act = mul(x=silu_out, y=up)
   } -> (act);
   ```
   (precedente verbatim: `fp16_swiglu_down_mil` L544–556).
2. Novo kernel Metal `qwen35_ane_merge*_act_*` (single/dual) em `qwen35_ane.metal`: copia as linhas `act` do planar ANE; aplica `gate*up/(1+exp(-gate))` (fórmula L142–144) **apenas no sufixo GPU** antes de concatenar. Instantiar para `float16_t`/`bfloat16_t` (padrão L426–465).
3. Exportar em `bindings.cpp` + wrappers `fast.py` — símbolos `qwen35_ane_dual_q4_act_t` / `qwen35_ane_dual_affine_act_t` (+ variantes single). Padrão de fallback em 3 camadas: ABI probe (`_verify_abi`) → `hasattr` por símbolo (RuntimeError só ao invocar) → canaries (`_EXT_HAS_*`). Adicionar nomes ao `NATIVE_SYMBOLS` (fast.py L75–102).

**Python** (`omlx/patches/qwen35_ane_prefill.py`):
4. Campo `swiglu_in_ane: bool = False` em `_AnePrefillConfig` (L86–98). Na seleção de estado/estado de símbolo, quando ativo: usar os novos símbolos `*_act_t` e o merge `act`; o `ane_n` consumido no merge cai à metade (padrão do `fuse_swiglu_` já existente em `qwen35_ane.mm:2514–2518`).
5. **Restrição**: forçar `cpu_fraction = 0` quando ativo (warning + doc). Nesta iteração o merge híbrido ANE-act + CPU-raw + GPU-raw fica fora de escopo.
6. Staging: `AneSwiGLUBankBuilder` novo OU método `add(gate, up)` no `AneLinearBankBuilder` — no fluxo de dual banks (`_enable_dual_procedure_banks`). `warmup()` obrigatório no load (padrão L2745–2807).

**Settings/UI**: `qwen35_ane_prefill_swiglu_in_ane: bool = False` — fio completo dos 11 registros. Linha Toggle na `ExperimentalSection` gated por `vm.qwen35AnePrefillEnabled`.

**Testes** (`tests/test_qwen35_ane_prefill.py`, sem hardware):
- Source-assert no `.mm` via fixture `ane_mm` (L3102): novo MIL contém `silu`/`mul` e termina em `act`; seleção do kernel `act` quando config ativa.
- Planner: `ane_n` pela metade no merge quando ativo.
- **Non-regressão**: com toggle off, seleção de símbolos/estado idêntica à atual (garante zero mudança de comportamento).
- Settings: `test_admin_model_settings.py`, `test_model_settings_profiles.py`, `test_engine_pool.py` (signature), `ModelSettingsScreenVMTests.swift`.
- Testes nativos novos em C++ se aplicável (padrão dos já existentes no repo).

**Validação em hardware** (M3 Ultra): rodar `benchmarks/qwen35_ane_silu_in_ane_poc.py` (criado no slice 5, ou manualmente via harness existente `qwen35_ane_down_fused_poc.py` como template) — comparar single-layer latência vs caminho atual + cosine similarity.

---

### Slice 2 — Feature C: Shared expert de MoE no ANE

**Python** (`omlx/patches/qwen35_ane_prefill.py`):
1. **Path-tracking no scan** (`enable_qwen35_ane_prefill`, scan atual L3208–3221): iterar a árvore com paths (named_modules ou rastreamento de pais). Regras:
   - Módulo com `gate_proj/up_proj/down_proj` **embaixo de um `SparseMoeBlock`** (pai tem `switch_mlp`) → só admitir como candidato se `qwen35_ane_prefill_moe_shared_expert=True` (é o `shared_expert`); senão pular explicitamente.
   - Routed experts (`SwitchGLu` / `switch_mlp`) já são excluídos por ausência estrutural dos atributos — manter.
   - Sem toggle, o pickup que hoje é acidental para de acontecer (comportamento muda apenas para MoE, que não tinha suporte oficial).
2. `_wrap_class`: adicionar `mlx_vlm.models.qwen3_5_moe.language.Qwen3_5MoeMLP` à lista de classes patcheadas (mlx-lm já coberto via `Qwen3NextMLP` = classe do MLP denso compartilhada).
3. **Coordenação com `omlx/patches/qwen35_moe_weighted_sum.py`**: o `_call_shared_expert` (L104–109) chama `shared_expert(x, target_verify)` → deve alcançar o class-wrap ANE. Testar/adicionar teste de ordem de aplicação dos patches (weighted_sum patch + ANE patch).
4. MoE-aware em:
   - `_omlx_ane_mlp_prefill_count` conta shared experts (contadores de expectativa).
   - `expected_operations` em `admin/benchmark.py` `_log_ane_benchmark_trace` (L884+): contagem MoE-aware (layers com MoE contam shared expert × tiles).
   - `ane_prefill_transient_bytes` (patch L3497): branch MoE usando `shared_expert_intermediate_size` do config.
   - `_qwen35_cpu_share_estimated_bytes` (`engine_pool.py` L78–178): rejeitar (retornar 0/None + log claro) CPU-sharing quando o modelo é MoE.
5. `target_verify` já propaga pelo weighted_sum patch; backend ANE já retorna `None` nesse caso — verificar com teste.

**Settings/UI**: `qwen35_ane_prefill_moe_shared_expert: bool = False` — fio completo. Gate de família já passa (`qwen3_5_moe`/`qwen3_6_moe` casam com prefixos `qwen3_5`/`qwen3_6` em routes.py L2369–2380 e `isQwen35AnePrefillModel` Swift). Reutilizar `qwen35_ane_prefill_fraction` e `qwen35_ane_prefill_max_layers` para governar shared experts (menos knobs).

**Testes**: fake model tree MoE (SparseMoeBlock-like: `shared_expert` com os 3 atributos, `switch_mlp` sem eles, `gate`, `shared_expert_gate`):
- Scan admite/nega conforme toggle; routed experts nunca; GDN z elegível em MoE.
- Interação weighted_sum + ANE patch (ordem, dispatch chegando ao wrap).
- Pattern de monkeypatch existente (`fast.qwen35_ane_available`→True, `_compile_pair` recording fakes).

**Validação em hardware**: MoE real (Qwen3.6-35B-A3B oQ4e se disponível localmente) — prefill tok/s vs GPU-only, cosine, determinismo greedy.

---

### Slice 3 — Feature B: o_proj com split de canais

**Python**:
1. `_eligible_oproj(linear)`: `_affine_spec` (q4/q5/q6/q8, group 64/128, sem bias) + shapes (output_dim múltiplo da granularidade dual-ANE; input_dim % group_size == 0).
2. Novo state `_CombinedOProjState` (análogo ao `_CombinedGDNState` L170–184, sem ativação) — **reusando os ops nativos existentes** `qwen35_ane_dual_affine_qmm_t` / `qwen35_ane_q4_affine_qmm_t`, apenas com novo `profile_category` (numerar categoria nova no profiler — feito em conjunto com o slice 4).
3. Scan: camadas de atenção (`Qwen3NextAttention`/`Qwen3_5Attention` — atributo `o_proj`) via path-tracking do slice 2; coletar `o_proj` apenas. **QKV/k/v_proj nunca** (não coletar).
4. Dispatch: mlx-lm → registro no backend registry de lineares do `qwen35_q4_mlp` (padrão `_GDN_MODULES` keyed-by-id, L1001–1004); VLM → estender o hook `_target_verify_linears` (que já dá first-refusal ao ANE, L2103–2114) para o o_proj de atenção.
5. Staging: procedimentos o_proj entram nos dual banks junto com MLP/GDN (banco pequeno: ~hidden×hidden × 16 camadas full-attention).
6. `target_verify` bypass (return None) como todo o resto.

**Settings/UI** (3 campos): `qwen35_ane_prefill_oproj: bool = False`; `qwen35_ane_prefill_oproj_fraction: float = 0.50` (validação 0.05–0.90 em routes.py); `qwen35_ane_prefill_oproj_max_layers: int = 16` (>=1, cap = nº de camadas full-attention). Fio completo + slider + numeric field.

**Testes**: eligibilidade (specs, shapes, gate), scan (o_proj coletado; qkv/k/v não), seleção de state/símbolo (dual_affine com categoria oproj), bypass verify, settings resolution em `enable_qwen35_ane_prefill` (novos kwargs), admin ranges, engine signature, Swift VM.

**Validação em hardware**: 27B denso — B isolado, B+C não aplicável (denso não tem MoE), prefill tok/s vs GPU-only, TTFT, cosine/top-1, memória de pico.

---

### Slice 4 — Tuner: 6ª dimensão (o_proj) + categoria "oproj" no profiler

**Nativo**: registrar categoria `oproj` no profiler — `_ANE_PROFILE_KEYS` em fast.py (L360–375), tuple de categorias `("mlp","gdn")` → `("mlp","gdn","oproj")` no snapshot, threading de `profile_category` nas chamadas de compile.

**`omlx/admin/ane_tuning.py`** (L38 `_GDN_SLOT = 3` é a referência):
1. `_OPROJ_SLOT = 6`; `_oproj_fraction_grid()` (ex.: `[0.0, 0.25, 0.50]`); campos em `_Candidate`/`_CalibrationChoice`; linhas em `_planned_rows()` (L174); contagem de pontos em `create_run()` (L185–216); flag `allow_ane_oproj` em `ANETuningRequest` (L43–66) + rotas `POST /api/bench/ane-tune/start` (routes.py L6738–6806).
2. Calibração sobre o `o_proj` de **uma camada de atenção real** do modelo carregado (estender `_calibrate_components` L1604+): empacotar no banco temporário de procedimento como as dims atuais; pular a dimensão graciosamente quando o modelo não tem camada de atenção elegível (padrão das dims CPU).
3. `_settings_for_candidate` (L418–486) mapeia `qwen35_ane_prefill_oproj*`; `recommendation` (L2223–2242) ganha as chaves oproj; `_profile_refinement` (L722–817) e `_ane_execution_observed` (L506–528) oproj-aware.

**`admin/benchmark.py`**: `ane_trace_config` (L1837–1885) + loop `(category, layer_key, compiled_key)` (L962–965) incluem `oproj`; `expected_operations` conta atenção × tiles.

**Swift**: `ANETuningStartRequest` (+`allowAneOProj`) e `ANETuningRecommendationDTO` (+campos o_proj) em `BenchDTO.swift` (L226–281); menu de overrides ("Allow o_proj on ANE") em `ModelSettingsScreen.swift` (L1176–1191); VM fields (L292–297); `applyANETuningRecommendation()` copiando novos campos ao working profile (VM L884–917).

**Testes**: extensão hermética de `tests/test_ane_tuning.py` (monkeypatch `fast.qwen35_ane_available`/`qwen35_ane_bank_compiler_available`): novo slot/grid/total de pontos, recommendation keys, skip quando sem atenção, `_settings_for_candidate`; `tests/test_benchmark.py` para flags/upload/trace-config novos.

---

### Slice 5 — Benchmarks, docs e integração final

1. **`benchmarks/qwen35_ane_prefill_bench.py`**: flags `--oproj-fraction` (+`--oproj-max-layers`), contadores `oproj` no JSON de saída (records atuais L85–140).
2. **Novo POC** `benchmarks/qwen35_ane_silu_in_ane_poc.py`: single-layer A vs caminho atual (template: `qwen35_ane_down_fused_poc.py`).
3. **Docs** (`docs/experimental/qwen35_ane_prefill.md`): seções novas por feature — mecânica, settings, expectativas honestas (o_proj cobre só 16/64 camadas nas 27B → ganho esperado modesto; MoE = shared expert + GDN z + o_proj), tabela de resultados preenchida **após** validação em hardware, riscos (APIs privadas, aproximação INT8 token-local), interações (A × CPU gate-sharing off; C × CPU-sharing off em MoE).
4. **Transplantar da Orion** a tabela de restrições ANE relevantes (16 BLOBFILEs/programa com custo por op elementwise, ordenação alfabética de superfícies, mínimo 49 KB) como referência do sub-sistema ANE do oMLX — seção docs-only.
5. **Rebuild da extensão nativa**: `OMLX_WITH_CUSTOM_KERNEL=1`, DerivedData apontando para o SSD externo `/Volumes/SSD 4TB/DEV/DerivedData` (instrução do usuário; evita db lock no SSD interno), **sem compilações concorrentes** (esperar build anterior terminar).
6. **Suíte completa**: `pytest -m "not slow and not integration"` + suítes específicas (`test_qwen35_ane_prefill.py`, `test_ane_tuning.py`, `test_admin_model_settings.py`, `test_model_settings_profiles.py`, `test_engine_pool.py`, `test_benchmark.py`).
7. **Gates de aceitação em hardware** (M3 Ultra; 27B validado + MoE A3B real):
   - Cosine similarity hidden/logits ≥ ~0.999; top-1 inalterado em prompts fixos; determinismo em repeated greedy runs.
   - Prefill tok/s vs baseline GPU-only por feature (A, B, C, B+C em MoE), TTFT, memória de pico (delta), com `OMLX_ANE_PROFILE=1` para duty-cycle.
   - Resultados registrados na doc (tabela).
   - Se algum gate falhar (ex.: qualidade), a feature fica default-off com os resultados documentados — não se força a habilitação.

**Commit final**: docs + resultados de validação.

---

## 3. Ordem de execução e definição de pronto por slice

**Ordem:** 1 (A) → 2 (C) → 3 (B) → 4 (tuner) → 5 (bench/docs). A e C são Python+MIL isolados; B depende do path-tracking do scan introduzido em C; tuner por último sobre B estabilizado.

**Cada slice termina com:**
- [ ] Testes verdes: `pytest -m "not slow and not integration"` + suítes específicas do slice.
- [ ] Rebuild do dylib (se slice tocou nativo) e smoke do import (`python -c "from omlx.custom_kernels.qwen35_prefill import fast"`).
- [ ] **Comportamento idêntico ao atual com toggles off** — verificado por teste de non-regressão (seleção de símbolos/estado/estado de símbolo inalterada).
- [ ] Commit individual com mensagem no padrão do repo (ex.: `feat(ane): move SwiGLU into the ANE gate+up program (opt-in)`).
- [ ] Se settings novos: os 11 registros faturados na tabela acima.
- [ ] Doc experimental atualizada (mesmo que incremental).

**Riscos e mitigação:**
| Risco | Mitigação |
|-------|-----------|
| MIL novo não compila no runtime privado (op rejeitada) | Precedente fp16 usa silu/mul em produção; se rejeitar, o fallback é trivial: manter SwiGLU no merge Metal (feature A aborta limpo, sem regredir nada) |
| Quantização INT8 do o_proj degrada qualidade | Fração default conservadora 0.50 + gate de aceitação cosine/top-1; se falhar, diminuir fração ou abortar feature B |
| Aproximação no shared expert degrada MoE | Mesmos gates de qualidade; shared expert é token-local (pós-atenção), mesmo perfil de segurança que MLP denso |
| Janela de ~4 GiB do ANE estoura com o_proj a mais | o_proj é pequeno (~hidden² × 16 camadas); monitorar via contadores `_omlx_ane_resident_program_count`/byte-count no staging do bank |
| APIs privadas quebram em macOS novo | Inerente ao recurso (já documentado); features ficam opt-in e default-off |
| Conflito com patch weighted_sum em MoE | Teste de ordem de aplicação no slice 2; weighted_sum já propaga `target_verify` |
| Compilação concorrente do dylib | Regra do usuário: esperar build anterior; DerivedData no SSD externo |

---

## 4. Fora de escopo (explícito, não implementar)

- D: steering de prefill por request no scheduler (prefill-ANE / decode-GPU simultâneos).
- E: draft model (SpecPrefill/MTP) rodando no ANE.
- Decode no ANE; atenção/SDPA no ANE; CPU-sharing em MoE; router MoE no ANE (nunca — paridade bit-exata da seleção é requisito do repo).

---

## 5. Checklist de handoff (próxima sessão)

- [ ] Branch `feat/ane-oproj-moe-swiglu` já existe e está sincronizado com `main` @ `e008a66b` (v0.6.4).
- [ ] Ler este arquivo completo antes de começar.
- [ ] Slice 1 primeiro; seguir a ordem 1→5.
- [ ] Consultar âncoras de linha via codegraph/grep antes de editar (números de linha podem ter deslocado após novos commits).
- [ ] Não esquecer: `ModelSettingsScreenVM.swift` agora mora em `AppView/ViewModels/` (não `Screens/`).
- [ ] DerivedData no SSD externo; sem builds concorrentes; usar `rtk` como prefixo de shell.
- [ ] Ao final de cada slice: commit individual + atualizar doc experimental.
- [ ] Features D e E: NÃO implementar (ver seção 4).
