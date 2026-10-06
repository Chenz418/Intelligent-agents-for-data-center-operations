Run the following commands from the repository root using Python 3.11 or 3.12.

1. Install the evaluation dependencies:

   ```bash
   python3.12 -m venv .venv
   .venv/bin/python -m pip install -e .
   ```

   For StateBundle evaluation, also install PyTorch:

   ```bash
   .venv/bin/python -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cpu
   ```

2. Configure the semantic judge required for Detection, Localization, and RCA.
   Select a model and endpoint supporting Chat Completions with strict JSON
   Schema output:

   ```bash
   export OPENAI_API_KEY='your-judge-api-key'
   export DC_TWIN_SEMANTIC_API_KEY_ENV=OPENAI_API_KEY
   export DC_TWIN_SEMANTIC_MODEL='YOUR_JUDGE_MODEL'
   export DC_TWIN_SEMANTIC_BASE_URL='https://api.openai.com/v1'
   ```

   Set these variables in the shell; `.env` files are not loaded automatically.
   If needed, add `--semantic-evaluator-reasoning-effort` and
   `--semantic-evaluator-timeout-seconds` to either evaluation command below.

3. Select the observation condition before running either agent.

   For Full-Canonical:

   ```bash
   export OBSERVATION=full-canonical
   ```

   For StateBundle:

   ```bash
   export OBSERVATION=statebundle
   ```

   StateBundle uses the bundled configuration and checkpoint by default. To
   change its configuration, checkpoint, or evidence-token budget, add these
   options to the evaluation command with your chosen values:

   ```text
   --statebundle-config configs/statebundle.dc_twin.yaml
   --statebundle-checkpoint checkpoints/statebundle-stage3/stage3-epoch-9.pt
   --observation-token-budget 4096
   ```

4. Evaluate a tool-calling agent on all 72 tasks:

   ```bash
   export DC_TWIN_LLM_API_KEY='your-agent-api-key'

   .venv/bin/python scripts/evaluate_data_center_twin.py \
     --agent-type tool-calling \
     --model YOUR_AGENT_MODEL \
     --provider openai \
     --base-url https://api.openai.com/v1 \
     --api-key-env DC_TWIN_LLM_API_KEY \
     --reasoning-effort medium \
     --observation "$OBSERVATION" \
     --expected-problem-count 72 \
     --output-dir "results/tool-calling-$OBSERVATION"
   ```

   Replace the model, endpoint, and provider for your agent. Supported provider
   values are `openai`, `openai_compatible`, `gemini_native`, and `dashscope`.
   Set or omit `--reasoning-effort` according to your provider's requirements.
   Adjust `--max-tokens`, `--temperature`, `--thinking-mode`, or
   `--use-max-completion-tokens` when required by the model.

5. Evaluate Codex on all 72 tasks. Install Docker, then build the agent image
   with your selected Codex CLI version:

   ```bash
   docker build -f containers/codex.Dockerfile \
     --build-arg CODEX_VERSION=YOUR_CODEX_CLI_VERSION \
     -t dcbench-codex:local .
   ```

   Configure the Codex credential and run the evaluation using the observation
   condition selected above:

   ```bash
   export CODEX_API_KEY='your-OpenAI-api-key'

   .venv/bin/python scripts/evaluate_data_center_twin.py \
     --agent-type codex \
     --observation "$OBSERVATION" \
     --codex-sandbox docker \
     --codex-docker-image dcbench-codex:local \
     --codex-env CODEX_API_KEY \
     --codex-command 'codex exec --ephemeral --skip-git-repo-check --dangerously-bypass-approvals-and-sandbox --model YOUR_CODEX_MODEL "Read TASK.md and complete the task using dc_twin_tool.py."' \
     --expected-problem-count 72 \
     --output-dir "results/codex-$OBSERVATION"
   ```

   Configure the Codex model and provider inside `--codex-command`. Forward any
   additional required credential variables with repeated `--codex-env VARIABLE`
   options. Keep `--codex-sandbox docker` with the command above so Docker
   provides the execution isolation.

6. To evaluate a small diagnostic/mitigation subset, replace
   `--expected-problem-count 72` in either command with:

   ```text
   --problem-filter '^data_center_twin-cooling_degradation-(detection|mitigation)-1$'
   --expected-problem-count 2
   ```

   Set `--max-steps` and `--timeout-seconds` to change the per-episode limits.
   Read `results.json`, `results.csv`, and `results.md` in the selected output
   directory after the evaluation. Use a separate output directory for each
   model and observation condition.
