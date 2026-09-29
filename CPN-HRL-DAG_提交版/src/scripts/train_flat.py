"""Train the Flat PPO small-scale ablation on the shared DAG environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cpn_hrl_dag.algorithms.flat_trainer import FlatPPOTrainer, flat_tensors
from cpn_hrl_dag.algorithms.ppo import PPOAgent
from cpn_hrl_dag.datasets.loading import default_registry, load_scenarios
from cpn_hrl_dag.datasets.resource_realizations import materialize_resources
from cpn_hrl_dag.datasets.split import SplitManager, SplitManifest, scenario_base_key
from cpn_hrl_dag.env.dag_env import CloudEdgeEndDAGEnv
from cpn_hrl_dag.evaluation import Evaluator, summarize, write_report
from cpn_hrl_dag.models.flat import FlatTaskNodeActorCritic
from cpn_hrl_dag.policies.flat import FlatPPOPolicy
from cpn_hrl_dag.utils.config import config_hash, load_config
from cpn_hrl_dag.utils.seed import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    data = config["dataset"]
    scenarios = load_scenarios(default_registry(data.get("grapheonrl_system_configs"), data.get("archive_task_counts")), data["roots"], limit_per_dataset=data.get("limit_per_dataset"))
    manifest_path = Path(data["split_manifest"])
    manifest = SplitManifest.read(manifest_path) if manifest_path.exists() else SplitManager.create((scenario_base_key(item) for item in scenarios), seed=seed)
    splits = materialize_resources(SplitManager.apply(scenarios, manifest), config.get("resources"), manifest, seed)
    device = torch.device(config.get("device", "cpu"))
    probe = CloudEdgeEndDAGEnv(); probe.reset(splits["train"][0]); observation = probe.get_flat_observation()
    model = FlatTaskNodeActorCritic(observation["task_features"].shape[1], observation["resource_features"].shape[1], observation["pair_node_features"].shape[-1], int(config.get("flat", {}).get("hidden_dim", 96))).to(device)
    ppo = config["training"]["ppo"]
    agent = PPOAgent(model, torch.optim.Adam(model.parameters(), lr=float(config.get("flat", {}).get("learning_rate", 5e-4))), lambda item: flat_tensors(item, device), model.forward, device, ppo["gamma_high"], ppo["gae_lambda_high"], ppo["clip_coef"], ppo["entropy_coef_high"], max_grad_norm=ppo["max_grad_norm"], update_epochs=int(ppo.get("update_epochs", 4)), batch_size=int(ppo.get("batch_size", 64)))
    trainer = FlatPPOTrainer(agent, device)
    episodes = args.episodes if args.episodes is not None else int(config.get("flat", {}).get("episodes", 10))
    history = [trainer.episode(splits["train"][index % len(splits["train"])]) for index in range(episodes)]
    output = Path(config.get("flat", {}).get("output_dir", Path(config["output_dir"]) / "flat_ppo"))
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config, "episodes": episodes}, output / "flat.pt")
    records, _ = Evaluator().evaluate(FlatPPOPolicy(model, device), splits["validation"])
    report = summarize(records, model="Flat-PPO", split="validation", seed=seed, config_hash=config_hash(config))
    write_report(output, records, report)
    (output / "train_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"Validation mean_ratio={report['mean_ratio']:.6f}; valid_schedule_rate={report['valid_schedule_rate']:.3f}; output={output}")


if __name__ == "__main__":
    main()
