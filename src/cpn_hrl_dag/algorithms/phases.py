"""Explicit hierarchical training-phase semantics.

One trainer serves every phase: a phase only declares which level samples its
own policy, which level follows a fixed rule, and which level may be updated.
Frozen levels are *not* merely skipped in ``optimizer.step``; they stay in
evaluation mode, receive no gradients, and never advance their scheduler.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

HIGH_ACTION_SPACES = ("ppo_sample", "ppo_deterministic", "fixed_heft_rank")
LOW_ACTION_SPACES = ("ppo_sample", "fixed_min_eft")
FROZEN_HIGH_MODES = ("stochastic", "deterministic")

LEGACY_PHASE_ALIASES = {"high_train": "high_only_eft"}


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    """How one training phase chooses tasks and nodes, and who learns."""

    name: str
    high: str
    low: str
    update_high: bool
    update_low: bool
    frozen_high_mode: str | None = None
    trainable: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.high not in HIGH_ACTION_SPACES:
            raise ValueError(f"unknown high-level action space: {self.high!r}")
        if self.low not in LOW_ACTION_SPACES:
            raise ValueError(f"unknown low-level action space: {self.low!r}")
        if self.update_high and self.high != "ppo_sample":
            raise ValueError("updated high level must sample its own PPO policy")
        if self.update_low and self.low != "ppo_sample":
            raise ValueError("updated low level must sample its own PPO policy")
        if self.update_high and self.update_low and (self.high != "ppo_sample" or self.low != "ppo_sample"):
            raise ValueError("jointly updated phases must sample both levels with their own policies")
        if not self.update_high and self.high == "ppo_sample":
            raise ValueError("a non-updated high level may not sample its own policy")
        if not self.update_low and self.low == "ppo_sample":
            raise ValueError("a non-updated low level may not sample its own policy")
        if self.frozen_high_mode is not None and self.frozen_high_mode not in FROZEN_HIGH_MODES:
            raise ValueError(f"unknown frozen high-level execution mode: {self.frozen_high_mode!r}")
        expected = tuple(name for name, flag in (("high", self.update_high), ("low", self.update_low)) if flag)
        if self.trainable and tuple(self.trainable) != expected:
            raise ValueError(f"trainable modules {self.trainable!r} contradict the update flags {expected!r}")

    @property
    def resolved_trainable(self) -> tuple[str, ...]:
        return tuple(name for name, flag in (("high", self.update_high), ("low", self.update_low)) if flag)

    @property
    def frozen(self) -> tuple[str, ...]:
        return tuple(name for name in ("high", "low") if name not in self.resolved_trainable)

    @property
    def high_is_frozen(self) -> bool:
        return "high" in self.frozen

    @property
    def low_is_frozen(self) -> bool:
        return "low" in self.frozen

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["trainable"] = list(self.resolved_trainable)
        payload["frozen"] = list(self.frozen)
        return payload


# The three phase literals requested by the stage-one specification.
HIGH_ONLY_EFT = DecisionPolicy(name="high_only_eft", high="ppo_sample", low="fixed_min_eft", update_high=True, update_low=False)
LOW_ONLY_FROZEN_HIGH_STOCHASTIC = DecisionPolicy(name="low_only_frozen_high", high="ppo_deterministic", low="ppo_sample", update_high=False, update_low=True, frozen_high_mode="stochastic")
LOW_ONLY_FROZEN_HIGH_DETERMINISTIC = DecisionPolicy(name="low_only_frozen_high", high="ppo_deterministic", low="ppo_sample", update_high=False, update_low=True, frozen_high_mode="deterministic")
JOINT = DecisionPolicy(name="joint", high="ppo_sample", low="ppo_sample", update_high=True, update_low=True)
LOW_PRETRAIN = DecisionPolicy(name="low_pretrain", high="fixed_heft_rank", low="ppo_sample", update_high=False, update_low=True)

PHASE_REGISTRY = {
    "high_only_eft": HIGH_ONLY_EFT,
    "low_only_frozen_high": LOW_ONLY_FROZEN_HIGH_DETERMINISTIC,
    "joint": JOINT,
    "low_pretrain": LOW_PRETRAIN,
}


def canonical_phase_name(name: str | bool) -> str:
    """Map a legacy boolean or phase label onto its explicit phase name."""
    if isinstance(name, bool):
        return "low_pretrain" if name else "joint"
    label = str(name)
    return LEGACY_PHASE_ALIASES.get(label, label)


def decision_policy(phase: str | bool | DecisionPolicy, *, frozen_high_mode: str = "deterministic") -> DecisionPolicy:
    """Resolve a phase label to its explicit sampling/update semantics.

    Legacy labels keep their original meaning: ``high_train`` is exactly
    ``high_only_eft`` (high PPO sampling, fixed minimum-EFT low level, low
    network and optimizer untouched).
    """
    if isinstance(phase, DecisionPolicy):
        return phase
    name = canonical_phase_name(phase)
    if name == "low_only_frozen_high":
        if frozen_high_mode not in FROZEN_HIGH_MODES:
            raise ValueError(f"unknown frozen high-level execution mode: {frozen_high_mode!r}")
        return LOW_ONLY_FROZEN_HIGH_STOCHASTIC if frozen_high_mode == "stochastic" else LOW_ONLY_FROZEN_HIGH_DETERMINISTIC
    if name not in PHASE_REGISTRY:
        raise ValueError(f"unknown hierarchical training phase: {name!r}")
    return PHASE_REGISTRY[name]


def parse_phase_schedule(entries: list[dict], *, attribute: str = "training.phases") -> list[dict]:
    """Validate one explicit ``training.phases`` schedule."""
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{attribute} must be a non-empty list of phase mappings")
    schedule: list[dict] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{attribute}[{index}] must be a mapping")
        unknown = set(entry) - {"name", "episodes", "frozen_high_mode"}
        if unknown:
            raise ValueError(f"{attribute}[{index}] has unsupported keys: {sorted(unknown)}")
        if "name" not in entry or "episodes" not in entry:
            raise ValueError(f"{attribute}[{index}] requires both 'name' and 'episodes'")
        episodes = int(entry["episodes"])
        if episodes < 0:
            raise ValueError(f"{attribute}[{index}] episodes must be non-negative")
        if "frozen_high_mode" in entry and canonical_phase_name(entry["name"]) != "low_only_frozen_high":
            raise ValueError("frozen_high_mode is only valid for low_only_frozen_high")
        policy = decision_policy(str(entry["name"]), frozen_high_mode=str(entry.get("frozen_high_mode", "deterministic")))
        schedule.append(dict(name=policy.name, episodes=episodes,
                             frozen_high_mode=(policy.frozen_high_mode if policy.high_is_frozen else None),
                             policy=policy))
    if sum(item["episodes"] for item in schedule) < 1:
        raise ValueError(f"{attribute} must schedule at least one episode")
    return schedule


def default_phase_schedule(config: dict, *, episodes: int | None = None) -> list[dict]:
    """Reproduce the pre-existing staged schedule when no explicit phases exist."""
    training = config["training"]
    low = int(training.get("low_pretrain_episodes", 0))
    high = int(training.get("high_train_episodes", 0))
    joint = int(training.get("joint_train_episodes", 0))
    if episodes is not None and high == 0 and joint == 0:
        joint = int(episodes)
    schedule = []
    for name, count in (("low_pretrain", low), ("high_only_eft", high), ("joint", joint)):
        if count:
            if count < 0:
                raise ValueError("phase episodes must be non-negative")
            schedule.append(dict(name=name, episodes=count, frozen_high_mode=None, policy=decision_policy(name)))
    if not schedule:
        raise ValueError("training configuration schedules no episodes")
    return schedule


def resolve_phase_schedule(config: dict, *, episodes: int | None = None) -> list[dict]:
    """Return the explicit schedule from ``training.phases`` or the legacy budget."""
    entries = config.get("training", {}).get("phases")
    if entries:
        return parse_phase_schedule(entries)
    return default_phase_schedule(config, episodes=episodes)


def summarize_phase_budget(schedule: list[dict]) -> dict:
    """Episode coverage plus a separate count of actual optimizer updates.

    Equal scenario coverage is not equal computation: a phase that only
    activates one level performs fewer optimizer updates.  Both numbers are
    reported so the distinction cannot be silently lost.
    """
    budget = {"episodes": 0, "by_phase": {}, "optimizer_updates": {"high": 0, "low": 0},
              "high_ppo_sampling_episodes": 0, "frozen_high_policy_activation_episodes": 0,
              "low_ppo_sampling_episodes": 0, "fixed_low_rule_episodes": 0,
              "frozen_high_mode_by_phase": {}, "fixed_high_rule_episodes": 0}
    for item in schedule:
        policy = item["policy"]
        episodes = int(item["episodes"])
        budget["episodes"] += episodes
        budget["by_phase"][policy.name] = budget["by_phase"].get(policy.name, 0) + episodes
        for level, active in (("high", policy.update_high), ("low", policy.update_low)):
            if active:
                budget["optimizer_updates"][level] += episodes
        if policy.high == "ppo_sample":
            budget["high_ppo_sampling_episodes"] += episodes
        elif policy.high == "fixed_heft_rank":
            budget["fixed_high_rule_episodes"] += episodes
        else:
            # Frozen high level: it still runs its policy, it just never learns.
            budget["frozen_high_policy_activation_episodes"] += episodes
            budget["frozen_high_mode_by_phase"][policy.name] = policy.frozen_high_mode
        if policy.low == "ppo_sample":
            budget["low_ppo_sampling_episodes"] += episodes
        else:
            budget["fixed_low_rule_episodes"] += episodes
    budget["optimizer_update_total"] = budget["optimizer_updates"]["high"] + budget["optimizer_updates"]["low"]
    budget["update_count_unit"] = "PPO update calls (episodes), not optimizer.step minibatches"
    return budget
