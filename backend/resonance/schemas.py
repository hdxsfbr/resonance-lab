"""Shared data contracts for Resonance Lab.

This module is the single source of truth for the shapes exchanged between the
simulation core, the HTTP API, the experiment runner, the model-provider layer
and the frontend (TypeScript types are generated from the OpenAPI document that
FastAPI derives from these models: `make types`).

Three kinds of quantity are deliberately kept apart everywhere:

1. MEASURED musical features  -> `SymbolicFeatures`, `AudioFeatures`
2. ENGINEERED internal state and learned associations -> `StateVector`,
   `LearnerSnapshot`, `MemoryItem`
3. GENERATED interpretation / narrative -> `NarrativeEvent` payloads, which are
   always labelled `kind="generated_narrative"` and never fed back into state
   updates or measurements.

Nothing in here is a claim about feelings. State variables are engineered
quantities with the operational definitions given on `StateVector`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Music representation
# ---------------------------------------------------------------------------

Instrument = Literal["sine", "triangle", "square", "pluck", "bell", "marimba"]


class Note(BaseModel):
    """One sounding event. Times are in BEATS relative to phrase start.

    Rests are implicit: any span of the phrase without a sounding note is a rest.
    """

    model_config = ConfigDict(extra="forbid")

    pitch: int = Field(ge=0, le=127, description="MIDI pitch number")
    onset: float = Field(ge=0, description="Onset time in beats from phrase start")
    duration: float = Field(gt=0, description="Duration in beats")
    velocity: int = Field(ge=1, le=127, description="MIDI-style velocity (loudness)")


class PhraseOrigin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["agent", "human", "preset", "replay", "symbol", "transformed", "model"] = "agent"
    agent_id: str | None = None
    source_phrase_id: str | None = Field(default=None, description="For replay/transformed phrases")
    transform: str | None = Field(default=None, description="Name of transform applied, if any")


class Phrase(BaseModel):
    """A short structured musical phrase (target length roughly 2-8 seconds)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    notes: list[Note]
    tempo_bpm: float = Field(gt=20, le=400)
    length_beats: float = Field(gt=0, description="Total phrase length in beats, including trailing rest")
    instrument: Instrument = "pluck"
    origin: PhraseOrigin = Field(default_factory=PhraseOrigin)
    motif_id: str | None = Field(default=None, description="Base motif this phrase was generated from, if any")
    tags: list[str] = Field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return self.length_beats * 60.0 / self.tempo_bpm


class TimingPattern(BaseModel):
    """A target rhythm for the cooperative timing task. Public knowledge (the set),
    but WHICH one is the target in an episode is private to the sender."""

    model_config = ConfigDict(extra="forbid")

    id: int
    name: str
    onsets: list[float] = Field(description="Onset times in beats within a 4-beat bar")
    length_beats: float = 4.0


# ---------------------------------------------------------------------------
# Measured features
# ---------------------------------------------------------------------------


class SymbolicFeatures(BaseModel):
    """Features computed deterministically from the NOTE LIST (not from audio)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["symbolic"] = "symbolic"
    note_count: int
    duration_seconds: float
    note_density: float = Field(description="Notes per second")
    rhythmic_regularity: float = Field(description="1 - normalised std of inter-onset intervals, in [0,1]")
    syncopation: float = Field(description="Fraction of onsets not on an eighth-note grid, in [0,1]")
    mean_pitch: float
    pitch_range: int = Field(description="max pitch - min pitch in semitones")
    contour: float = Field(description="Normalised slope of pitch over time, in [-1,1]")
    contour_class: Literal["rising", "falling", "arch", "valley", "flat", "zigzag"]
    repetition: float = Field(description="Fraction of repeated (interval, ioi) bigrams, in [0,1]")
    interval_histogram: list[float] = Field(
        description="Fractions of melodic intervals by class: [unison, step(1-2), third(3-4), fourth/fifth(5-7), large(>7)]"
    )
    mean_velocity: float
    velocity_variation: float = Field(description="std(velocity)/mean(velocity)")
    dissonance_proxy: float = Field(
        description="Fraction of melodic intervals whose pitch-class interval is in {1, 6, 11} semitones (minor 2nd, tritone, major 7th). Defined here; not a perceptual claim."
    )
    tempo_bpm: float
    vector: list[float] = Field(description="Normalised feature vector phi(phrase) used by learners (fixed order, see FEATURE_NAMES)")


FEATURE_NAMES: list[str] = [
    "note_density",
    "rhythmic_regularity",
    "syncopation",
    "mean_pitch",
    "pitch_range",
    "contour",
    "repetition",
    "iv_unison",
    "iv_step",
    "iv_third",
    "iv_fourth_fifth",
    "iv_large",
    "mean_velocity",
    "velocity_variation",
    "dissonance_proxy",
    "tempo",
]


class AudioFeatures(BaseModel):
    """Features MEASURED from a rendered waveform of the phrase (server-side synth)."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["audio"] = "audio"
    sample_rate: int
    duration_seconds: float
    rms: float
    peak: float
    spectral_centroid_hz: float
    onset_count_estimate: int = Field(description="Energy-envelope onset estimate; may differ from note_count")
    synth_version: str = Field(description="Identifier of the server synth used; browser playback uses a separate Web Audio implementation of the same instrument spec")


class PhraseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phrase_id: str
    symbolic: SymbolicFeatures
    audio: AudioFeatures | None = None


# ---------------------------------------------------------------------------
# Engineered agent state, memory, learning
# ---------------------------------------------------------------------------


class StateVector(BaseModel):
    """Bounded engineered state. Operational definitions:

    activation      [0,1]  drive/arousal-like scalar. Raised by dense/loud input (if the
                           hand-authored acoustic influence is enabled) and by surprising
                           outcomes; decays toward baseline. Modulates expressive tempo/velocity
                           and exploration temperature when coupling is enabled.
    expected_value  [0,1]  running expectation of the coordination score.
    uncertainty     [0,1]  running magnitude of recent prediction error.
    affiliation    [-1,1]  running credit toward the partner: rises with shared success,
                           falls with shared failure. Scales effective learning rate when coupled.
    """

    model_config = ConfigDict(extra="forbid")

    activation: float = Field(ge=0, le=1)
    expected_value: float = Field(ge=0, le=1)
    uncertainty: float = Field(ge=0, le=1)
    affiliation: float = Field(ge=-1, le=1)


STATE_DIMS: list[str] = ["activation", "expected_value", "uncertainty", "affiliation"]


class StateUpdateInputs(BaseModel):
    """Everything that fed one state update, so the inspector can show cause -> effect."""

    model_config = ConfigDict(extra="forbid")

    music_drive: dict[str, float] = Field(
        default_factory=dict, description="Hand-authored acoustic contribution per state dim (0 if disabled)"
    )
    prediction_error: float | None = None
    outcome: float | None = None
    partner_similarity: float | None = Field(
        default=None, description="Cosine similarity of partner's phrase features to own last phrase (0..1)"
    )
    decay_applied: bool = True
    frozen: bool = False
    notes: list[str] = Field(default_factory=list)


class MemoryItem(BaseModel):
    """One episodic memory of a musical interaction and its outcome."""

    model_config = ConfigDict(extra="forbid")

    step: int
    role: Literal["sender", "receiver"]
    partner_id: str
    phrase_id: str
    motif_id: str | None = None
    feature_vector: list[float]
    action: int = Field(description="motif index (sender) or pattern id (receiver)")
    score: float
    state_before: StateVector
    state_after: StateVector


class RetrievedMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: MemoryItem
    similarity: float


class LearnerSnapshot(BaseModel):
    """Inspectable learned associations.

    receiver_weights: one weight vector per timing pattern over FEATURE_NAMES (+ bias).
    sender_values:    Q[target_id][motif_index] expected score.
    """

    model_config = ConfigDict(extra="forbid")

    receiver_weights: list[list[float]]
    receiver_bias: list[float]
    sender_values: list[list[float]]
    sender_counts: list[list[int]]
    updates: int = Field(description="Number of learning updates applied")


class PolicyTrace(BaseModel):
    """What the policy computed for one decision (scores -> probabilities)."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["sender", "receiver"]
    policy_kind: Literal["local", "model", "human", "replay"]
    scores: list[float]
    probabilities: list[float]
    temperature: float
    temperature_base: float
    coupling_enabled: bool
    chosen: int
    exploration_draw: float | None = None
    expressive_modulation: dict[str, float] = Field(
        default_factory=dict, description="e.g. tempo_multiplier, velocity_offset actually applied"
    )
    notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class StateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline: StateVector = StateVector(activation=0.5, expected_value=0.5, uncertainty=0.5, affiliation=0.0)
    inertia: dict[str, float] = Field(
        default_factory=lambda: {"activation": 0.6, "expected_value": 0.8, "uncertainty": 0.7, "affiliation": 0.85},
        description="Fraction of the previous value retained per step; drive is scaled by (1 - inertia)",
    )
    decay: dict[str, float] = Field(
        default_factory=lambda: {"activation": 0.15, "expected_value": 0.02, "uncertainty": 0.05, "affiliation": 0.03},
        description="Per-step pull toward baseline",
    )
    max_step: dict[str, float] = Field(
        default_factory=lambda: {"activation": 0.3, "expected_value": 0.3, "uncertainty": 0.3, "affiliation": 0.3},
        description="Bound on |delta| per step",
    )
    acoustic_activation_enabled: bool = Field(
        default=True,
        description="HAND-AUTHORED: dense/loud phrases push activation up. Switchable; not a learned association.",
    )
    acoustic_activation_gain: float = 0.5
    prediction_error_gain: dict[str, float] = Field(
        default_factory=lambda: {"activation": 0.4, "uncertainty": 1.0}
    )
    outcome_gain: dict[str, float] = Field(default_factory=lambda: {"expected_value": 1.0, "affiliation": 0.8})
    partner_similarity_gain: float = Field(
        default=0.2, description="HAND-AUTHORED: similar partner phrase nudges affiliation up"
    )


class CouplingConfig(BaseModel):
    """How state influences the policy. All effects are off when enabled=False."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    uncertainty_to_temperature: float = Field(default=1.0, description="tau *= 1 + k*(U - 0.5)")
    activation_to_temperature: float = Field(default=0.6, description="tau *= 1 + k*(0.5 - A)")
    activation_to_tempo: float = Field(default=0.3, description="tempo_mult = 1 + k*(A - 0.5)")
    activation_to_velocity: float = Field(default=30.0, description="velocity += k*(A - 0.5)")
    affiliation_to_learning_rate: float = Field(default=0.5, description="lr *= 1 + k*affiliation")
    temperature_min: float = 0.05
    temperature_max: float = 5.0


class LearningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    receiver_learning_rate: float = 0.3
    sender_learning_rate: float = 0.3
    temperature_base: float = 0.12
    receiver_l2: float = 0.001
    optimistic_init: float = Field(default=0.0, description="Initial sender Q value")
    receiver_init_scale: float = Field(default=0.0, description="Scale of random init for receiver weights")


class MemoryConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capacity: int = Field(default=200, ge=0)
    retrieval_k: int = 5


class MusicConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phrase_length_beats: float = 8.0
    tempo_min: float = 80.0
    tempo_max: float = 160.0
    motif_bank: str = Field(default="default", description="Named motif bank in resonance/music/motifs.py")
    n_motifs: int = Field(default=8, description="Sender action space size")
    feature_weights: dict[str, float] = Field(
        default_factory=dict, description="Optional per-feature multipliers applied to phi(phrase); default 1.0"
    )
    generation_noise: float = Field(default=0.05, description="Velocity/timing jitter sd applied by rng_gen")


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["timing"] = "timing"
    n_patterns: int = Field(default=4, ge=2, le=8)
    partial_credit: bool = True


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["none", "scripted", "llamacpp", "anthropic", "openai_compatible"] = "none"
    model_id: str = ""
    base_url: str | None = None
    api_key_env: str = Field(default="", description="Name of env var holding the key; the key itself is never stored")
    timeout_seconds: float = 20.0
    max_retries: int = 1
    call_budget: int = Field(default=200, description="Max model calls per session; then fall back to local policy")
    temperature: float = 0.2
    use_for: list[Literal["sender", "receiver", "narrative"]] = Field(default_factory=list)
    narrative_enabled: bool = False
    prompts_dir: str = "prompts"
    local_model_path: str = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    n_threads: int = 4


class ConditionName(str, Enum):
    full = "full"
    no_history = "no_history"
    state_fixed = "state_fixed"
    state_decoupled = "state_decoupled"
    symbol = "symbol"
    perturbed = "perturbed"


Perturbation = Literal[
    "none",
    "transpose",  # +5 semitones; preserves intervals, contour, rhythm (expressive-structure change)
    "velocity_flatten",  # all velocities -> 80; removes dynamics
    "tempo_shift",  # tempo * 1.25; changes notes/sec but not beat-relative structure
    "contour_invert",  # mirror pitches around mean; destroys contour, keeps rhythm
    "rhythm_shuffle",  # permute inter-onset intervals; destroys rhythmic message content
    "pitch_shuffle",  # permute pitch order; destroys contour and interval sequence
]


class ConditionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ConditionName = ConditionName.full
    perturbation: Perturbation = "none"
    description: str = ""


class ExperimentConfig(BaseModel):
    """Everything a run depends on. Recorded verbatim with each run."""

    model_config = ConfigDict(extra="forbid")

    seed: int = 7
    episodes: int = 120
    state: StateConfig = Field(default_factory=StateConfig)
    coupling: CouplingConfig = Field(default_factory=CouplingConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    music: MusicConfig = Field(default_factory=MusicConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    agents: list[AgentParams] = Field(default_factory=lambda: [AgentParams(id="A", name="Aria"), AgentParams(id="B", name="Bram")])


class AgentParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    color: str = "#7c9cff"
    instrument: Instrument = "pluck"
    baseline_override: StateVector | None = None
    sensitivity: float = Field(default=1.0, description="Multiplier on all state drives")
    policy_kind: Literal["local", "model"] = "local"


ExperimentConfig.model_rebuild()


# ---------------------------------------------------------------------------
# Events (append-only log; the export is a list of these plus the config)
# ---------------------------------------------------------------------------

EventType = Literal[
    "session_created",
    "target_assigned",
    "phrase_sent",
    "phrase_received",
    "action_chosen",
    "outcome",
    "state_update",
    "learning_update",
    "intervention",
    "model_call",
    "generated_narrative",
    "episode_complete",
    "session_reset",
]


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seq: int
    step: int = Field(description="Episode index this event belongs to")
    t: str = Field(description="ISO-8601 wall-clock timestamp (not used for determinism)")
    type: EventType
    agent_id: str | None = None
    visibility: Literal["public", "sender_private", "experimenter"] = "public"
    payload: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """EXACTLY what the receiver gets. Must never contain the target."""

    model_config = ConfigDict(extra="forbid")

    channel: Literal["music", "symbol"]
    phrase: Phrase | None = Field(default=None, description="Present for music channel (possibly perturbed)")
    features: SymbolicFeatures | None = None
    symbol_id: int | None = Field(default=None, description="Present for symbol channel: identity of sender's message")
    sender_id: str
    step: int


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


class AgentSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    color: str
    instrument: Instrument
    params: AgentParams
    state: StateVector
    baseline: StateVector
    memory_size: int
    learner: LearnerSnapshot
    last_trace: PolicyTrace | None = None
    frozen_state: bool = False
    coupling_enabled: bool = True
    policy_kind: Literal["local", "model"] = "local"


class EpisodeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: int
    sender_id: str
    receiver_id: str
    target_id: int
    chosen_pattern_id: int | None = None
    score: float | None = None
    phrase: Phrase | None = None
    observation: Observation | None = None


class SessionMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episodes: int
    mean_score: float
    rolling_score: float = Field(description="Mean over last 20 episodes")
    success_rate: float = Field(description="Fraction with score == 1.0")
    score_history: list[float]


class SessionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    created_at: str
    mode: Literal["live", "replay", "demo"]
    preset: str | None = None
    seed: int
    condition: ConditionSpec
    config: ExperimentConfig
    step: int
    status: Literal["ready", "running", "finished"]
    agents: list[AgentSnapshot]
    current_episode: EpisodeSummary | None = None
    metrics: SessionMetrics
    event_count: int
    patterns: list[TimingPattern]


class SessionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    created_at: str
    mode: Literal["live", "replay", "demo"]
    preset: str | None
    seed: int
    condition: ConditionSpec
    step: int
    status: Literal["ready", "running", "finished"]


class CreateSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: ExperimentConfig | None = None
    condition: ConditionSpec = Field(default_factory=ConditionSpec)
    preset: str | None = None


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int = Field(default=1, ge=1, le=500)


class StepResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[Event]
    snapshot: SessionSnapshot


class Intervention(BaseModel):
    """Experimenter interventions. Each one is logged as an `intervention` event
    with `payload.effective_from_step` so the UI can show when it took effect.

    kinds:
      replay_motif       phrase_id (or phrase) -> used as the NEXT sent phrase regardless of sender policy
      reset_memory       agent_id -> clears episodic memory AND learned associations; state vector untouched
      reset_state        agent_id -> state vector := baseline; memory and associations untouched
      freeze_state       agent_id, value -> state updates skipped while frozen
      set_coupling       agent_id, value -> state may evolve but cannot influence policy when False
      swap_feature       transform (Perturbation) -> applied to every subsequent sent phrase until 'none'
      set_param          path (dotted config path), value -> live config change (e.g. state.decay.activation)
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal[
        "replay_motif",
        "reset_memory",
        "reset_state",
        "freeze_state",
        "set_coupling",
        "swap_feature",
        "set_param",
    ]
    agent_id: str | None = None
    value: bool | float | int | str | None = None
    phrase_id: str | None = None
    phrase: Phrase | None = None
    transform: Perturbation | None = None
    path: str | None = None


class HumanPhraseRequest(BaseModel):
    """A human acts as sender for one episode."""

    model_config = ConfigDict(extra="forbid")

    phrase: Phrase
    target_id: int | None = Field(default=None, description="If None, a target is drawn by the environment")
    receiver_id: str | None = None


class AgentInspection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: AgentSnapshot
    memory: list[MemoryItem]
    retrieved: list[RetrievedMemory] = Field(default_factory=list, description="For the most recent observation")
    last_state_inputs: StateUpdateInputs | None = None
    state_history: list[StateVector]
    information_received: list[dict[str, Any]] = Field(
        default_factory=list, description="Log of what this agent actually observed, per step"
    )


class RunExport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format_version: str = "1"
    exported_at: str
    session: SessionSummary
    config: ExperimentConfig
    condition: ConditionSpec
    events: list[Event]
    final_snapshot: SessionSnapshot


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


class RunMetrics(BaseModel):
    """Per-run (one seed, one condition) metrics. Runs are the unit of analysis;
    turns within a run are correlated and are NOT treated as independent samples."""

    model_config = ConfigDict(extra="forbid")

    seed: int
    condition: str
    perturbation: str = "none"
    episodes: int
    mean_score: float
    final_block_score: float = Field(description="Mean score over the last 20% of episodes")
    first_block_score: float = Field(description="Mean score over the first 20% of episodes")
    success_rate: float
    learning_curve: list[float] = Field(description="Block means (10 blocks)")
    policy_shift_kl: dict[str, float] = Field(description="KL(final policy || initial policy) per agent, averaged over contexts")
    state_effect_persistence: dict[str, float] = Field(
        description="Steps for activation to return within 0.05 of baseline after a probe pulse, per agent"
    )
    familiar_vs_unfamiliar: dict[str, float] = Field(
        description="max policy prob for familiar (stored) vs transformed/novel motifs, per agent"
    )
    generalization_score: float | None = Field(default=None, description="Score on transposed motifs at test")
    state_similarity: float | None = Field(default=None, description="Correlation of agents' activation trajectories")
    extra: dict[str, float] = Field(default_factory=dict)


class ConditionAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition: str
    perturbation: str = "none"
    n_runs: int
    mean_final_block_score: float
    sd_final_block_score: float
    mean_success_rate: float
    sd_success_rate: float
    mean_learning_curve: list[float]
    sd_learning_curve: list[float]
    per_run_final_block: list[float]


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "batch"
    conditions: list[ConditionSpec]
    seeds: list[int]
    config: ExperimentConfig | None = None


class ExperimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    created_at: str
    config: ExperimentConfig
    runs: list[RunMetrics]
    aggregates: list[ConditionAggregate]
    notes: list[str] = Field(default_factory=list, description="Caveats and honest framing; not conclusions")


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


class PresetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    title: str
    summary: str
    manipulation: str
    evidence_criterion: str
    constructed_note: str = Field(default="", description="If the result is built by the training schedule, say so")
    default_episodes: int
    condition: ConditionSpec


class PresetResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preset: str
    seed: int
    sessions: list[SessionSummary]
    comparison: dict[str, Any] = Field(description="Preset-specific measured comparison (documented per preset)")
    narrative: str | None = Field(default=None, description="Generated or templated explanation, labelled as such")


# ---------------------------------------------------------------------------
# Model provider layer
# ---------------------------------------------------------------------------


class ModelRequest(BaseModel):
    """What is sent to a model. `input_modality` must be truthful."""

    model_config = ConfigDict(extra="forbid")

    purpose: Literal["choose_pattern", "choose_motif", "propose_phrase", "narrative"]
    system_prompt: str
    user_prompt: str
    json_schema: dict[str, Any] | None = None
    input_modality: Literal["symbolic_features", "symbolic_notes", "audio"] = "symbolic_features"
    max_tokens: int = 256
    temperature: float = 0.2


class ModelResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    provider: str
    model_id: str
    content: str = ""
    parsed: dict[str, Any] | None = None
    latency_ms: float = 0.0
    error: str | None = None
    usage: dict[str, int] = Field(default_factory=dict)
    tested_in_this_environment: bool = Field(
        default=False, description="True only for providers that were exercised against a real model here"
    )


class ModelStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str
    model_id: str
    available: bool
    tested_in_this_environment: bool
    accepts_audio: bool
    input_modality: Literal["symbolic_features", "symbolic_notes", "audio", "none"]
    detail: str = ""
    calls_made: int = 0
    call_budget: int = 0


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    version: str
    mode_note: str = "Engineered agent simulation. State labels are operational definitions, not claims about experience."
    model: ModelStatus


class SavedMotif(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    phrase: Phrase
    created_at: str
    features: SymbolicFeatures | None = None
