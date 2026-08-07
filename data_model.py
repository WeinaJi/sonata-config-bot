"""
Pydantic v2 data models for the SONATA Simulation Configuration file format.

Spec: https://sonata-extension.readthedocs.io/en/latest/sonata_simulation.html
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class IntegrationMethod(str, Enum):
    euler = "euler"
    crank_nicolson = "crank_nicolson"
    crank_nicolson_ion = "crank_nicolson_ion"


class SpikesSortOrder(str, Enum):
    none = "none"
    by_id = "by_id"
    by_time = "by_time"


class SpikeLocation(str, Enum):
    soma = "soma"
    ais = "AIS"


class ReportType(str, Enum):
    compartment = "compartment"
    summation = "summation"
    synapse = "synapse"
    lfp = "lfp"
    compartment_set = "compartment_set"


class ReportSections(str, Enum):
    soma = "soma"
    axon = "axon"
    dend = "dend"
    apic = "apic"
    all = "all"


class ReportScaling(str, Enum):
    none = "none"
    area = "area"


class ReportCompartments(str, Enum):
    center = "center"
    all = "all"


class ModificationType(str, Enum):
    section_list = "section_list"
    section = "section"
    compartment_set = "compartment_set"
    ttx = "ttx"
    configure_all_sections = "configure_all_sections"


class InputType(str, Enum):
    spikes = "spikes"
    extracellular_stimulation = "extracellular_stimulation"
    current_clamp = "current_clamp"
    voltage_clamp = "voltage_clamp"
    conductance = "conductance"


# ---------------------------------------------------------------------------
# run section
# ---------------------------------------------------------------------------


class Run(BaseModel):
    """Global simulation settings."""

    tstop: float = Field(..., description="Simulation end time in ms.")
    dt: float = Field(..., description="Integration timestep in ms.")
    random_seed: int = Field(..., description="Random seed (positive integer).")
    spike_threshold: float = Field(
        default=-30.0,
        description="Spike detection threshold in mV. Default -30.0 mV.",
    )
    integration_method: IntegrationMethod = Field(
        default=IntegrationMethod.euler,
        description="NEURON/CoreNEURON integration method.",
    )
    stimulus_seed: int = Field(
        default=0,
        description="Seed for noise stimuli. Default 0.",
        ge=0,
    )
    ionchannel_seed: int = Field(
        default=0,
        description="Seed for stochastic ion channels. Default 0.",
        ge=0,
    )
    minis_seed: int = Field(
        default=0,
        description="Seed for Poisson minis processes. Default 0.",
        ge=0,
    )
    synapse_seed: int = Field(
        default=0,
        description="Seed for stochastic synapses. Default 0.",
        ge=0,
    )


# ---------------------------------------------------------------------------
# output section
# ---------------------------------------------------------------------------


class Output(BaseModel):
    """Simulation output parameters."""

    output_dir: str = Field(
        default="output",
        description="Directory for output files (spikes, reports).",
    )
    log_file: Optional[str] = Field(
        default=None,
        description="Console output filename. Default is STDOUT.",
    )
    spikes_file: str = Field(
        default="out.h5",
        description="Filename for action potentials. Default out.h5.",
    )
    spikes_sort_order: SpikesSortOrder = Field(
        default=SpikesSortOrder.by_time,
        description="Sort order for action potentials.",
    )


# ---------------------------------------------------------------------------
# conditions section
# ---------------------------------------------------------------------------


class Modification(BaseModel):
    """One experimental manipulation in the conditions.modifications list."""

    name: str = Field(..., description="Descriptive name for the modification.")
    node_set: Optional[str] = Field(
        default=None,
        description="Node set receiving the manipulation.",
    )
    type: ModificationType = Field(..., description="Manipulation type.")
    section_configure: Optional[str] = Field(
        default=None,
        description="Python snippet for section/segment configuration.",
    )
    compartment_set: Optional[str] = Field(
        default=None,
        description="Compartment set name (used when type=compartment_set).",
    )

    @model_validator(mode="after")
    def check_node_set_or_compartment_set(self) -> "Modification":
        has_node_set = self.node_set is not None
        has_compartment_set = self.compartment_set is not None
        if not has_node_set and not has_compartment_set:
            raise ValueError(
                "Either 'node_set' or 'compartment_set' must be provided in a modification."
            )
        if has_node_set and has_compartment_set:
            raise ValueError(
                "'node_set' and 'compartment_set' are mutually exclusive in a modification."
            )
        return self


# Mechanisms are a free-form dict: { "SUFFIX_NAME": { "var": value, ... } }
MechanismsConfig = Dict[str, Dict[str, Any]]


class Conditions(BaseModel):
    """Global experimental condition parameters."""

    celsius: float = Field(
        default=34.0,
        description="Temperature of experiment in °C.",
    )
    v_init: float = Field(
        default=-80.0,
        description="Initial membrane voltage in mV.",
    )
    spike_location: SpikeLocation = Field(
        default=SpikeLocation.soma,
        description="Spike detection location: 'soma' or 'AIS'.",
    )
    extracellular_calcium: Optional[float] = Field(
        default=None,
        description="Extracellular calcium concentration for synapse scaling.",
    )
    randomize_gaba_rise_time: bool = Field(
        default=False,
        description="Enable legacy GABA_A rise time randomization.",
    )
    mechanisms: Optional[MechanismsConfig] = Field(
        default=None,
        description="GLOBAL variable overrides for synapse/mechanism MOD files.",
    )
    modifications: Optional[List[Modification]] = Field(
        default=None,
        description="Ordered list of circuit modifications.",
    )


# ---------------------------------------------------------------------------
# inputs section — one discriminated union per module
# ---------------------------------------------------------------------------


class _InputBase(BaseModel):
    """Fields shared by every input entry."""

    input_type: InputType
    delay: float = Field(..., description="Activation start time in ms.")
    duration: float = Field(..., description="Duration of the input in ms.")
    node_set: Optional[str] = Field(
        default=None,
        description="Node set affected by input. Mutually exclusive with compartment_set.",
    )
    compartment_set: Optional[str] = Field(
        default=None,
        description="Compartment set from compartment_sets.json. Mutually exclusive with node_set.",
    )

    @model_validator(mode="after")
    def check_target(self) -> "_InputBase":
        has_node_set = self.node_set is not None
        has_compartment_set = self.compartment_set is not None
        if not has_node_set and not has_compartment_set:
            raise ValueError(
                "Exactly one of 'node_set' or 'compartment_set' must be provided."
            )
        if has_node_set and has_compartment_set:
            raise ValueError(
                "'node_set' and 'compartment_set' are mutually exclusive."
            )
        return self


class LinearInput(_InputBase):
    """Continuous current injection (linear ramp)."""

    module: Literal["linear"] = "linear"
    amp_start: float = Field(..., description="Initial current amplitude in nA.")
    amp_end: Optional[float] = Field(
        default=None,
        description="Final current amplitude in nA (interpolated).",
    )
    represents_physical_electrode: bool = Field(default=False)


class RelativeLinearInput(_InputBase):
    """Continuous current injection relative to threshold."""

    module: Literal["relative_linear"] = "relative_linear"
    percent_start: float = Field(
        ..., description="Percentage of threshold current at activation."
    )
    percent_end: Optional[float] = Field(
        default=None,
        description="Percentage of threshold current at conclusion.",
    )
    represents_physical_electrode: bool = Field(default=False)


class PulseInput(_InputBase):
    """Series of current pulse injections."""

    module: Literal["pulse"] = "pulse"
    amp_start: float = Field(..., description="Current amplitude per pulse in nA.")
    width: float = Field(..., description="Pulse duration in ms.")
    frequency: float = Field(..., description="Pulse train frequency in Hz.")
    represents_physical_electrode: bool = Field(default=False)


class SinusoidalInput(_InputBase):
    """Sinusoidal current injection."""

    module: Literal["sinusoidal"] = "sinusoidal"
    amp_start: float = Field(..., description="Peak amplitude in nA.")
    frequency: float = Field(..., description="Waveform frequency in Hz.")
    dt: float = Field(default=0.025, description="Signal timestep in ms.")
    represents_physical_electrode: bool = Field(default=False)


class SubthresholdInput(_InputBase):
    """Continuous current injection adjusted below threshold."""

    module: Literal["subthreshold"] = "subthreshold"
    percent_less: int = Field(
        ...,
        description=(
            "Percentage below 100%% of threshold current. "
            "E.g. 20 → 80%%, -20 → 120%%."
        ),
    )
    represents_physical_electrode: bool = Field(default=False)


class HyperpolarizingInput(_InputBase):
    """Hyperpolarizing holding current injection."""

    module: Literal["hyperpolarizing"] = "hyperpolarizing"
    represents_physical_electrode: bool = Field(default=False)


class SynapseReplayInput(_InputBase):
    """Replay spikes from a file into post-synaptic targets."""

    module: Literal["synapse_replay"] = "synapse_replay"
    spike_file: str = Field(..., description="Path to the .h5 spikes file.")


class SeclampInput(_InputBase):
    """Voltage clamp (SEClamp)."""

    module: Literal["seclamp"] = "seclamp"
    voltage: float = Field(
        ...,
        description="Initial holding voltage in mV (may be overridden by voltage_levels[0]).",
    )
    duration_levels: Optional[List[float]] = Field(
        default=None,
        description="Durations of each voltage step in ms.",
    )
    voltage_levels: Optional[List[float]] = Field(
        default=None,
        description="Holding voltages for each step in mV.",
    )
    series_resistance: float = Field(
        default=0.01,
        description="Series resistance in MΩ.",
    )


class NoiseInput(_InputBase):
    """Continuous current with randomized noise."""

    module: Literal["noise"] = "noise"
    mean: Optional[float] = Field(
        default=None,
        description="Mean current in nA. Mutually exclusive with mean_percent.",
    )
    mean_percent: Optional[float] = Field(
        default=None,
        description="Mean as % of threshold. Mutually exclusive with mean.",
    )
    variance: Optional[float] = Field(
        default=None,
        description="Variance of the normal distribution.",
    )
    represents_physical_electrode: bool = Field(default=False)

    @model_validator(mode="after")
    def check_mean_fields(self) -> "NoiseInput":
        if self.mean is None and self.mean_percent is None:
            raise ValueError("One of 'mean' or 'mean_percent' must be provided.")
        if self.mean is not None and self.mean_percent is not None:
            raise ValueError("'mean' and 'mean_percent' are mutually exclusive.")
        return self


class ShotNoiseInput(_InputBase):
    """Poisson shot noise — base parameters."""

    module: Literal["shot_noise"] = "shot_noise"
    rise_time: float = Field(..., description="Rise time of bi-exponential shots in ms.")
    decay_time: float = Field(..., description="Decay time of bi-exponential shots in ms.")
    rate: float = Field(..., description="Poisson event rate in Hz.")
    amp_mean: float = Field(..., description="Mean of gamma-distributed amplitudes.")
    amp_var: float = Field(..., description="Variance of gamma-distributed amplitudes.")
    reversal: float = Field(default=0.0, description="Reversal potential in mV.")
    dt: float = Field(default=0.25, description="Signal timestep in ms.")
    random_seed: Optional[int] = Field(default=None)
    represents_physical_electrode: bool = Field(default=False)


class RelativeShotNoiseInput(_InputBase):
    """Shot noise with mean/sigma relative to threshold/input-resistance."""

    module: Literal["relative_shot_noise"] = "relative_shot_noise"
    rise_time: float = Field(..., description="Rise time of bi-exponential shots in ms.")
    decay_time: float = Field(..., description="Decay time of bi-exponential shots in ms.")
    mean_percent: float = Field(..., description="Signal mean as % of threshold/input-resistance.")
    sd_percent: float = Field(..., description="Signal std dev as % of threshold/input-resistance.")
    relative_skew: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Signal skewness fraction [0, 1]. Default 0.5.",
    )
    reversal: float = Field(default=0.0, description="Reversal potential in mV.")
    dt: float = Field(default=0.25, description="Signal timestep in ms.")
    random_seed: Optional[int] = Field(default=None)
    represents_physical_electrode: bool = Field(default=False)


class AbsoluteShotNoiseInput(_InputBase):
    """Shot noise with absolute mean/sigma."""

    module: Literal["absolute_shot_noise"] = "absolute_shot_noise"
    rise_time: float = Field(..., description="Rise time of bi-exponential shots in ms.")
    decay_time: float = Field(..., description="Decay time of bi-exponential shots in ms.")
    mean: float = Field(..., description="Signal mean in nA (current_clamp) or uS (conductance).")
    sigma: float = Field(..., description="Signal std dev in nA (current_clamp) or uS (conductance).")
    relative_skew: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Signal skewness fraction [0, 1]. Default 0.5.",
    )
    reversal: float = Field(default=0.0, description="Reversal potential in mV.")
    dt: float = Field(default=0.25, description="Signal timestep in ms.")
    random_seed: Optional[int] = Field(default=None)
    represents_physical_electrode: bool = Field(default=False)


class OrnsteinUhlenbeckInput(_InputBase):
    """Ornstein-Uhlenbeck process with absolute mean/sigma."""

    module: Literal["ornstein_uhlenbeck"] = "ornstein_uhlenbeck"
    tau: float = Field(..., description="Relaxation time constant in ms.")
    mean: float = Field(..., description="Signal mean in nA (current_clamp) or uS (conductance).")
    sigma: float = Field(..., description="Signal std dev in nA (current_clamp) or uS (conductance).")
    reversal: float = Field(default=0.0, description="Reversal potential in mV.")
    dt: float = Field(default=0.25, description="Signal timestep in ms.")
    random_seed: Optional[int] = Field(default=None)
    represents_physical_electrode: bool = Field(default=False)


class RelativeOrnsteinUhlenbeckInput(_InputBase):
    """Ornstein-Uhlenbeck process relative to threshold/input-resistance."""

    module: Literal["relative_ornstein_uhlenbeck"] = "relative_ornstein_uhlenbeck"
    tau: float = Field(..., description="Relaxation time constant in ms.")
    mean_percent: float = Field(..., description="Signal mean as % of threshold/input-resistance.")
    sd_percent: float = Field(..., description="Signal std dev as % of threshold/input-resistance.")
    reversal: float = Field(default=0.0, description="Reversal potential in mV.")
    dt: float = Field(default=0.25, description="Signal timestep in ms.")
    random_seed: Optional[int] = Field(default=None)
    represents_physical_electrode: bool = Field(default=False)


class SpatiallyUniformEFieldInput(_InputBase):
    """Spatially uniform extracellular electric field stimulus."""

    module: Literal["spatially_uniform_e_field"] = "spatially_uniform_e_field"


# Discriminated union — Pydantic dispatches on the "module" field.
AnyInput = Union[
    LinearInput,
    RelativeLinearInput,
    PulseInput,
    SinusoidalInput,
    SubthresholdInput,
    HyperpolarizingInput,
    SynapseReplayInput,
    SeclampInput,
    NoiseInput,
    ShotNoiseInput,
    RelativeShotNoiseInput,
    AbsoluteShotNoiseInput,
    OrnsteinUhlenbeckInput,
    RelativeOrnsteinUhlenbeckInput,
    SpatiallyUniformEFieldInput,
]


# ---------------------------------------------------------------------------
# reports section
# ---------------------------------------------------------------------------


class Report(BaseModel):
    """One data-collection report block."""

    type: ReportType = Field(..., description="Report type.")
    dt: float = Field(..., description="Reporting interval in ms.")
    start_time: float = Field(..., description="Report start time in ms.")
    end_time: float = Field(..., description="Report end time in ms.")

    cells: Optional[str] = Field(
        default=None,
        description="Node set to report. Defaults to the simulation node_set.",
    )
    sections: ReportSections = Field(
        default=ReportSections.soma,
        description="Morphology sections to report.",
    )
    scaling: ReportScaling = Field(
        default=ReportScaling.area,
        description="Density-to-area scaling for summation reports.",
    )
    compartments: Optional[ReportCompartments] = Field(
        default=None,
        description=(
            "Which compartments per section to report. "
            "Defaults to 'center' for soma, 'all' otherwise."
        ),
    )
    variable_name: Optional[str] = Field(
        default=None,
        description=(
            "Simulation variable to record. "
            "Comma-separated list for summation. Not applicable for 'lfp'."
        ),
    )
    unit: Optional[str] = Field(
        default=None,
        description="Descriptive unit label (not validated).",
    )
    file_name: Optional[str] = Field(
        default=None,
        description="Output filename within output_dir. '.h5' added if absent.",
    )
    enabled: bool = Field(
        default=True,
        description="Set to false to suppress this report.",
    )
    compartment_set: Optional[str] = Field(
        default=None,
        description="Compartment set name (required when type='compartment_set').",
    )
    electrodes_file: Optional[str] = Field(
        default=None,
        description="Path to HDF5 electrode weights file (required for 'lfp' type).",
    )

    @model_validator(mode="after")
    def check_cells_or_compartment_set(self) -> "Report":
        has_cells = self.cells is not None
        has_compartment_set = self.compartment_set is not None
        if has_cells and has_compartment_set:
            raise ValueError("'cells' and 'compartment_set' are mutually exclusive in a report.")
        if not has_cells and not has_compartment_set:
            raise ValueError("Exactly one of 'cells' or 'compartment_set' must be provided in a report.")
        return self

    @model_validator(mode="after")
    def check_lfp_and_variable_name(self) -> "Report":
        if self.type == ReportType.lfp:
            if self.variable_name is not None:
                raise ValueError(
                    "'variable_name' is not allowed for 'lfp' report type."
                )
            if self.electrodes_file is None:
                raise ValueError(
                    "'electrodes_file' is mandatory for 'lfp' report type."
                )
        else:
            if self.variable_name is None:
                raise ValueError(
                    "'variable_name' is mandatory for non-'lfp' report types."
                )
        return self

    @model_validator(mode="after")
    def check_compartment_set_type(self) -> "Report":
        if self.type == ReportType.compartment_set and self.compartment_set is None:
            raise ValueError(
                "'compartment_set' is required when type='compartment_set'."
            )
        return self


# ---------------------------------------------------------------------------
# connection_overrides section
# ---------------------------------------------------------------------------


class ConnectionOverride(BaseModel):
    """Adjust synaptic strength or properties for a source→target pair."""

    name: str = Field(..., description="Descriptive name for this override.")
    source: str = Field(..., description="Node set specifying presynaptic nodes.")
    target: str = Field(..., description="Node set specifying postsynaptic nodes.")
    weight: Optional[float] = Field(
        default=None,
        description="Conductance multiplier for synaptic strength.",
    )
    spont_minis: Optional[float] = Field(
        default=None,
        description="Spontaneous mini rate for affected synapses.",
    )
    synapse_configure: Optional[str] = Field(
        default=None,
        description="HOC snippet for synapse objects (use %%s as the synapse reference).",
    )
    modoverride: Optional[str] = Field(
        default=None,
        description="Prefix for the synapse helper file (e.g. 'GluSynapse').",
    )
    synapse_delay_override: Optional[float] = Field(
        default=None,
        description="Override for synaptic delay in ms.",
    )
    delay: Optional[float] = Field(
        default=None,
        description="Apply weight adjustment after this delay in ms.",
    )
    neuromodulation_dtc: Optional[float] = Field(
        default=None,
        description="Neuromodulator decay time constant override in ms.",
    )
    neuromodulation_strength: Optional[float] = Field(
        default=None,
        description="Neuromodulator concentration increase override in µM.",
    )


# ---------------------------------------------------------------------------
# Top-level SimulationConfig
# ---------------------------------------------------------------------------


class SimulationConfig(BaseModel):
    """
    Top-level model for the SONATA Simulation Configuration file.

    Reference: https://sonata-extension.readthedocs.io/en/latest/sonata_simulation.html
    """

    version: Optional[str] = Field(
        default=None,
        description="Config version (current: '2.4').",
    )
    manifest: Optional[Dict[str, str]] = Field(
        default=None,
        description="Path variables used throughout the config (e.g. '${BASE_DIR}').",
    )
    network: Optional[str] = Field(
        default="circuit_config.json",
        description="Path to the circuit configuration file.",
    )
    target_simulator: Optional[str] = Field(
        default=None,
        description="Simulator to use (falls back to circuit_config.json value).",
    )
    node_sets_file: Optional[str] = Field(
        default=None,
        description="Path to a file defining additional node sets.",
    )
    node_set: Optional[str] = Field(
        default=None,
        description="Node set to instantiate. Absence means all non-virtual nodes.",
    )
    compartment_sets_file: Optional[str] = Field(
        default=None,
        description="Path to compartment_sets.json.",
    )
    run: Run = Field(..., description="Mandatory global simulation settings.")
    output: Output = Field(
        default_factory=Output,
        description="Output configuration (optional, has defaults).",
    )
    conditions: Optional[Conditions] = Field(
        default=None,
        description="Global experimental conditions.",
    )
    inputs: Optional[Dict[str, AnyInput]] = Field(
        default=None,
        description="Named stimulus inputs keyed by user-defined name.",
    )
    reports: Optional[Dict[str, Report]] = Field(
        default=None,
        description="Named report blocks keyed by user-defined name.",
    )
    connection_overrides: Optional[List[ConnectionOverride]] = Field(
        default=None,
        description="Ordered list of synaptic property overrides.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Free-form remarks about the simulation (not used for running).",
    )
    beta_features: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Experimental feature flags (not yet production-ready).",
    )

    model_config = {"populate_by_name": True}
