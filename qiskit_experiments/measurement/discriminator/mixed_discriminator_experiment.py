"""
Discriminator Experiment class for mixed states.
"""


from .mixed_discriminator_analysis import MixedDiscriminatorAnalysis
from .discriminator_experiment import DiscriminatorExperiment
from typing import Optional, Union, Iterable
import copy
from qiskit import transpile, assemble
from qiskit.providers.backend import Backend
from qiskit.providers.basebackend import BaseBackend as LegacyBackend
from qiskit_experiments.experiment_data import ExperimentData



class MixedDiscriminatorExperiment(DiscriminatorExperiment):
    """Discriminator Experiment class"""

    # Analysis class for experiment
    __analysis_class__ = MixedDiscriminatorAnalysis

    def __init__(
        self,
        qubits: Union[int, Iterable[int]]
    ):
        """Standard discriminator experiment

        Args:
            qubits: the number of qubits or list of
                    physical qubits for the experiment.
        """
        super().__init__(qubits)

    def run(
        self,
        backend: Backend,
        analysis: bool = True,
        experiment_data: Optional[ExperimentData] = None,
        discriminator_type: str = "LDA",
        plot: bool = True,
        distance: Optional[list] = None,
        threshold: Optional[list] = None,
        **run_options,
    ) -> ExperimentData:
        """Run an experiment and perform analysis.

        Args:
            backend: The backend to run the experiment on.
            analysis: If True run analysis on the experiment data.
            experiment_data: Optional, add results to existing
                experiment data. If None a new ExperimentData object will be
                returned.
            discriminator_type: The type of the mixed discriminator LDA, QDA,
            EM, RM and GMM. Default LDA
            plot: If True plot the results
            distance: Optional, Coordinates of the boundaries for ellipse
            and rectangle method. You have to specify it for each cluster
            O and 1, and for each qubit. By default the distance is 2*sd
            threshold: Optional, the probability(1-p) used to
            classify the mixed states.
            run_options: backend runtime options used for circuit execution.

        Returns:
            The experiment data object.
        """
        # Create new experiment data
        if experiment_data is None:
            experiment_data = self.__experiment_data__(self, backend=backend)
        # Generate and transpile circuits
        circuits = transpile(self.circuits(backend), backend, **self.transpile_options.__dict__)

        # Run circuits on backend
        run_opts = copy.copy(self.run_options)
        run_opts.update_options(**run_options)
        run_opts = run_opts.__dict__

        if isinstance(backend, LegacyBackend):
            qobj = assemble(circuits, backend=backend, **run_opts)
            job = backend.run(qobj)
        else:
            job = backend.run(circuits, **run_opts)

        # Add Job to ExperimentData
        experiment_data.add_data(job)

        # Queue analysis of data for when job is finished
        if analysis and self.__analysis_class__ is not None:
            self.run_analysis(experiment_data, discriminator_type=discriminator_type, plot=plot,
            distance=distance, threshold=threshold)

        # Return the ExperimentData future
        return experiment_data