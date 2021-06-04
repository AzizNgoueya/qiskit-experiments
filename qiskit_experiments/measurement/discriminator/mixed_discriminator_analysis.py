"""
Discriminator analysis class for mixed states.
"""

import numpy as np
from qiskit_experiments.base_analysis import BaseAnalysis, AnalysisResult
from .mixed_discriminator_methods import *


try:
    from matplotlib import pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class MixedDiscriminatorAnalysis(BaseAnalysis):
    def _run_analysis(
        self, experiment_data, discriminator_type="LDA", plot: bool = True, distance = None, threshold = None, **options
    ):
        """Run analysis on discriminator data.
        Args:
            experiment_data (ExperimentData): The experiment data to analyze.
            discriminator_type (str): The type of the mixed discriminator LDA, QDA,
            EM, RM and GMM. Default LDA
            plot (bool): If True plot the results
            distance (list): Optional, Coordinates of the boundaries for ellipse
            and rectangle method. You have to specify it for each cluster
            O and 1, and for each qubit. By default the distance is 2*sd
            threshold (list): Optional, the probability(1-p) used to
            options: kwarg options for analysis function.
        Returns:
            tuple: A pair ``(analysis_results, figures)`` where
                ``analysis_results`` may be a single or list of
                AnalysisResult objects, and ``figures`` may be
                None, a single figure, or a list of figures.
        """
        nqubits = len(experiment_data.data(0)["metadata"]["ylabel"])
        discriminator = [None] * nqubits
        score = [None] * nqubits
        if discriminator_type == 'GMM':
            fig, ax = plt.subplots(nqubits, 2)
        else:
            fig, ax = plt.subplots(nqubits)
        fig.tight_layout()
        if nqubits == 1:
            ax = [ax]
        if distance is None:
            distance = [None] * nqubits
    
        elif distance is not None and len(distance) != nqubits:
            raise ValueError('length of the distance must be equal to the number of qubits')
        
        if threshold is None:
            threshold = [None] * nqubits
        elif threshold is not None and len(threshold) != nqubits:
            raise ValueError('length of the threshold must be equal to the number of qubits')
        
        for q in range(nqubits):
            _xdata, _ydata = self._process_data(experiment_data, q)

            if discriminator_type == "LDA" or discriminator_type == "QDA":
                discriminator[q] = GeometricMethodSklearn(threshold[q], discriminator_type = discriminator_type)
    
            elif discriminator_type == "EM" or discriminator_type == "RM":
                 discriminator[q] = GeometricMethod(distance[q], discriminator_type = discriminator_type)
            elif discriminator_type == "GMM":
                discriminator[q] = GaussianMixtureModel(threshold = threshold[q])

            discriminator[q].fit(_ydata, _xdata)

            if plot:
                discriminator[q].plot(_ydata, _xdata, ax[q])
            score[q] = discriminator[q].score(_ydata, _xdata)
            

        if discriminator_type == "LDA":
            analysis_result = AnalysisResult(
                {
                    "discriminator": [d._discriminator for d in discriminator],
                    "coef": [d._discriminator.coef_ for d in discriminator],
                    "intercept": [d._discriminator.intercept_ for d in discriminator],
                    "score": score,
                    "plt": ax
                }
            )

        elif discriminator_type == "QDA":
            analysis_result = AnalysisResult(
                {
                    "discriminator": [d._discriminator for d in discriminator],
                    "rotations": [d._discriminator.rotations_  for d in discriminator],
                    "score": score,
                    "plt": ax
                }
            )
        elif discriminator_type == "EM":
            analysis_result = AnalysisResult(
                {
                    "discriminator": discriminator,
                    "score": score,
                    "plt": ax
                }
            )
        elif  discriminator_type == "RM":
            analysis_result = AnalysisResult(
                {
                    "discriminator": discriminator,
                    "score": score,
                    "plt": ax
                }
            )
        elif  discriminator_type == "GMM":
            analysis_result = AnalysisResult(
                {
                    "discriminator": discriminator,
                    "score": score,
                    "plt": ax
                }
            )

        return analysis_result, None

    def _process_data(self, experiment_data, qubit):
        """Returns x and y data for discriminator on specific qubit."""
        xdata = np.array(
            [int(experiment_data.data(0)["metadata"]["ylabel"][qubit])]
            * len(experiment_data.data(0)["memory"])
        )
        ydata = np.array(experiment_data.data(0)["memory"])[:, qubit, :]
        xdata = np.concatenate(
            (
                xdata,
                [int(experiment_data.data(1)["metadata"]["ylabel"][qubit])]
                * len(experiment_data.data(1)["memory"]),
            )
        )
        ydata = np.concatenate((ydata, np.array(experiment_data.data(1)["memory"])[:, qubit, :]))
        return xdata, ydata