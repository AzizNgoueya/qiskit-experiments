# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Quantum Volume analysis class.
"""

import math
import warnings
from typing import Optional
import numpy as np

from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.base_analysis import AnalysisResult
from qiskit_experiments.analysis import plotting
from .qv_analysis import QVAnalysis
from copy import deepcopy
import itertools

class MixedQVanalysis(QVAnalysis):
    """Quantum Volume Analysis class."""

    # pylint: disable = arguments-differ
    def _run_analysis(
        self,
        experiment_data,
        simulation_data,
        discriminator,
        rejection = True,
        plot: bool = True,
        ax: Optional["plotting.pyplot.AxesSubplot"] = None,
    ):
        """Run analysis on circuit data.
        Args:
            experiment_data (ExperimentData): the experiment data to analyze.
            simulation_data (ExperimentData): the ideal experiment data to analyze.
            plot: If True generate a plot of fitted data.
            ax: Optional, matplotlib axis to add plot to.
        Returns:
            tuple: A pair ``(analysis_result, figures)`` where
                   ``analysis_results`` may be a single or list of
                   AnalysisResult objects, and ``figures`` may be
                   None, a single figure, or a list of figures.
        """
        depth = experiment_data.experiment.num_qubits
        num_trials = experiment_data.experiment.trials
        data = experiment_data.data()
        ideal_data = simulation_data.data()

        heavy_outputs = np.zeros(num_trials, dtype=list)
        heavy_output_prob_exp = np.zeros(num_trials, dtype=list)

        # analyse ideal data to calculate all heavy outputs
        # must calculate first the ideal data, because the non-ideal calculation uses it
        for ideal_data_trial in ideal_data:
            if not ideal_data_trial["metadata"].get("is_simulation", None):
                continue
            trial = ideal_data_trial["metadata"]["trial"]
            trial_index = trial - 1  # trials starts from 1, so as index use trials - 1

            heavy_outputs[trial_index] = self._calc_ideal_heavy_output(ideal_data_trial)

        # analyse non-ideal data
        for data_trial in data:
            if data_trial["metadata"].get("is_simulation", None):
                continue
            trial = data_trial["metadata"]["trial"]
            trial_index = trial - 1  # trials starts from 1, so as index use trials - 1
            if rejection:
                data_trial['counts'] = self._predict_state(data_trial, discriminator, depth, rejection)
                heavy_output_prob_exp[trial_index] = self._calc_exp_heavy_output_probability(
                data_trial, heavy_outputs[trial_index]
                )
            else:
                heavy_output_prob_exp[trial_index] = self._predict_proba(
                data_trial, discriminator, depth, heavy_outputs[trial_index] )

        analysis_result = AnalysisResult(
            self._calc_quantum_volume(heavy_output_prob_exp, depth, num_trials)
        )

        if plot and plotting.HAS_MATPLOTLIB:
            ax = self._format_plot(ax, analysis_result)
            figures = [ax.get_figure()]
        else:
            figures = None
        return analysis_result, figures


    @staticmethod
    def _predict_state(data, discriminator, num_qubits, rejection =  True):
        counts = {}
        for shots in range(len(data['memory'])):
            skip=False
            digit = [0] * num_qubits
            for qubit in range(num_qubits):
                value = data['memory'][shots][qubit]
                prediction = discriminator['discriminator'][qubit].predict([value])[0]
                if prediction == 2:
                    skip = True
                    break
                digit[num_qubits-qubit-1] = str(prediction)
            if not skip:
                dig_str = "".join(digit)
                if dig_str in counts:
                    counts[dig_str]+=1
                else:
                    counts[dig_str]=1
        return counts
    
    @staticmethod
    def _predict_proba(data, discriminator, num_qubits, heavy_outputs):
        prob_heavy_outputs = []
        for heavy_string in heavy_outputs:
            counts_proba = 0
            for shots in range(len(data['memory'])):
                proba = 1
                for qubit in range(len(heavy_string)):
                    value = int(heavy_string[num_qubits-1-qubit])
                    memory = data['memory'][shots][qubit]
                    proba = proba * discriminator['discriminator']['qubit'].predict(memory, return_proba= True)[value]
                counts_proba += proba
            prob_heavy_outputs.append(counts_proba/len(data['memory']))
        
        return prob_heavy_outputs