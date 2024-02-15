import numpy as np
import pandas as pd
from .utils import *
from .gates import *
from ..run import *
from ..utils import info_print
from qscgrn.qcircuit import cnot_gate, ry_gate


__all__ = ['quantum_circuit']


class quantum_circuit(qscgrn_model):
    """
    Attributes
    ----------
    ngenes : int
        Number of genes in the Quantum GRN.
    genes : list
        Gene list for Quantum GRN modeling.
    theta : pd.Series
        Theta values given edges in the Quantum GRN.
    edges : list
        Edges for the Quantum GRN.
    indexes : list
        Numerical values of the edges for usage in the quantum circuit.
    encoder : np.array
        Matrix representation of the encoder layer `L_enc`.
    regulation : np.array
        Matrix representation of the regulation layers `L_k`. The
        encoder layers are grouped in a single array.
    circuit : np.array
        A single matrix representation for the quantum circuit
        transformation.
    input : np.array
        The quantum state as input for the quantum circuit model.
    derivatives : pd.DataFrame
        Derivatives of the quantum state in the output register
        with respect to each parameter.
    drop_zero : bool
        If True, a normalization step for `p^out` that sets the `|0>_n`
        state to 0, and rescales the rest of the distribution.

    Methods
    -------
    compute_encoder()
        Computes the transformation matrix for the `L_enc` into the
        encoder attribute.
    compute_regulation()
        Computes the transformation matrix for the `L_k` into the
        regulation attribute.
    generate_circuit()
        Compute the transformation matrix for the `L_enc` and `L_k`.
    transform_matrix()
        Compute the transformation matrix for the entire quantum
        circuit.
    output_state()
        Compute the quantum state in the output register given an
        input state.
    output_probabilities(drop_zeros)
        Compute the probability distribution in the output register.
        If drop_zero is True, a normalization step is done.
    create_derivatives()
        Creates a pd.DataFrame to store the derivatives of the
        output state with respect to the parameters.
    der_encoder()
        Computes the derivatives with respect to the parameters
        in the `L_enc` layer.
    der_regulation()
        Computes the derivatives with respect to the parameters
        in the `L_k` layers.
    compute_derivatives()
        Computes the derivatives by calling the der_encoder
        and the der_regulation methods.
    """

    def __init__(self, genes, theta, edges, drop_zero=True):
        """
        Parameters
        ----------
        genes : list
            Gene list for Quantum GRN modeling.
        theta : pd.Series
            Theta values given edges in the Quantum GRN.
        edges : list
            Edges for the Quantum GRN.
        drop_zero : bool
            If True, a normalization step for `p^out` that sets the
            `|0>_n` state to 0, and rescales the rest of the
            distribution.
        """
        super().__init__(genes, theta, edges, drop_zero)
        # numerical indexes are needed to construct the circuit
        self.indexes = edges_to_index(genes, edges)
        # array storage for the quantum circuit (Lenc, Lk and
        # and transformation matrix)
        self.encoder = None
        self.regulation = None
        self.circuit = False
        # parameters for quantum circuit such as input state,
        # derivatives and drop_zero
        self.input = np.zeros((2**self.ngenes, 1))
        self.input[0, 0] = 1.
        self.derivatives = None

    def __str__(self):
        return ("Quantum circuit for {ngenes} genes for GRN"
                " modeling".format(ngenes=len(self.genes)))

    def _circuit_is_empty(self):
        """
        Validates whether the quantum circuit is initialized or not.
        Raises
        ------
        AttributeError
            If circuit attribute is a None object.
        """
        if not self.circuit:
            info_print("The Quantum GRN model is not initialized")
            raise AttributeError("The quantum circuit for GRN model "
                                 "is not constructed")

    def _der_is_not_empty(self):
        """
        Validates if the derivatives for the quantum circuit are
        not initialized.
        Raises
        ------
        AttributeError
            If derivatives is not a None object
        """
        if self.derivatives is not None:
            info_print("Derivatives for the Quantum GRN are already "
                        "initialized", level="E")
            raise AttributeError("The quantum circuit for GRN model "
                                 "has derivatives initialized")

    def _der_is_empty(self):
        """
        Validates if the derivatives for the quantum circuit are
        initialized
        Raises
        ------
        AttributeError
            If derivatives is a None object
        """
        if self.derivatives is None:
            info_print("Derivatives for the Quantum GRN are not "
                        "initialized", level="E")
            raise AttributeError("The quantum circuit for GRN model "
                                 "does not have derivatives "
                                 "initialized")

    def compute_encoder(self):
        """
        Computes the transformation matrices of each gate in `L_enc`
        layer and saves the result into self.encoder
        """
        RR = np.zeros((len(self.genes), 2, 2))

        for idx, gene in enumerate(self.genes):
            RR[idx] = ry_gate(self.theta[(gene, gene)])

        self.encoder = RR

    def compute_regulation(self):
        """
        Computes the transformation matrices of each gate in `L_k`
        layer and saves the result into self.regulation
        """
        arr = np.zeros((len(self.edges), 2**self.ngenes, 2**self.ngenes))

        for i, edge in enumerate(self.edges):
            idx = self.indexes[i]
            control, target = idx[0], idx[1]
            theta_edge = self.theta[edge]

            # Decompose the controlled-Ry gate into CNOT and rotation gates
            cnot = cnot_gate(control, target)
            rotation = ry_gate(theta_edge)
            arr[i] = np.dot(cnot, rotation)

        self.regulation = arr

    def generate_circuit(self):
        """
        Computes the `L_enc` and `L_k` accordingly to parameters
        such as `theta` and edges.
        """
        self.circuit = False
        self.compute_encoder()
        self.compute_regulation()
        self.circuit = True

    def transform_matrix(self):
        """
        Computes the transformation matrix for the quantum circuit
        once `L_enc` and `L_k` are computed.
        Returns
        -------
        T : ndarray
            The transformation matrix for the whole quantum circuit.
        """
        self._circuit_is_empty()
        transform_regulation = matrix_multiplication(self.regulation)
        transform_encoder = tensor_product(self.encoder)
        return np.dot(transform_regulation, transform_encoder)

    def output_state(self):
        """
        Computes the quantum state in the output register of the
        quantum circuit given an input state.
        Returns
        -------
        state : ndarray
            The quantum state in the output register.
        """
        self._circuit_is_empty()
        T = self.transform_matrix()
        return np.dot(T, self.input)

    def output_probabilities(self, drop_zeros=True):
        """
        Computes the probability distribution in the output register
        of the quantum circuit.
        If drop_zeros is True, a normalization step is done.
        Parameters
        ----------
        drop_zeros : bool, optional
            If True, the output probabilities are normalized such
            that the |0>_n state is set to 0.
        Returns
        -------
        probabilities : ndarray
            The probability distribution in the output register.
        """
        state = self.output_state()
        probabilities = np.abs(state)**2

        if drop_zeros:
            probabilities[0] = 0
            probabilities /= probabilities.sum()

        return probabilities

    def create_derivatives(self):
        """
        Creates a pd.DataFrame to store the derivatives of the
        output state with respect to the parameters.
        """
        self._der_is_not_empty()
        self.derivatives = pd.DataFrame(index=self.genes)

    def der_encoder(self):
        """
        Computes the derivatives with respect to the parameters
        in the `L_enc` layer.
        """
        self._circuit_is_empty()
        self._der_is_empty()

        for idx, gene in enumerate(self.genes):
            self.derivatives[gene] = np.dot(self.encoder[idx], self.input)

    def der_regulation(self):
        """
        Computes the derivatives with respect to the parameters
        in the `L_k` layers.
        """
        self._circuit_is_empty()
        self._der_is_empty()

        for i, edge in enumerate(self.edges):
            idx = self.indexes[i]
            control, target = idx[0], idx[1]
            theta_edge = self.theta[edge]

            cnot = cnot_gate(control, target)
            der_cnot = np.zeros_like(cnot)
            rotation = ry_gate(theta_edge)
            der_rotation = der_ry_gate(theta_edge)

            self.derivatives[edge] = np.dot(der_cnot, rotation) + np.dot(cnot, der_rotation)

    def compute_derivatives(self):
        """
        Computes the derivatives by calling the der_encoder
        and the der_regulation methods.
        """
        self.der_encoder()
        self.der_regulation()
