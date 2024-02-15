import numpy as np
import pandas as pd
from .utils import *
from .gates import *
from ..run import *
from ..utils import info_print


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
        arr = np.zeros((len(self.edges),
                        2**self.ngenes, 2**self.ngenes))

        for i, edge in enumerate(self.edges):
            idx = self.indexes[i]
            # Decompose C-RY gate into CNOT and rotation gates
            arr[i] = cnot_gate(idx[0], idx[1]).dot(ry_gate(self.theta[edge]))

        self.regulation = arr

    def generate_circuit(self):
        """
        Computes the `L_enc` and `L_k` accordingly to parameters
        such as `theta` and edges
