# QUANTUM NEURAL NETWORK CON IBM QUANTUM
# Esempio completo: classificazione con QNN su hardware reale IBM

"""
SETUP INIZIALE:
1. Installa i pacchetti necessari:
   pip install qiskit qiskit-ibm-runtime qiskit-machine-learning
   pip install qiskit-aer scikit-learn matplotlib numpy

2. Crea account gratuito su IBM Quantum:
   https://quantum.ibm.com/
   
3. Ottieni il tuo API token da:
   https://quantum.ibm.com/account
"""

import numpy as np
from sklearn.datasets import make_moons, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_aer import AerSimulator

# Qiskit Machine Learning
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# IBM Quantum Runtime
from qiskit_ibm_runtime import QiskitRuntimeService, Estimator, Session

from qiskit_algorithms.optimizers import COBYLA, SPSA


# ============================================================================
# STEP 1: CONFIGURAZIONE IBM QUANTUM
# ============================================================================

def setup_ibm_quantum(api_token=None, use_real_hardware=False):
    """
    Configura la connessione a IBM Quantum.
    
    Args:
        api_token: Il tuo IBM Quantum API token
        use_real_hardware: True per usare hardware reale, False per simulatore
    """
    if api_token:
        # Salva il token (solo la prima volta)
        try:
            QiskitRuntimeService.save_account(
                channel="ibm_cloud",
                token=api_token,
                overwrite=True
            )
        except Exception as e:
            print(f"⚠ Errore salvataggio token: {e}")
    
    # Carica il servizio
    try:
        service = QiskitRuntimeService(channel="ibm_cloud")
        
        if use_real_hardware:
            # Seleziona il backend meno occupato
            backend = service.least_busy(operational=True, simulator=False)
            print(f"✓ Connesso a hardware reale: {backend.name}")
            print(f"  Qubit disponibili: {backend.num_qubits}")
            print(f"  Jobs in coda: {backend.status().pending_jobs}")
        else:
            # Usa simulatore IBM (più veloce per testing)
            backend = service.get_backend('ibmq_qasm_simulator')
            print(f"✓ Usando simulatore IBM: {backend.name}")
        
        return service, backend
    
    except Exception as e:
        print(f"⚠ Errore connessione IBM Quantum: {e}")
        print("→ Usando simulatore locale Aer")
        return None, AerSimulator()


# ============================================================================
# STEP 2: DATASET PER CLASSIFICAZIONE
# ============================================================================

def prepare_dataset(n_samples=100, dataset_type='moons'):
    """
    Prepara dataset per classificazione binaria.
    """
    if dataset_type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    else:
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=2, 
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1,
            random_state=42
        )
    
    # Normalizza in [0, π] per quantum encoding
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X)
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    # Converti labels in {-1, +1} per QNN
    y_train = 2 * y_train - 1
    y_test = 2 * y_test - 1
    
    print(f"\n📊 Dataset preparato:")
    print(f"   Training: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, X, y


# ============================================================================
# STEP 3: COSTRUZIONE QUANTUM NEURAL NETWORK
# ============================================================================

def create_qnn_circuit(n_qubits=2, reps=2):
    """
    Crea il circuito della Quantum Neural Network.
    
    Architettura:
    1. Feature Map: codifica i dati classici in stati quantistici
    2. Variational Circuit: parametri trainabili (pesi della rete)
    """
    # Feature Map: codifica i dati di input
    # ZZFeatureMap crea entanglement tra features
    feature_map = ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=1,
        entanglement='linear'
    )
    
    # Ansatz: circuito parametrico (i "pesi" della QNN)
    # RealAmplitudes è una scelta standard per QNN
    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=reps,
        entanglement='full'
    )
    
    # Combina feature map + ansatz
    qc = QuantumCircuit(n_qubits)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    
    print(f"\n🔮 Quantum Neural Network creata:")
    print(f"   Qubits: {n_qubits}")
    print(f"   Layers variazionali: {reps}")
    print(f"   Parametri trainabili: {ansatz.num_parameters}")
    print(f"   Gates totali: {qc.size()}")
    
    return feature_map, ansatz, qc


# ============================================================================
# STEP 4: TRAINING DELLA QNN
# ============================================================================

def train_qnn(X_train, y_train, backend, feature_map, ansatz):
    """
    Addestra la Quantum Neural Network.
    """
    print("\n🎯 Inizio training della QNN...")
    
    # Observable: misura che vogliamo ottimizzare
    # Usiamo Pauli Z su tutti i qubit
    from qiskit.quantum_info import SparsePauliOp
    observable = SparsePauliOp.from_list([("Z" * feature_map.num_qubits, 1)])
    
    # Crea circuito completo combinando feature_map e ansatz
    qc_complete = QuantumCircuit(feature_map.num_qubits)
    qc_complete.compose(feature_map, inplace=True)
    qc_complete.compose(ansatz, inplace=True)
    
    # Crea la QNN usando Estimator V2 (risolve deprecation warning)
    if isinstance(backend, AerSimulator):
        # Simulatore locale - usa StatevectorEstimator (V2)
        from qiskit.primitives import StatevectorEstimator
        estimator = StatevectorEstimator()
    else:
        # IBM Quantum - usa Estimator V2 con sessione
        session = Session(backend=backend)
        from qiskit_ibm_runtime import EstimatorV2
        estimator = EstimatorV2(session=session)
    
    # Crea QNN con il circuito completo
    qnn = EstimatorQNN(
        circuit=qc_complete,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator
    )
    
    # Callback per monitorare il training
    # Il NeuralNetworkClassifier callback riceve (weights, objective_value)
    training_history = []
    
    def training_callback(weights, objective_value):
        """Callback per NeuralNetworkClassifier"""
        training_history.append(objective_value)
        iteration = len(training_history)
        if iteration % 10 == 0 or iteration == 1:
            print(f"   Iteration {iteration}: Loss = {objective_value:.6f}")
    
    # Ottimizzatore: usa COBYLA dalle nuove API di Qiskit
    # Aumentiamo maxiter e aggiungiamo altre opzioni
    from qiskit_algorithms.optimizers import COBYLA
    optimizer = COBYLA(maxiter=200, tol=1e-6)
    optimizer = SPSA(maxiter=200, learning_rate=0.01, perturbation=0.01)
    
    # Classificatore con QNN
    print(f"   Creazione classificatore...")
    print(f"   Parametri iniziali da ottimizzare: {qnn.num_weights}")
    
    classifier = NeuralNetworkClassifier(
        neural_network=qnn,
        optimizer=optimizer,
        callback=training_callback,
        initial_point=[0,0,0,0,0,0]  # Lascia che sia random
    )
    
    # Training
    print("   Training in corso (può richiedere alcuni minuti)...")
    print("   (Se non vedi iterazioni, potrebbe esserci un problema con il callback)")
    
    try:
        result = classifier.fit(X_train, y_train)
    except Exception as e:
        print(f"   ⚠ Errore durante il training: {e}")
        raise
    
    print(f"✓ Training completato!")
    print(f"   Iterazioni totali: {len(training_history)}")
    
    # Se il callback non è stato chiamato, verifica il fit_result
    if len(training_history) == 0:
        print("   ⚠ Warning: Il callback non è stato chiamato!")
        print("   Questo può succedere se:")
        print("   - L'ottimizzatore converge immediatamente")
        print("   - C'è un problema con la configurazione del callback")
        
        # Prova a ottenere info dal risultato del fit
        if hasattr(classifier, '_fit_result') and classifier._fit_result is not None:
            print(f"   Numero di valutazioni della funzione: {classifier._fit_result.nfev}")
            print(f"   Valore finale: {classifier._fit_result.fun}")
            # Aggiungi almeno il valore finale alla history
            training_history.append(classifier._fit_result.fun)
    else:
        if training_history:
            print(f"   Loss finale: {training_history[-1]:.6f}")
    
    return classifier, training_history


# ============================================================================
# STEP 5: EVALUATION E VISUALIZZAZIONE
# ============================================================================

def evaluate_qnn(classifier, X_train, X_test, y_train, y_test, X_full, y_full, training_history):
    """
    Valuta le performance della QNN e visualizza i risultati.
    """
    # Accuracy
    train_score = classifier.score(X_train, y_train)
    test_score = classifier.score(X_test, y_test)
    
    print(f"\n📈 Performance della QNN:")
    print(f"   Training Accuracy: {train_score:.2%}")
    print(f"   Test Accuracy: {test_score:.2%}")
    
    # Visualizzazione decision boundary
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Dataset
    plt.subplot(1, 3, 1)
    plt.scatter(X_full[:, 0], X_full[:, 1], c=y_full, cmap='coolwarm', 
                edgecolors='k', s=50)
    plt.title('Dataset Originale')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Class')
    
    # Plot 2: Decision Boundary
    plt.subplot(1, 3, 2)
    h = 0.02
    x_min, x_max = X_full[:, 0].min() - 0.1, X_full[:, 0].max() + 0.1
    y_min, y_max = X_full[:, 1].min() - 0.1, X_full[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X_full[:, 0], X_full[:, 1], c=y_full, cmap='coolwarm',
                edgecolors='k', s=50)
    plt.title(f'QNN Decision Boundary\nAccuracy: {test_score:.2%}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Plot 3: Training History
    plt.subplot(1, 3, 3)
    plt.plot(training_history)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qnn_results.png', dpi=150, bbox_inches='tight')
    print("✓ Grafico salvato come 'qnn_results.png'")
    plt.show()


# ============================================================================
# MAIN: ESECUZIONE COMPLETA
# ============================================================================

def main():
    """
    Pipeline completa: setup → data → QNN → training → evaluation
    """
    print("="*70)
    print("QUANTUM NEURAL NETWORK CON IBM QUANTUM")
    print("="*70)
    
    # CONFIGURAZIONE
    # Inserisci il tuo token qui (o None per simulatore locale)
    IBM_TOKEN = "IBMid-6970015RQI"  # Es: "your_token_here" oppure metti None per locale
    USE_REAL_HARDWARE = True  # True per usare computer quantistici reali
    
    # Setup IBM Quantum
    service, backend = setup_ibm_quantum(IBM_TOKEN, USE_REAL_HARDWARE)
    
    # Prepara dataset
    X_train, X_test, y_train, y_test, X_full, y_full = prepare_dataset(
        n_samples=80,  # Riduci per hardware reale (più veloce)
        dataset_type='moons'
    )
    
    # Crea QNN
    feature_map, ansatz, qc = create_qnn_circuit(n_qubits=2, reps=2)
    
    # Visualizza il circuito
    print("\n📐 Struttura del circuito quantistico:")
    print(qc.draw('text'))
    
    # Training
    classifier, training_history = train_qnn(
        X_train, y_train, backend, feature_map, ansatz
    )
    
    # Evaluation
    evaluate_qnn(classifier, X_train, X_test, y_train, y_test, X_full, y_full, training_history)
    
    print("\n" + "="*70)
    print("✓ COMPLETATO!")
    print("="*70)
    
    # Info per usare hardware reale
    if not USE_REAL_HARDWARE:
        print("\n💡 Per usare hardware quantistico IBM reale:")
        print("   1. Vai su https://quantum.ibm.com/")
        print("   2. Crea un account gratuito")
        print("   3. Copia il tuo API token")
        print("   4. Modifica IBM_TOKEN = 'tuo_token' e USE_REAL_HARDWARE=True")
        print("   5. Ri-esegui lo script")
        print("\n⚠️  NOTA: Hardware reale è più lento (minuti/ore per job)")
        print("         ma ti dà accesso a veri computer quantistici!")


if __name__ == "__main__":
    main()


# ============================================================================
# BONUS: CONFRONTO CON NEURAL NETWORK CLASSICA
# ============================================================================

def compare_with_classical():
    """
    Confronta QNN con MLP classico per vedere le differenze.
    """
    from sklearn.neural_network import MLPClassifier
    
    print("\n" + "="*70)
    print("CONFRONTO: QNN vs Classical NN")
    print("="*70)
    
    X_train, X_test, y_train, y_test, _, _ = prepare_dataset(n_samples=80)
    
    # Classical NN
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_score = mlp.score(X_test, y_test)
    
    print(f"\n📊 Risultati:")
    print(f"   Classical NN Accuracy: {mlp_score:.2%}")
    print(f"   Quantum NN Accuracy: (esegui main() prima)")
    print(f"\n💡 Note:")
    print("   - QNN può avere vantaggio su dati ad alta dimensione")
    print("   - Hardware NISQ attuale limita le performance")
    print("   - QNN è più interessante per il potenziale futuro")

# Decommentare per eseguire il confronto
compare_with_classical()