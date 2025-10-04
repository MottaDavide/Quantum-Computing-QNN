# ⚛️ Quantum Neural Network (QNN) per Classificazione con Qiskit

Questo repository contiene un esempio completo di implementazione di una **Quantum Neural Network (QNN)** per la classificazione binaria, utilizzando il framework **Qiskit** e i suoi moduli di Machine Learning. Il codice è progettato per eseguire il training su un simulatore locale (`AerSimulator`) o su hardware quantistico reale di IBM (se configurato).

Viene inoltre fornito un confronto con un modello di **Neural Network Classica (MLP)** per valutare le differenze prestazionali.

---

## 🚀 Pipeline Esecutiva (Passo dopo Passo)

Il codice segue una pipeline logica suddivisa in 5 passaggi principali:

### STEP 1: Configurazione IBM Quantum
- **Obiettivo:** Stabilire la connessione con l'ambiente IBM Quantum per selezionare il backend di esecuzione (simulatore o hardware reale).
- **Funzione:** `setup_ibm_quantum(api_token, use_real_hardware)`
- **Esecuzione Fornita:**
    - La connessione a IBM Quantum è fallita (`⚠ Errore connessione IBM Quantum`).
    - Il sistema ha ripiegato sull'uso del **simulatore locale Aer** di Qiskit (`→ Usando simulatore locale Aer`).

### STEP 2: Preparazione del Dataset
- **Obiettivo:** Generare un dataset sintetico per la classificazione binaria (es. *moons dataset*) e prepararlo per il QNN.
- **Processamento:**
    1. Generazione di 80 campioni totali.
    2. **Normalizzazione:** I dati vengono scalati nell'intervallo **$[0, \pi]$** (`MinMaxScaler`), un requisito comune per molte *Feature Map* quantistiche che codificano i dati tramite rotazioni.
    3. **Split:** Divisione in set di training (56 campioni) e test (24 campioni).
    4. **Etichette:** Le etichette di classe vengono convertite da $\{0, 1\}$ a **$\{-1, +1\}$** (standard per i QNN basati su *Estimator*).

### STEP 3: Costruzione del Circuito QNN
- **Obiettivo:** Definire l'architettura quantistica della rete (il "modello").
- **Architettura (Quantum Circuit):**
    1. **Feature Map:** `ZZFeatureMap` (2 qubit, 1 rep) - Codifica i 2 input classici.
    2. **Ansatz (Varioational Circuit):** `RealAmplitudes` (2 qubit, 2 reps) - Contiene i parametri trainabili.
- **Dettagli:** La QNN utilizza **2 qubit** e ha **6 parametri trainabili** (i "pesi" della rete).

### STEP 4: Training della QNN
- **Obiettivo:** Ottimizzare i parametri del circuito (l'Ansatz) minimizzando una funzione di costo.
- **Componenti:**
    - **Observable:** Viene misurata l'osservabile di Pauli $\sum Z_i$ (in questo caso $Z \otimes Z$), che mappa lo stato quantistico (output) a un valore scalare tra $-1$ e $+1$.
    - **Estimator:** Utilizza `EstimatorQNN` per calcolare l'output atteso della misurazione.
    - **Ottimizzatore:** `COBYLA` (maxiter=200).
- **Risultati di Training Forniti:**
    - **Valutazioni della funzione:** 400 (il numero di volte che il circuito è stato eseguito/simulato).
    - **Loss Finale:** $0.6577$ (Il valore della funzione di costo alla fine dell'ottimizzazione).
    - **Callback Warning:** Il warning sul callback è atteso con l'ottimizzatore `COBYLA` e indica solo che la cronologia passo-passo non è stata tracciata correttamente, non che il training sia fallito.

### STEP 5: Valutazione e Visualizzazione
- **Obiettivo:** Misurare l'accuratezza del modello sui dati non visti e visualizzare il *decision boundary*.

---

## 📊 Commento sui Risultati

I risultati mostrano il confronto diretto tra il Quantum Neural Network (QNN) e il Classical Neural Network (MLP) sullo stesso dataset (Moons, 80 campioni).

| Modello | Backend | Training Accuracy | Test Accuracy | Note |
| :--- | :--- | :--- | :--- | :--- |
| **QNN** | Aer Simulator | **83.93%** | **70.83%** | Ottenuta dopo 400 iterazioni COBYLA. |
| **MLP Classico** | CPU | **91.07%** | **87.50%** | Addestrato per 1000 iterazioni (Non ha raggiunto la convergenza completa). |

### Analisi Dettagliata:

1.  **Prestazioni (Test Accuracy):**
    - Il **MLP Classico (87.50%)** ha superato nettamente la **QNN (70.83%)** nel compito di classificazione per il *moons dataset*.
    - **Motivazione:** Per dataset a bassa dimensionalità e non eccessivamente complessi come le lune, un MLP ben ottimizzato (con 2 hidden layer) su un hardware classico performa tipicamente meglio. Il QNN, limitato a 2 qubit e con parametri fissi (come la scelta di Ansatz e Feature Map), fatica a trovare un *decision boundary* ottimale per questa specifica distribuzione, specialmente utilizzando ottimizzatori derivate-free (come COBYLA) su simulatori NISQ-style (Noisy Intermediate-Scale Quantum).

2.  **Training QNN:**
    - La *Training Accuracy* (83.93%) è significativamente più alta della *Test Accuracy* (70.83%), suggerendo che la QNN ha **overfittato** (memorizzato troppo bene) i dati di training, fallendo nella generalizzazione sui dati non visti.

3.  **Training MLP:**
    - Il MLP ha raggiunto un'ottima accuratezza ma ha generato un **`ConvergenceWarning`**. Questo significa che l'ottimizzatore non ha raggiunto il criterio di tolleranza dopo 1000 iterazioni, pur migliorando le performance. Un numero maggiore di iterazioni o una regolazione dei suoi iperparametri (`learning_rate`, dimensione dei layer) potrebbe aumentare ulteriormente la sua accuratezza, evidenziando il divario prestazionale con l'attuale QNN.

### Conclusione:

L'esperimento dimostra l'implementazione riuscita di un QNN in Qiskit. Tuttavia, come spesso accade nell'era **NISQ**, i modelli quantistici implementati su simulatori o hardware reali hanno difficoltà a competere con algoritmi classici maturi e ben ottimizzati su compiti di classificazione di dati a bassa dimensionalità. Il valore del QNN risiede nel suo potenziale futuro per dati complessi o *feature spaces* intrinsecamente quantistici.