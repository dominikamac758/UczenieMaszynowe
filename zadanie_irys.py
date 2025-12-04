Na początku kod korzysta z pakietu fetch_openml, aby pobrać zbiór danych Iris
Zawiera 150 przykładów i trzy klasy gatunków
Każdy przykład opisany jest czterema cechami liczbowymi: długością i szerokością płatka oraz działki.
Następnie dane są dzielone na zbiór treningowy i testowy w proporcji 80 do 20, przy czym wykorzystuję tu parametr stratify=y, 
aby proporcje klas zostały zachowane w obu zbiorach.
Dzięki temu test jest bardziej wiarygodny, ponieważ reprezentuje strukturę danych rzeczywistych.
własny skaler o nazwie RobustMinMaxScaler.
Jest to połączenie dwóch pomysłów:
robust scaling — czyli używania percentyli zamiast średniej i odchylenia standardowego,
min-max scaling — czyli sprowadzania wszystkiego do przedziału [0, 1].
Następnie tworzę słownik, w którym przechowuję wszystkie typy skalowania: standardowe, robust, min-max oraz mój własny robust-minmax.
Słownik jest później wykorzystywany przez Optunę, która automatycznie wybiera, który skaler działa najlepiej dla każdego modelu.
Optuna jest biblioteką do automatycznej optymalizacji hiperparametrów
system sam szuka najlepszych ustawień, testując wiele wariantów i oceniając ich działanie.
Funkcja objective_lr losuje różne wartości parametrów, takich jak:
liczba cech, które mają pozostać po selekcji RFE,
wybór solvera,
parametr kary regularyzacyjnej C,
wybór skalera,
w przypadku mojego skalera — również wartości percentyli.
Następnie te elementy są łączone w jeden pipeline, który składa się z:
skalera,
selektora cech RFE,
właściwego modelu regresji logistycznej.
Funkcja celu dla kNN
Analogicznie przygotowałem funkcję celu dla algorytmu kNN.
W tym przypadku Optuna dobiera:
liczbę sąsiadów,
rodzaj wag,
parametr p w metryce Minkowskiego,
liczbę wybranych cech,
rodzaj skalera.
Podobnie jak wcześniej, tworzony jest pipeline

Po zdefiniowaniu funkcji celu uruchamiam optymalizację dla regresji logistycznej.
Optuna wykonuje 30 prób, w każdej testując inną konfigurację hyperparametrów.
Po zakończeniu zapisuje najlepszy znaleziony zestaw parametrów.
Analogicznie mogę uruchomić optymalizację dla kNN.

Następnie buduję końcowy model korzystając z najlepszych parametrów wybranych przez Optunę.
Trenuję pipeline na pełnym zbiorze treningowym, a następnie wykonuję predykcję na danych testowych.
Obliczam podstawowe miary jakości:
accuracy,
precision,
recall,
f1-score.
Dzięki temu wiem, jak dobrze model generalizuje na danych, których wcześniej nie widział.

Aby ostatecznie porównać oba modele, wykonuję jeszcze raz walidację krzyżową dla regresji logistycznej i dla kNN przy tych najlepszych parametrach.
Zapisuję dokładności uzyskane w każdym foldzie.
Następnie stosuję test Wilcoxona, aby sprawdzić, czy jeden model rzeczywiście radzi sobie istotnie lepiej od drugiego, czy też różnice wynikają z przypadku.
Ten test jest nieparametryczny i idealnie nadaje się do porównywania dwóch powiązanych prób

RFE to technika, która automatycznie wybiera najważniejsze cechy, usuwając te mniej istotne.
Działa według prostego schematu:
Tworzy model (np. LogisticRegression, SVM, kNN, RandomForest).
Trenuje go na wszystkich cechach.
Mierzy ważność każdej cechy – np. współczynniki modelu lub ich wpływ na wynik.
Usuwa najmniej ważną cechę.
Trenuje model ponownie już bez tej cechy.
Powtarza te kroki, aż zostanie wybrana wymagana liczba cech.


class RobustMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, q_range = (25, 75), clip=False):
        self.clip = clip  # czy obcinać wartości do przedziału [0,1]
        self.q_range = q_range  # przedział percentyli do skalowania

    def fit(self, X, y=None):
        # Obliczenie dolnego i górnego percentyla dla każdej cechy
        self.qL = np.percentile(X, self.q_range[0], axis=0)
        self.qU = np.percentile(X, self.q_range[1], axis=0)
        return self

    def transform(self, X):
        # Skalowanie danych do zakresu [0,1] na podstawie percentyli
        X_n = (X - self.qL) / (self.qU - self.qL)
        if self.clip:
            X_n = np.clip(X_n, 0, 1)  # obcięcie wartości wychodzących poza [0,1]
        return X_n

# ===============================
# 3. Słownik skalersów do wyboru w Optunie
# ===============================
mappingScalar = {
    "standard": StandardScaler(),
    "robust_minmax": RobustMinMaxScaler(),
    "robust": RobustScaler(),
    "minmax": MinMaxScaler()
}

# ===============================
# 4. Funkcja celu dla regresji logistycznej (Optuna)
# ===============================
def objective_lr(trial):
    # Optuna losuje liczbę cech, solver, typ regularyzacji, siłę regularyzacji i typ skalera
    n_features = trial.suggest_int('n_features', 1, X_train.shape[1])
    solver = trial.suggest_categorical('solver', ['lbfgs', 'saga'])
    penalty = trial.suggest_categorical('penalty', ['l2'])
    C = trial.suggest_loguniform('C', 1e-4, 1e4)
    scaler_name = trial.suggest_categorical('scaler', ["standard", "robust_minmax", "robust", "minmax"])

    # Tworzymy obiekt skalera w zależności od wyboru
    if scaler_name == 'standard':
        scaler_obj = StandardScaler()
    elif scaler_name == 'robust_minmax':
        q = trial.suggest_float('q_low', 2.5, 25)
        clip = trial.suggest_categorical('clip', [True, False])
        scaler_obj = RobustMinMaxScaler(q_range=(q, 100-q), clip=clip)
    elif scaler_name == 'robust':
        scaler_obj = RobustScaler()
    else:
        scaler_obj = MinMaxScaler()

    # Pipeline: skalowanie → selekcja cech RFE → regresja logistyczna
    pipeline_lr = Pipeline([
        ('scaler', scaler_obj),
        ('rfe', RFE(estimator=LogisticRegression(max_iter=10000), n_features_to_select=n_features)),
        ('clf', LogisticRegression(max_iter=10000, penalty=penalty, C=C, solver=solver))
    ])

    # 5-krotna walidacja krzyżowa z zachowaniem proporcji klas
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    score = cross_val_score(pipeline_lr, X_train, Y_train, cv=cv, scoring='balanced_accuracy').mean()

    return score  # Optuna maksymalizuje balanced_accuracy

# ===============================
# 5. Funkcja celu dla kNN (Optuna)
# ===============================
def objective_knn(trial):
    n_features = trial.suggest_int('n_features', 1, X_train.shape[1])
    n_neighbors = trial.suggest_int('n_neighbors', 1, 50)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    p = trial.suggest_int('p', 1, 3)
    scaler_name = trial.suggest_categorical('scaler', ["standard", "robust_minmax", "robust_minmax", "minmax"])
    scaler_obj = mappingScalar[scaler_name]

    # Pipeline: skalowanie → selekcja cech RFE → kNN
    pipeline_knn = Pipeline([
        ('scaler', scaler_obj),
        ('rfe', RFE(estimator=KNeighborsClassifier(), n_features_to_select=n_features)),
        ('clf', KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True)
    score = cross_val_score(pipeline_knn, X_train, Y_train, cv=cv, scoring='accuracy').mean()
    return score  # Optuna maksymalizuje accuracy

# ===============================
# 6. Tworzenie i uruchamianie studium Optuna dla LR
# ===============================
sampler = TPESampler()
study_lr = optuna.create_study(direction='maximize', sampler=sampler)
study_lr.optimize(objective_lr, n_trials=30)  # testujemy 30 kombinacji parametrów

# ===============================
# 7. Trenowanie finalnego modelu LR i predykcja
# ===============================
lr_best_params = study_lr.best_params.copy()
n_features_lr = lr_best_params.pop('n_features')  # liczba cech do RFE
pipeline_lr.fit(X_train, Y_train)  # trenowanie pipeline
y_pred_lr = pipeline_lr.predict(X_test)  # predykcja na zbiorze testowym

# ===============================
# 8. Wyświetlanie wyników LR
# ===============================
print("Logistic Regression:")
print(f"Accuracy: {accuracy_score(Y_test, y_pred_lr):.3f}")
print(f"Precision (macro): {precision_score(Y_test, y_pred_lr, average='macro'):.3f}")
print(f"Recall (macro): {recall_score(Y_test, y_pred_lr, average='macro'):.3f}")
print(f"F1 (macro): {f1_score(Y_test, y_pred_lr, average='macro'):.3f}")

# ===============================
# 9. Trenowanie finalnego modelu kNN i predykcja
# ===============================
knn_best_params = study_knn.best_params.copy()
n_features_knn = knn_best_params.pop('n_features')
pipeline_knn.set_params(rfe__n_features_to_select=n_features_knn)
pipeline_knn.set_params(clf__n_neighbors=knn_best_params['n_neighbors'], clf__weights=knn_best_params['weights'], clf__p=knn_best_params['p'])
pipeline_knn.fit(X_train, Y_train)
y_pred_knn = pipeline_knn.predict(X_test)

Interpretacja:

Model LR dokładnie sklasyfikował wszystkie próbki testowe (accuracy 100%).

Precision (macro) = 1 → wszystkie przewidywane klasy były poprawne.

Recall (macro) = 1 → wszystkie prawdziwe klasy zostały poprawnie wykryte.

F1 (macro) = 1 → idealny balans między precyzją a recall.

✅ Wniosek: LR perfekcyjnie poradził sobie na zbiorze testowym Iris.

k-Nearest Neighbors (kNN)
Accuracy: 0.967
Precision (macro): 0.970
Recall (macro): 0.967
F1 (macro): 0.967


Interpretacja:

Model kNN zrobił 1–2 drobne błędy (accuracy 96.7%).

Precyzja, recall i F1 są bardzo wysokie, bliskie 1 → model jest nadal bardzo dobry.

✅ Wniosek: kNN działa bardzo dobrze, ale LR jest minimalnie lepszy na tym zestawie.

2. Walidacja krzyżowa (CV)
CV accs LR: [1.    1.    0.875 0.875 1.   ]
CV accs kNN: [0.9583 1.     1.     0.875  0.9583]


Interpretacja:
CV accs LR → dokładność LR w każdym foldzie 5-krotnej walidacji:

Fold 1: 100%
Fold 2: 100%
Fold 3: 87.5%
Fold 4: 87.5%
Fold 5: 100%
Średnia dokładność LR ≈ 95–97%.

CV accs kNN → dokładność kNN w każdym foldzie:
Fold 1: 95.8%
Fold 2: 100%
Fold 3: 100%
Fold 4: 87.5%
Fold 5: 95.8%
Średnia dokładność kNN ≈ 95–97%

✅ Wniosek: Oba modele działają bardzo dobrze w walidacji krzyżowej, wyniki są stabilne.

3. Test statystyczny Wilcoxona
WilcoxonResult(statistic=3.0, pvalue=1.0)


Co to oznacza:
Test Wilcoxona porównuje dwie pary próbek (tu: dokładności foldów LR vs kNN).
statistic = 3.0 → wartość statystyki testowej
pvalue = 1.0 → brak istotnej różnicy statystycznej

✅ Interpretacja:
Nie ma statystycznie istotnej różnicy między dokładnością LR a kNN.
Chociaż LR miał 100% na teście, a kNN 96.7%, to w ujęciu statystycznym (na CV) różnica jest niewielka i może być losowa.
