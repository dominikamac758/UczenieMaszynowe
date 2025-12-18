(a) Usunięcie braków i transformacja:

- wczytać zbiór danych Hitters
- usunąć obserwacje, dla których Salary jest nieznane (NA)
- wykonać transformację Salary → log(Salary)
Po co:
brakujące dane uniemożliwiają uczenie modelu
logarytm stabilizuje wariancję i poprawia jakość dopasowania

(b) Podział na zbiór treningowy i testowy:

- utworzyć zbiór treningowy z pierwszych 200 obserwacji
- utworzyć zbiór testowy z pozostałych obserwacji
Po co:
żeby rzetelnie ocenić jakość predykcji na nowych danych

(c) Boosting – wpływ shrinkage na błąd treningowy

-dopasować model boosting z 1000 drzewami
-przetestować kilka wartości shrinkage (np. 0.001, 0.01, 0.05, 0.1)
-dla każdej wartości:
policzyć MSE na zbiorze treningowym
narysować wykres:

Po co:
żeby zobaczyć, jak shrinkage wpływa na dopasowanie modelu
małe λ → wolniejsze, ale stabilniejsze uczenie

(d) Boosting – wpływ shrinkage na błąd testowy:

- dla tych samych wartości shrinkage:
policzyć MSE na zbiorze testowym
narysować drugi wykres:

Po co:
żeby znaleźć shrinkage dający najlepszą generalizację

(e) Porównanie z innymi metodami regresji

- dopasować dwie metody regresji znane z:
-rozdziału 3 (np. regresja liniowa)
-rozdziału 6 (np. ridge lub lasso)
-obliczyć test MSE dla każdej metody
-porównać ich wyniki z boostingiem

Po co:
żeby sprawdzić, czy boosting rzeczywiście daje lepsze wyniki

(f) Najważniejsze predyktory w boosting

- wybrać najlepszy model boosting (najniższy test MSE)
-sprawdzić importance variables
-wypisać najważniejsze zmienne

Po co:
żeby zinterpretować model
zrozumieć, co najbardziej wpływa na Salary

(g) Bagging:

- zastosować bagging na zbiorze treningowym
- użyć wielu drzew (np. 500 lub więcej)
-obliczyć test MSE
-porównać wynik z boostingiem

Po co:
żeby zobaczyć różnicę między:
redukcją wariancji (bagging)
sekwencyjnym uczeniem (boosting)


PODPUNKT C:

shrinkage_values → lista różnych wartości parametru λ (learning rate), które chcemy sprawdzić.
train_mse i test_mse → puste listy, w których będziemy zapisywać błędy (MSE) dla każdej wartości shrinkage.

Shrinkage w boosting kontroluje „krok” każdego drzewa.
Chcemy zobaczyć, jak zmienia się błąd treningowy w zależności od λ.
Puste listy są potrzebne do zapisania wyników w pętli.

Przechodzimy kolejno przez wszystkie wartości λ z listy. lr -learning rate
Dla każdej wartości budujemy nowy model boosting i obliczamy MSE.
Musimy porównać, która wartość shrinkage daje najlepszy model.

GradientBoostingRegressor → tworzy model boosting dla regresji

Parametry:
n_estimators=1000 → liczba drzew w boosting
learning_rate=lr → shrinkage (mniejsza wartość = mniejszy krok każdego drzewa)
max_depth=3 → maksymalna głębokość każdego drzewa (kontrola overfittingu)
random_state=1 → zapewnia powtarzalność wyników
boost.fit(X_train, y_train) → trenuje model na danych treningowych

Boosting działa sekwencyjnie – każde drzewo poprawia błędy poprzednich.
Parametr learning_rate decyduje, jak duży wpływ ma każde drzewo.
Trenujemy model dla każdej wartości shrinkage, żeby porównać wyniki.

train_pred → przewidywane log(Salary) na danych treningowych
test_pred → przewidywane log(Salary) na danych testowych
Musimy policzyć błąd modelu dla danych, na których był trenowany, żeby wykres pokazywał MSE 
w zależności od shrinkage.

mean_squared_error(y_train, train_pred) → oblicza średni błąd kwadratowy dla danych treningowych
append() → dodaje wynik do listy train_mse
Analogicznie dla testu → test_mse

(e)
lin_reg = LinearRegression()
Tworzymy obiekt LinearRegression – klasyczny model regresji liniowej.
Model będzie szukał współczynników β dla każdej cechy, 
aby najlepiej dopasować log(Salary) do danych treningowych.

{lin_reg.fit(X_train, y_train)}
Trenujemy model na zbiorze treningowym (X_train, y_train).
Model oblicza optymalne współczynniki β, minimalizując błąd kwadratowy między przewidywanymi 
a prawdziwymi wartościami y_train.

{lin_pred = lin_reg.predict(X_test)}
Robimy przewidywania log(Salary) dla danych testowych (X_test).
Wynik to lista przewidywanych wartości, które porównamy z rzeczywistymi wartościami y_test.

{lin_mse = mean_squared_error(y_test, lin_pred)}
Obliczamy średni błąd kwadratowy (MSE) między rzeczywistymi wartościami y_test a przewidywaniami lin_pred.
MSE pokazuje, jak bardzo model błądzi na danych, których nie widział podczas treningu.

{ridge = Ridge(alpha=1.0)}
Tworzymy obiekt Ridge Regression (regresja liniowa z regularyzacją L2).
alpha=1.0 → siła regularyzacji (większe α = większa kara za duże współczynniki β).
Regularyzacja zmniejsza ryzyko overfittingu, szczególnie przy dużej liczbie cech.

{ridge.fit(X_train, y_train)}
Trenujemy Ridge Regression na danych treningowych.
Model dopasowuje współczynniki β, jednocześnie ograniczając ich wielkość, 
żeby uniknąć nadmiernego dopasowania do treningu.

{ridge_pred = ridge.predict(X_test)}
Robimy przewidywania log(Salary) dla danych testowych przy użyciu Ridge Regression.

{ridge_mse = mean_squared_error(y_test, ridge_pred)}
Obliczamy MSE dla przewidywań Ridge Regression.
Porównamy je z Linear Regression i Boosting, żeby ocenić, która metoda najlepiej generalizuje.

(f)
np.argmin(test_mse) → zwraca indeks najmniejszej wartości w liście test_mse
(czyli najlepsze λ pod względem błędu na teście)

shrinkage_values[...] → wybiera odpowiadającą temu indeksowi wartość shrinkage
Wynik zapisujemy w best_lr
Chcemy użyć w Boostingu optymalnego learning rate, które daje najlepsze przewidywania na zbiorze testowym.
Tworzy nowy obiekt Gradient Boosting Regressor:
n_estimators=1000 → liczba drzew w boosting
learning_rate=best_lr → najlepszy shrinkage wybrany wcześniej
max_depth=3 → maksymalna głębokość każdego drzewa
random_state=42 → zapewnia powtarzalność wyników

Trenujemy finalny model Boosting na danych treningowych z optymalnym parametrem shrinkage.
{best_boost.fit(X_train, y_train)}

Model uczy się na zbiorze treningowym (X_train, y_train).
Każde drzewo poprawia błędy poprzednich drzew.

Po treningu możemy sprawdzić, które cechy są najważniejsze w przewidywaniu log(Salary).
{importance = (
    pd.Series(best_boost.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
)}

best_boost.feature_importances_ → lista ważności każdej cechy w modelu boosting.
Wyższa wartość → cecha bardziej wpływa na przewidywania.
pd.Series(..., index=X.columns) → tworzy serię Pandas, gdzie indeksy to nazwy zmiennych, a wartości to ich ważność.
.sort_values(ascending=False) → sortuje cechy od najważniejszej do najmniej ważnej.

(g)
  bagging = RandomForestRegressor(
    n_estimators=500,
    max_features=X.shape[1],
    random_state=42
)

Tworzy Random Forest Regressor, który w tym przypadku działa jako bagging:
n_estimators=500 → liczba drzew w lesie
max_features=X.shape[1] → każde drzewo używa wszystkich cech (to właśnie sprawia, że jest klasyczny bagging, 
a nie typowy random forest z losowymi cechami)
random_state=42 → zapewnia powtarzalność wyników
Bagging tworzy wiele drzew na różnych próbkach danych (bootstrap) 
i średnia ich przewidywań daje bardziej stabilny model.

Użycie wszystkich cech na każdym drzewie to klasyczny sposób implementacji baggingu.
bagging.fit(X_train, y_train)
Co robi:
Trenuje Random Forest na zbiorze treningowym.
Każde drzewo uczy się na losowej próbce z bootstrap (próbkowanie z powtórzeniami).
Model uśrednia przewidywania wszystkich drzew, co zmniejsza wariancję i poprawia stabilność predykcji.
bag_pred = bagging.predict(X_test)
Co robi:
Robi przewidywania log(Salary) dla zbioru testowego.
Wynik to lista przewidywanych wartości dla każdego zawodnika w teście.
bag_mse = mean_squared_error(y_test, bag_pred)
Oblicza średni błąd kwadratowy między rzeczywistymi wartościami (y_test) a przewidywaniami modelu (bag_pred).

MSE pozwala porównać skuteczność baggingu z Boostingiem i regresjami liniowymi.

