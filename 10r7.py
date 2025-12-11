
Dane: zbiór danych „College”.
Cel: przewidywanie czesnego dla studentów spoza stanu (Outstate) na podstawie pozostałych zmiennych.

(a) Podział danych i selekcja cech

Podziel dane na zbiór treningowy i zbiór testowy (np. 70% trening, 30% test).

Użyj forward stepwise selection (selekcja krokowa do przodu) na zbiorze treningowym:

Celem jest wybranie tylko tych zmiennych, które naprawdę pomagają przewidzieć Outstate.

Zaczynamy od pustego modelu i dodajemy po jednej zmiennej w każdej iteracji, wybierając tę, która najbardziej poprawia dopasowanie modelu.

Kończymy, gdy żadna zmienna nie poprawia modelu.

Efekt: lista zmiennych, które będą użyte w modelu GAM.

(b) Dopasowanie GAM i wykresy

Dopasuj GAM (Generalized Additive Model) na zbiorze treningowym.

Zmienna objaśniana: Outstate.

Predyktory: tylko te wybrane w kroku (a).

GAM pozwala uchwycić nieliniowe zależności między zmiennymi a czesnym.

Wygeneruj wykresy partial dependence dla każdej zmiennej:

Pokażą, jak zmiana wartości zmiennej wpływa na przewidywane czesne.

Możesz od razu ocenić, które zmienne mają efekt liniowy, a które nieliniowy.

Wyjaśnij wyniki:

Które zmienne mają największy wpływ na czesne?

Czy wpływ jest liniowy czy nieliniowy?

(c) Ocena modelu na zbiorze testowym

Oblicz RMSE i R² na danych testowych.

RMSE: średni błąd predykcji w jednostkach czesnego.

R²: jaka część zmienności czesnego została wyjaśniona przez model.

Porównaj wyniki:

Czy model dobrze przewiduje czesne na nowych danych?

Czy uwzględnienie nieliniowości w GAM poprawiło predykcję w porównaniu do prostego modelu liniowego?

(d) Ocena nieliniowości

Na podstawie wykresów partial dependence i ewentualnie p-values:

Zidentyfikuj zmienne, które mają nieliniowy wpływ na Outstate.

Wykresy zakrzywione → nieliniowe zależności.

Płaskie, prawie liniowe krzywe → efekt liniowy.

Podsumowanie po polsku

Krok (a) = wybór najważniejszych zmiennych metodą forward stepwise.

Krok (b) = dopasowanie GAM i analiza kształtu funkcji zależności.

Krok (c) = sprawdzenie jakości modelu na danych testowych (RMSE, R²).

Krok (d) = wskazanie zmiennych z nieliniowym wpływem.
Na początku wczytałem dane z pliku College.csv. Zmienną, którą chcemy przewidzieć, jest Outstate, 
czyli czesne dla studentów spoza stanu. Pozostałe kolumny traktowałem jako predyktory,
a kolumnę Private zamieniłem na wartości 0 i 1, żeby można ją było użyć w modelu.
Nazwy uczelni wykorzystałem jako indeks. Następnie podzieliłem dane na zbiór treningowy i testowy w proporcji 70 do 30 procent,
z ustalonym random_state, żeby wyniki były powtarzalne.

Na zbiorze treningowym zastosowałem forward stepwise selection. 
To metoda, która pozwala wybrać tylko te zmienne, które rzeczywiście pomagają przewidywać czesne w modelu liniowym.
Algorytm działa iteracyjnie: zaczynamy od pustego modelu i w każdej iteracji testujemy dodanie każdej pozostałej zmiennej,
obliczając średni błąd walidacji krzyżowej. Dodajemy zmienną, która najbardziej poprawia wynik, 
i powtarzamy proces aż do momentu, gdy żadna kolejna zmiana nie poprawia błędu. W efekcie otrzymujemy listę predyktorów, 
które najlepiej wyjaśniają Outstate i jednocześnie nie przeucza modelu.
Średni MSE z 5 foldów mówi, jak dokładnie model przewiduje Outstate, jeśli dodamy tę zmienną.
Wybieramy zmienną, która minimalizuje średni MSE, bo oznacza, że poprawia predykcję najlepiej.

Na wybranym podzbiorze predyktorów dopasowałem GAM, czyli Generalized Additive Model.
To model, który pozwala na nieliniowe zależności: każda zmienna ma swoją funkcję wygładzającą — splajn.
Parametry wygładzania dopasowałem automatycznie przy użyciu grid search. 
Dzięki temu mogę zobaczyć, czy wpływ danej zmiennej na czesne jest liniowy czy nieliniowy, a efekt każdej zmiennej można analizować osobno.

Dla każdej zmiennej wygenerowałem wykresy partial dependence. Oś pozioma pokazuje wartość zmiennej,
a pionowa wkład tej zmiennej w przewidywane czesne. Jeśli krzywa jest prawie prosta, oznacza to efekt liniowy. 
Jeśli krzywa jest zakrzywiona lub ma spłaszczenia i progi, oznacza to efekt nieliniowy. 
W ten sposób można zobaczyć, jak zmiana zmiennej w różnych przedziałach wpływa na czesne.

Dodatkowo sprawdziłem istotność efektów przy pomocy p-values. Niskie p-value, poniżej 0,05, oznacza, że efekt jest statystycznie istotny. 
W połączeniu z wykresem krzywej spline pozwala określić, które zmienne mają faktyczny nieliniowy wpływ na Outstate, 
a które działają w sposób liniowy lub mają bardzo słaby wpływ.

Na koniec oceniłem model na zbiorze testowym, licząc RMSE i R². 
RMSE mówi o średnim błędzie przewidywania, a R² pokazuje, jaka część wariancji Outstate została wyjaśniona przez model.
Dzięki temu mogę stwierdzić, czy model dobrze generalizuje i czy uwzględnienie nieliniowości poprawiło przewidywania w porównaniu do modelu liniowego.

Podsumowując, najpierw wybrałem najważniejsze zmienne metodą forward selection, potem dopasowałem GAM, 
wygenerowałem wykresy partial dependence, sprawdziłem istotność efektów i oceniłem model na danych testowych.
Na tej podstawie mogę powiedzieć, które zmienne mają nieliniowy wpływ na czesne, a które działają liniowo. 
Cały proces pozwala zarówno uprościć model, jak i poprawić jego trafność przy zachowaniu interpretowalności.”


Każdy z 10 wykresów pokazuje wpływ jednej zmiennej wybranej w forward stepwise selection na przewidywane czesne.
Oś pozioma pokazuje wartości zmiennej, a oś pionowa jej wkład w przewidywane Outstate.
Niektóre wykresy są prawie liniowe, co oznacza, że zmiana zmiennej powoduje proporcjonalną zmianę czesnego.
Inne wykresy są wyraźnie zakrzywione — tu widzimy nieliniowe zależności, np. efekt nasycenia, progi lub zmieniające się nachylenie.
Krzywe o dużym nachyleniu pokazują zmienne, które mają największy wpływ na czesne, a płaskie odcinki wskazują, 
że zmiana zmiennej w tym zakresie nie wpływa znacząco na czesne.
Analizując p-values dla każdego splajnu możemy dodatkowo określić, które efekty są istotne statystycznie.
Na tej podstawie mogę wskazać zmienne, które mają silny i nieliniowy wpływ na czesne, a które są liniowe lub mało istotne.”

Ocena modelu na zbiorze testowym

Test RMSE = 1989.947
– Średni błąd predykcji czesnego wynosi około 1990 dolarów.
To daje praktyczną miarę, jak bardzo przewidywane czesne różnią się od rzeczywistych wartości w danych testowych.

Test R² = 0.752
– Model wyjaśnia 75,2% zmienności czesnego Outstate.
– To dość dobry wynik, oznacza, że GAM dobrze przewiduje czesne i uwzględnienie nieliniowości poprawia dopasowanie w porównaniu 
do prostego modelu liniowego (który zwykle miałby niższe R²).

Najważniejsze zmienne – Expend, Private, Room.Board, Grad.Rate – mają bardzo niski p-value (<0,01), więc są silnie powiązane z czesnym.

Zmienne z umiarkowaną istotnością – perc.alumni, Enroll, S.F.Ratio – też mają wpływ, choć nie tak mocny jak pierwsze cztery.

Nieistotne zmienne – Terminal, PhD, Personal – ich p-value > 0,05, więc nie wnoszą istotnej informacji do modelu w kontekście GAM.

