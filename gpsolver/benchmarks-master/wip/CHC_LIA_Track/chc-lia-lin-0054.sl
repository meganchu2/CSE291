(set-logic CHC_LIA)

(synth-fun cleaned_symbol_0 ((x_0 Int) (x_1 Int) (x_2 Int) (x_3 Int)) Bool)

(synth-fun cleaned_symbol_1 ((x_0 Int) (x_1 Int) (x_2 Int) (x_3 Int)) Bool)

(synth-fun cleaned_symbol_2 ((x_0 Int) (x_1 Int) (x_2 Int) (x_3 Int)) Bool)

(synth-fun cleaned_symbol_3 ((x_0 Int) (x_1 Int) (x_2 Int)) Bool)

(synth-fun cleaned_symbol_4 ((x_0 Int) (x_1 Int) (x_2 Int)) Bool)

(constraint (forall ((cleaned_symbol_5 Int) (cleaned_symbol_6 Int) (cleaned_symbol_7 Int) (cleaned_symbol_8 Int) (cleaned_symbol_9 Int) (cleaned_symbol_10 Int) (cleaned_symbol_11 Int)) (let ((a!1 (and (= (not (= 0 cleaned_symbol_5)) (>= cleaned_symbol_6 cleaned_symbol_7)) (= cleaned_symbol_8 (+ cleaned_symbol_6 1)) (= cleaned_symbol_9 1) (not (not (= 0 cleaned_symbol_5))) (cleaned_symbol_0 cleaned_symbol_10 cleaned_symbol_11 cleaned_symbol_9 cleaned_symbol_6) true true true))) (=> a!1 (cleaned_symbol_1 cleaned_symbol_10 cleaned_symbol_11 cleaned_symbol_7 cleaned_symbol_8)))))
(constraint (forall ((cleaned_symbol_5 Int) (cleaned_symbol_6 Int) (cleaned_symbol_7 Int) (cleaned_symbol_12 Int) (cleaned_symbol_13 Int)) (let ((a!1 (and (= (not (= 0 cleaned_symbol_5)) (>= cleaned_symbol_6 cleaned_symbol_7)) (not (= 0 cleaned_symbol_5)) true (cleaned_symbol_1 cleaned_symbol_12 cleaned_symbol_13 cleaned_symbol_7 cleaned_symbol_6) true true))) (=> a!1 (cleaned_symbol_2 cleaned_symbol_12 cleaned_symbol_13 cleaned_symbol_7 cleaned_symbol_6)))))
(constraint (forall ((cleaned_symbol_5 Int) (cleaned_symbol_6 Int) (cleaned_symbol_7 Int) (cleaned_symbol_9 Int) (cleaned_symbol_12 Int) (cleaned_symbol_14 Int)) (let ((a!1 (and (= (not (= 0 cleaned_symbol_5)) (>= cleaned_symbol_6 cleaned_symbol_7)) (= cleaned_symbol_9 1) (not (not (= 0 cleaned_symbol_5))) true (cleaned_symbol_1 cleaned_symbol_12 cleaned_symbol_14 cleaned_symbol_7 cleaned_symbol_6) true true))) (=> a!1 (cleaned_symbol_3 cleaned_symbol_12 cleaned_symbol_14 cleaned_symbol_6)))))
(constraint (forall ((cleaned_symbol_5 Int) (cleaned_symbol_6 Int) (cleaned_symbol_7 Int) (cleaned_symbol_8 Int) (cleaned_symbol_9 Int) (cleaned_symbol_15 Int) (cleaned_symbol_16 Int)) (let ((a!1 (and (= (not (= 0 cleaned_symbol_5)) (>= cleaned_symbol_6 cleaned_symbol_7)) (= cleaned_symbol_8 (+ cleaned_symbol_6 1)) (= cleaned_symbol_9 1) (not (not (= 0 cleaned_symbol_5))) (cleaned_symbol_2 cleaned_symbol_15 cleaned_symbol_16 cleaned_symbol_7 cleaned_symbol_8) true true true))) (=> a!1 (cleaned_symbol_2 cleaned_symbol_15 cleaned_symbol_16 cleaned_symbol_7 cleaned_symbol_6)))))
(constraint (forall ((cleaned_symbol_17 Int) (cleaned_symbol_18 Int) (cleaned_symbol_19 Int) (cleaned_symbol_20 Int)) (=> (and (= cleaned_symbol_17 0) (cleaned_symbol_4 cleaned_symbol_18 cleaned_symbol_19 cleaned_symbol_20) true) (cleaned_symbol_1 cleaned_symbol_18 cleaned_symbol_19 cleaned_symbol_20 cleaned_symbol_17))))
(constraint (forall ((cleaned_symbol_17 Int) (cleaned_symbol_21 Int) (cleaned_symbol_22 Int) (cleaned_symbol_23 Int) (cleaned_symbol_24 Int) (cleaned_symbol_25 Int) (cleaned_symbol_26 Int) (cleaned_symbol_20 Int)) (let ((a!1 (= (not (= 0 cleaned_symbol_23)) (and (not (= 0 cleaned_symbol_24)) (not (= 0 cleaned_symbol_25)))))) (let ((a!2 (and (= cleaned_symbol_17 0) (= (not (= 0 cleaned_symbol_21)) (>= cleaned_symbol_22 1)) a!1 (= (not (= 0 cleaned_symbol_25)) (< cleaned_symbol_26 cleaned_symbol_20)) (= (not (= 0 cleaned_symbol_24)) (<= 0 cleaned_symbol_26)) (not (not (= 0 cleaned_symbol_21))) (not (= 0 cleaned_symbol_23)) (cleaned_symbol_2 cleaned_symbol_22 cleaned_symbol_26 cleaned_symbol_20 cleaned_symbol_17)))) (=> a!2 false)))))
(constraint (forall ((cleaned_symbol_27 Int) (cleaned_symbol_28 Int) (cleaned_symbol_29 Int) (cleaned_symbol_30 Int) (cleaned_symbol_31 Int) (cleaned_symbol_32 Int)) (let ((a!1 (= (not (= 0 cleaned_symbol_27)) (and (not (= 0 cleaned_symbol_28)) (not (= 0 cleaned_symbol_29)))))) (let ((a!2 (and a!1 (= (not (= 0 cleaned_symbol_29)) (< cleaned_symbol_30 cleaned_symbol_31)) (= (not (= 0 cleaned_symbol_28)) (<= 0 cleaned_symbol_30)) (= cleaned_symbol_32 0) (not (= 0 cleaned_symbol_27)) true true))) (=> a!2 (cleaned_symbol_4 cleaned_symbol_32 cleaned_symbol_30 cleaned_symbol_31))))))
(constraint (forall ((cleaned_symbol_27 Int) (cleaned_symbol_28 Int) (cleaned_symbol_29 Int) (cleaned_symbol_30 Int) (cleaned_symbol_31 Int) (cleaned_symbol_33 Int)) (let ((a!1 (= (not (= 0 cleaned_symbol_27)) (and (not (= 0 cleaned_symbol_28)) (not (= 0 cleaned_symbol_29)))))) (let ((a!2 (and a!1 (= (not (= 0 cleaned_symbol_29)) (< cleaned_symbol_30 cleaned_symbol_31)) (= (not (= 0 cleaned_symbol_28)) (<= 0 cleaned_symbol_30)) (= cleaned_symbol_33 (- 1)) (not (not (= 0 cleaned_symbol_27))) true true))) (=> a!2 (cleaned_symbol_4 cleaned_symbol_33 cleaned_symbol_30 cleaned_symbol_31))))))
(constraint (forall ((cleaned_symbol_34 Int) (cleaned_symbol_35 Int) (cleaned_symbol_36 Int) (cleaned_symbol_37 Int) (cleaned_symbol_38 Int) (cleaned_symbol_39 Int)) (let ((a!1 (and (= (not (= 0 cleaned_symbol_34)) (= cleaned_symbol_35 cleaned_symbol_36)) (= cleaned_symbol_37 cleaned_symbol_38) (not (not (= 0 cleaned_symbol_34))) true true (cleaned_symbol_3 cleaned_symbol_38 cleaned_symbol_35 cleaned_symbol_36) true))) (=> a!1 (cleaned_symbol_0 cleaned_symbol_37 cleaned_symbol_35 cleaned_symbol_39 cleaned_symbol_36)))))
(constraint (forall ((cleaned_symbol_34 Int) (cleaned_symbol_35 Int) (cleaned_symbol_36 Int) (cleaned_symbol_40 Int) (cleaned_symbol_39 Int)) (let ((a!1 (and (= (not (= 0 cleaned_symbol_34)) (= cleaned_symbol_35 cleaned_symbol_36)) (= cleaned_symbol_40 cleaned_symbol_39) (not (= 0 cleaned_symbol_34)) true true true))) (=> a!1 (cleaned_symbol_0 cleaned_symbol_40 cleaned_symbol_35 cleaned_symbol_39 cleaned_symbol_36)))))

(check-synth)
