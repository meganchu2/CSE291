(set-logic CHC_LIA)

(synth-fun starexecinv1 ((x_0 Int) (x_1 Int) (x_2 Int) (x_3 Int)) Bool)

(constraint (forall ((cleaned_symbol_0 Int) (cleaned_symbol_1 Int) (cleaned_symbol_2 Int) (cleaned_symbol_3 Int)) (let ((a!1 (> cleaned_symbol_2 (* (+ 0 (- 1)) cleaned_symbol_0))) (a!2 (> cleaned_symbol_2 (* (+ 0 (- 1)) cleaned_symbol_1))) (a!3 (> cleaned_symbol_3 (* (+ 0 (- 1)) cleaned_symbol_0))) (a!4 (> cleaned_symbol_3 (* (+ 0 (- 1)) cleaned_symbol_1)))) (let ((a!5 (and true (= cleaned_symbol_0 (+ 0 (- 2))) (= cleaned_symbol_1 (+ 0 1)) (> cleaned_symbol_2 cleaned_symbol_0) (> cleaned_symbol_2 cleaned_symbol_1) a!1 a!2 (> cleaned_symbol_3 cleaned_symbol_0) (> cleaned_symbol_3 cleaned_symbol_1) a!3 a!4))) (=> a!5 (starexecinv1 cleaned_symbol_0 cleaned_symbol_1 cleaned_symbol_2 cleaned_symbol_3))))))
(constraint (forall ((cleaned_symbol_4 Int) (cleaned_symbol_5 Int) (gh0 Int) (gh1 Int) (cleaned_symbol_0 Int) (cleaned_symbol_1 Int) (cleaned_symbol_3 Int) (cleaned_symbol_2 Int)) (let ((a!1 (and (= cleaned_symbol_3 (- gh1 (+ 0 1))) (> gh1 (+ 0 0)) (= cleaned_symbol_2 gh0))) (a!2 (> cleaned_symbol_3 (* (+ 0 (- 1)) cleaned_symbol_0))) (a!3 (> cleaned_symbol_3 (* (+ 0 (- 1)) cleaned_symbol_1))) (a!5 (>= (* (+ 0 (- 1)) cleaned_symbol_4) (+ 0 2))) (a!6 (>= (* (+ 0 (- 1)) cleaned_symbol_5) (+ 0 (- 1))))) (let ((a!4 (and (= cleaned_symbol_2 (- gh0 (+ 0 1))) (> cleaned_symbol_3 cleaned_symbol_0) (> cleaned_symbol_3 cleaned_symbol_1) a!2 a!3 (<= gh1 (+ 0 0))))) (let ((a!7 (and (starexecinv1 cleaned_symbol_4 cleaned_symbol_5 gh0 gh1) (> cleaned_symbol_4 (+ 0 0)) (= cleaned_symbol_0 (+ cleaned_symbol_4 cleaned_symbol_5)) (= cleaned_symbol_1 (+ cleaned_symbol_5 (+ 0 1))) (or a!1 a!4) (>= (* (+ 0 1) cleaned_symbol_4) (+ 0 (- 2))) a!5 (>= (* (+ 0 1) cleaned_symbol_5) (+ 0 1)) a!6))) (=> a!7 (starexecinv1 cleaned_symbol_0 cleaned_symbol_1 cleaned_symbol_2 cleaned_symbol_3)))))))
(constraint (forall ((cleaned_symbol_4 Int) (cleaned_symbol_5 Int) (gh0 Int) (gh1 Int)) (let ((a!1 (>= (* (+ 0 (- 1)) cleaned_symbol_4) (+ 0 2))) (a!2 (>= (* (+ 0 (- 1)) cleaned_symbol_5) (+ 0 (- 1))))) (let ((a!3 (and (starexecinv1 cleaned_symbol_4 cleaned_symbol_5 gh0 gh1) (> cleaned_symbol_4 (+ 0 0)) (< gh0 (+ 0 0)) (>= (* (+ 0 1) cleaned_symbol_4) (+ 0 (- 2))) a!1 (>= (* (+ 0 1) cleaned_symbol_5) (+ 0 1)) a!2))) (=> a!3 false)))))

(check-synth)

