(set-logic LIA)

(synth-inv inv-f ((conf_0 Int) (conf_1 Int) (conf_2 Int) (conf_3 Int) (conf_4 Int) (x Int) (conf_0_0 Int) (conf_1_0 Int) (conf_1_1 Int) (conf_1_2 Int) (conf_2_0 Int) (conf_3_0 Int) (conf_4_0 Int) (x_0 Int) (x_1 Int) (x_2 Int) (x_3 Int)))

(define-fun pre-f ((conf_0 Int) (conf_1 Int) (conf_2 Int) (conf_3 Int) (conf_4 Int) (x Int) (conf_0_0 Int) (conf_1_0 Int) (conf_1_1 Int) (conf_1_2 Int) (conf_2_0 Int) (conf_3_0 Int) (conf_4_0 Int) (x_0 Int) (x_1 Int) (x_2 Int) (x_3 Int)) Bool
    (and (= conf_0 conf_0_0) (= conf_1 conf_1_0) (= conf_2 conf_2_0) (= conf_3 conf_3_0) (= conf_4 conf_4_0) (= x x_1) (= conf_0_0 0) (= conf_1_0 3) (= conf_2_0 6) (= conf_3_0 9) (= conf_4_0 1) (= x_1 0)))
(define-fun trans-f ((conf_0 Int) (conf_1 Int) (conf_2 Int) (conf_3 Int) (conf_4 Int) (x Int) (conf_0_0 Int) (conf_1_0 Int) (conf_1_1 Int) (conf_1_2 Int) (conf_2_0 Int) (conf_3_0 Int) (conf_4_0 Int) (x_0 Int) (x_1 Int) (x_2 Int) (x_3 Int) (conf_0! Int) (conf_1! Int) (conf_2! Int) (conf_3! Int) (conf_4! Int) (x! Int) (conf_0_0! Int) (conf_1_0! Int) (conf_1_1! Int) (conf_1_2! Int) (conf_2_0! Int) (conf_3_0! Int) (conf_4_0! Int) (x_0! Int) (x_1! Int) (x_2! Int) (x_3! Int)) Bool
    (or (and (= conf_1_1 conf_1) (= x_2 x) (= conf_1_1 conf_1!) (= x_2 x!) (= conf_0 conf_0!) (= conf_1 conf_1!) (= conf_2 conf_2!) (= conf_3 conf_3!) (= conf_4 conf_4!)) (and (= conf_1_1 conf_1) (= x_2 x) (< x_2 100) (= x_3 (+ x_2 1)) (= conf_1_2 (+ 486 216)) (= conf_1_2 conf_1!) (= x_3 x!) (= conf_0 conf_0_0) (= conf_0! conf_0_0) (= conf_2 conf_2_0) (= conf_2! conf_2_0) (= conf_3 conf_3_0) (= conf_3! conf_3_0) (= conf_4 conf_4_0) (= conf_4! conf_4_0))))
(define-fun post-f ((conf_0 Int) (conf_1 Int) (conf_2 Int) (conf_3 Int) (conf_4 Int) (x Int) (conf_0_0 Int) (conf_1_0 Int) (conf_1_1 Int) (conf_1_2 Int) (conf_2_0 Int) (conf_3_0 Int) (conf_4_0 Int) (x_0 Int) (x_1 Int) (x_2 Int) (x_3 Int)) Bool
    (or (not (and (= conf_0 conf_0_0) (= conf_1 conf_1_1) (= conf_2 conf_2_0) (= conf_3 conf_3_0) (= conf_4 conf_4_0) (= x x_2))) (not (and (not (< x_2 100)) (not (= x_2 100))))))

(inv-constraint inv-f pre-f trans-f post-f)

(check-synth)
