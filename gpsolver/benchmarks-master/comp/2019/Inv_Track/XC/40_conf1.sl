(set-logic LIA)

(declare-primed-var c Int)
(declare-primed-var conf_0 Int)
(declare-primed-var n Int)
(declare-primed-var tmp Int)
(declare-primed-var c_0 Int)
(declare-primed-var c_1 Int)
(declare-primed-var c_2 Int)
(declare-primed-var c_3 Int)
(declare-primed-var c_4 Int)
(declare-primed-var c_5 Int)
(declare-primed-var conf_0_0 Int)
(declare-primed-var conf_0_1 Int)
(declare-primed-var conf_0_2 Int)
(declare-primed-var conf_0_3 Int)
(declare-primed-var conf_0_4 Int)
(declare-primed-var n_0 Int)
(synth-inv inv-f ((c Int) (conf_0 Int) (n Int) (tmp Int) (c_0 Int) (c_1 Int) (c_2 Int) (c_3 Int) (c_4 Int) (c_5 Int) (conf_0_0 Int) (conf_0_1 Int) (conf_0_2 Int) (conf_0_3 Int) (conf_0_4 Int) (n_0 Int)))

(define-fun pre-f ((c Int) (conf_0 Int) (n Int) (tmp Int) (c_0 Int) (c_1 Int) (c_2 Int) (c_3 Int) (c_4 Int) (c_5 Int) (conf_0_0 Int) (conf_0_1 Int) (conf_0_2 Int) (conf_0_3 Int) (conf_0_4 Int) (n_0 Int)) Bool
    (and (and (and (and (and (= c c_1) (= conf_0 conf_0_0)) (= n n_0)) (= conf_0_0 2)) (= c_1 0)) (> n_0 0)))
(define-fun trans-f ((c Int) (conf_0 Int) (n Int) (tmp Int) (c_0 Int) (c_1 Int) (c_2 Int) (c_3 Int) (c_4 Int) (c_5 Int) (conf_0_0 Int) (conf_0_1 Int) (conf_0_2 Int) (conf_0_3 Int) (conf_0_4 Int) (n_0 Int) (c! Int) (conf_0! Int) (n! Int) (tmp! Int) (c_0! Int) (c_1! Int) (c_2! Int) (c_3! Int) (c_4! Int) (c_5! Int) (conf_0_0! Int) (conf_0_1! Int) (conf_0_2! Int) (conf_0_3! Int) (conf_0_4! Int) (n_0! Int)) Bool
    (or (or (or (or (and (and (and (and (and (and (and (= c_2 c) (= conf_0_1 conf_0)) (= c_2 c!)) (= conf_0_1 conf_0!)) (= c c!)) (= conf_0 conf_0!)) (= n n!)) (= tmp tmp!)) (and (and (and (and (and (and (and (and (and (and (and (= c_2 c) (= conf_0_1 conf_0)) (> c_2 n_0)) (= c_3 (+ c_2 1))) (= conf_0_2 (- 161 conf_0_1))) (= c_4 c_3)) (= conf_0_3 conf_0_2)) (= c_4 c!)) (= conf_0_3 conf_0!)) (= n n_0)) (= n! n_0)) (= tmp tmp!))) (and (and (and (and (and (and (and (and (and (= c_2 c) (= conf_0_1 conf_0)) (not (> c_2 n_0))) (= c_4 c_2)) (= conf_0_3 conf_0_1)) (= c_4 c!)) (= conf_0_3 conf_0!)) (= n n_0)) (= n! n_0)) (= tmp tmp!))) (and (and (and (and (and (and (and (and (and (and (and (= c_2 c) (= conf_0_1 conf_0)) (= c_2 n_0)) (= c_5 1)) (= conf_0_4 (+ conf_0_1 483))) (= c_4 c_5)) (= conf_0_3 conf_0_4)) (= c_4 c!)) (= conf_0_3 conf_0!)) (= n n_0)) (= n! n_0)) (= tmp tmp!))) (and (and (and (and (and (and (and (and (and (= c_2 c) (= conf_0_1 conf_0)) (not (= c_2 n_0))) (= c_4 c_2)) (= conf_0_3 conf_0_1)) (= c_4 c!)) (= conf_0_3 conf_0!)) (= n n_0)) (= n! n_0)) (= tmp tmp!))))
(define-fun post-f ((c Int) (conf_0 Int) (n Int) (tmp Int) (c_0 Int) (c_1 Int) (c_2 Int) (c_3 Int) (c_4 Int) (c_5 Int) (conf_0_0 Int) (conf_0_1 Int) (conf_0_2 Int) (conf_0_3 Int) (conf_0_4 Int) (n_0 Int)) Bool
    (or (not (and (and (= c c_2) (= conf_0 conf_0_1)) (= n n_0))) (not (and (not (= c_2 n_0)) (not (>= c_2 0))))))

(inv-constraint inv-f pre-f trans-f post-f)

(check-synth)

