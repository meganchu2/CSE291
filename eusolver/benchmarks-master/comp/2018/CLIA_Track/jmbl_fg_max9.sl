; mux_5.sl
; Synthesize the mux_imum of 5 integers, from a purely declarative spec

(set-logic LIA)

(synth-fun mux_9 ((x1 Int) (x2 Int) (x3 Int) (x4 Int) (x5 Int)
                 (x6 Int) (x7 Int) (x8 Int) (x9 Int)) Int
)

(declare-var x1 Int)
(declare-var x2 Int)
(declare-var x3 Int)
(declare-var x4 Int)
(declare-var x5 Int)
(declare-var x6 Int)
(declare-var x7 Int)
(declare-var x8 Int)
(declare-var x9 Int)

(constraint (>= (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9) x1))
(constraint (>= (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9) x2))
(constraint (>= (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9) x3))
(constraint (>= (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9) x4))
(constraint (>= (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9) x5))
(constraint (>= (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9) x6))
(constraint (>= (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9) x7))
(constraint (>= (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9) x8))
(constraint (>= (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9) x9))


(constraint (or (= x1 (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9))
            (or (= x2 (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9))
            (or (= x3 (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9))
            (or (= x4 (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9))
            (or (= x5 (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9))
            (or (= x6 (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9))
            (or (= x7 (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9))
            (or (= x8 (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9))
	        (= x9 (mux_9 x1 x2 x3 x4 x5 x6 x7 x8 x9)))))))))))

(check-synth)


