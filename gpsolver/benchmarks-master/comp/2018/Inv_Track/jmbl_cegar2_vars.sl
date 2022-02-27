(set-logic LIA)

(synth-inv InvF ((x Int) (n Int) (m Int) (z1 Int) (z2 Int) (z3 Int)))

(declare-primed-var x Int)
(declare-primed-var n Int)
(declare-primed-var m Int)
(declare-primed-var z1 Int)
(declare-primed-var z2 Int)
(declare-primed-var z3 Int)

(define-fun PreF ((x Int) (n Int) (m Int) (z1 Int) (z2 Int) (z3 Int)) Bool
(and (= x 0) (= m 0)))

(define-fun TransF ((x Int) (n Int) (m Int) (z1 Int) (z2 Int) (z3 Int) (x! Int) (n! Int) (m! Int) (z1! Int) (z2! Int) (z3! Int)) Bool
(or
(and (and (and (< x n) (= x! (+ x 1))) (= n! n)) (= m! m))

(and (and (and (< x n) (= x! (+ x 1))) (= n! n)) (= m! x))))


(define-fun PostF ((x Int) (n Int) (m Int) (z1 Int) (z2 Int) (z3 Int)) Bool
(not (and (and (>= x n) (> n 0))
(or (<= n m) (< m 0)))))

(inv-constraint InvF PreF TransF PostF)

(check-synth)