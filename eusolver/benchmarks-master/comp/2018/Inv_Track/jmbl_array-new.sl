(set-logic LIA)

(synth-inv InvF ((x Int) (y Int) (z Int)))

(declare-primed-var x Int)
(declare-primed-var y Int)
(declare-primed-var z Int)

(define-fun PreF ((x Int) (y Int) (z Int)) Bool
(= x 0))

(define-fun TransF ((x Int) (y Int) (z Int) (x! Int) (y! Int) (z! Int)) Bool
(or 
(and (= x! (+ x 1))
(and (= y! z!)
(and (<= z! y)
(< x 500))))

(and (= x! (+ x 1))
(and (= y! y)
(and (> z! y)
(< x 500))))
))



(define-fun PostF ((x Int) (y Int) (z Int)) Bool
(not (and (>= x 500) (< z y))))

(inv-constraint InvF PreF TransF PostF)

(check-synth)