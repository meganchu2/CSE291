(set-logic BV)

(define-fun origCir ((LN89 Bool) (k7 Bool)) Bool
    (and LN89 k7))
(synth-fun skel ((LN89 Bool) (k7 Bool)) Bool
    ((Start Bool ((and depth1 depth1) (not depth1) (or depth1 depth1) (xor depth1 depth1)))
    (depth1 Bool ((and depth2 depth2) (not depth2) (or depth2 depth2) (xor depth2 depth2) LN89))
    (depth2 Bool ((and depth3 depth3) (not depth3) (or depth3 depth3) (xor depth3 depth3)))
    (depth3 Bool ((and depth4 depth4) (not depth4) (or depth4 depth4) (xor depth4 depth4)))
    (depth4 Bool (k7))))

(declare-var LN89 Bool)
(declare-var k7 Bool)
(constraint (= (origCir LN89 k7) (skel LN89 k7)))

(check-synth)

