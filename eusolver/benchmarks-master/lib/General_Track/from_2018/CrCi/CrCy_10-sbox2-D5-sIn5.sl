(set-logic BV)

(define-fun origCir ((LN17 Bool) (k7 Bool) (LN34 Bool) (LN40 Bool)) Bool
    (xor LN40 (xor (xor k7 LN17) LN34)))
(synth-fun skel ((LN17 Bool) (k7 Bool) (LN34 Bool) (LN40 Bool)) Bool
    ((Start Bool) (depth1 Bool) (depth2 Bool) (depth3 Bool) (depth4 Bool))
    ((Start Bool ((and depth1 depth1) (not depth1) (or depth1 depth1) (xor depth1 depth1)))
    (depth1 Bool ((and depth2 depth2) (not depth2) (or depth2 depth2) (xor depth2 depth2)))
    (depth2 Bool ((and depth3 depth3) (not depth3) (or depth3 depth3) (xor depth3 depth3) LN34 LN40))
    (depth3 Bool ((and depth4 depth4) (not depth4) (or depth4 depth4) (xor depth4 depth4) LN17))
    (depth4 Bool (k7))))

(declare-var LN17 Bool)
(declare-var k7 Bool)
(declare-var LN34 Bool)
(declare-var LN40 Bool)
(constraint (= (origCir LN17 k7 LN34 LN40) (skel LN17 k7 LN34 LN40)))

(check-synth)

