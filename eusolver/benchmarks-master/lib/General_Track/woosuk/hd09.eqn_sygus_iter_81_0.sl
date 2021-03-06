(set-logic BV)

(define-fun origCir ((n109 Bool) (i22 Bool) (n233 Bool) (n111 Bool)) Bool
    (and (xor (and (and n109 i22) (not n233)) n109) (not (and n233 n111))))
(synth-fun skel ((n109 Bool) (i22 Bool) (n233 Bool) (n111 Bool)) Bool
    ((Start Bool) (depth6 Bool) (depth5 Bool) (depth4 Bool) (depth3 Bool) (depth2 Bool) (depth1 Bool) (depth0 Bool))
    ((Start Bool (depth6))
    (depth6 Bool ((and depth5 depth5) (or depth5 depth5) (xor depth6 depth6) (not depth6) depth5))
    (depth5 Bool ((and depth4 depth4) (or depth4 depth4) (xor depth5 depth5) (not depth5) depth4 n233))
    (depth4 Bool ((and depth3 depth3) (or depth3 depth3) (xor depth4 depth4) (not depth4) depth3))
    (depth3 Bool ((and depth2 depth2) (or depth2 depth2) (xor depth3 depth3) (not depth3) depth2))
    (depth2 Bool ((and depth1 depth1) (or depth1 depth1) (xor depth2 depth2) (not depth2) depth1))
    (depth1 Bool ((and depth0 depth0) (or depth0 depth0) (xor depth1 depth1) (not depth1) depth0))
    (depth0 Bool (true false (xor depth0 depth0) (not depth0) n109 i22 n111))))

(declare-var n109 Bool)
(declare-var i22 Bool)
(declare-var n233 Bool)
(declare-var n111 Bool)
(constraint (= (origCir n109 i22 n233 n111) (skel n109 i22 n233 n111)))

(check-synth)

