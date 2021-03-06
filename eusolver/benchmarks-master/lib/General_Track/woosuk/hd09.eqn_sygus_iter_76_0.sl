(set-logic BV)

(define-fun origCir ((n80 Bool) (n95 Bool) (n99 Bool) (n108 Bool) (n103 Bool) (n110 Bool)) Bool
    (and (and (and (and n108 n99) n103) (and n95 n80)) n110))
(synth-fun skel ((n80 Bool) (n95 Bool) (n99 Bool) (n108 Bool) (n103 Bool) (n110 Bool)) Bool
    ((Start Bool) (depth5 Bool) (depth4 Bool) (depth3 Bool) (depth2 Bool) (depth1 Bool) (depth0 Bool))
    ((Start Bool (depth5))
    (depth5 Bool ((and depth4 depth4) (or depth4 depth4) (xor depth5 depth5) (not depth5) depth4))
    (depth4 Bool ((and depth3 depth3) (or depth3 depth3) (xor depth4 depth4) (not depth4) depth3))
    (depth3 Bool ((and depth2 depth2) (or depth2 depth2) (xor depth3 depth3) (not depth3) depth2 n80 n95 n103))
    (depth2 Bool ((and depth1 depth1) (or depth1 depth1) (xor depth2 depth2) (not depth2) depth1))
    (depth1 Bool ((and depth0 depth0) (or depth0 depth0) (xor depth1 depth1) (not depth1) depth0))
    (depth0 Bool (true false (xor depth0 depth0) (not depth0) n99 n108 n110))))

(declare-var n80 Bool)
(declare-var n95 Bool)
(declare-var n99 Bool)
(declare-var n108 Bool)
(declare-var n103 Bool)
(declare-var n110 Bool)
(constraint (= (origCir n80 n95 n99 n108 n103 n110) (skel n80 n95 n99 n108 n103 n110)))

(check-synth)

