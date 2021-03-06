(set-logic BV)

(define-fun origCir ((LN8 Bool) (k4 Bool) (LN75 Bool) (LN222 Bool) (LN226 Bool)) Bool
    (not (not (xor LN226 (xor (xor LN75 (and LN8 k4)) LN222)))))
(synth-fun skel ((LN8 Bool) (k4 Bool) (LN75 Bool) (LN222 Bool) (LN226 Bool)) Bool
    ((Start Bool) (depth1 Bool) (depth2 Bool) (depth3 Bool) (depth4 Bool) (depth5 Bool) (depth6 Bool))
    ((Start Bool ((and depth1 depth1) (not depth1) (or depth1 depth1) (xor depth1 depth1)))
    (depth1 Bool ((and depth2 depth2) (not depth2) (or depth2 depth2) (xor depth2 depth2)))
    (depth2 Bool ((and depth3 depth3) (not depth3) (or depth3 depth3) (xor depth3 depth3)))
    (depth3 Bool ((and depth4 depth4) (not depth4) (or depth4 depth4) (xor depth4 depth4) LN226))
    (depth4 Bool ((and depth5 depth5) (not depth5) (or depth5 depth5) (xor depth5 depth5) LN222))
    (depth5 Bool ((and depth6 depth6) (not depth6) (or depth6 depth6) (xor depth6 depth6) LN8 LN75))
    (depth6 Bool (k4))))

(declare-var LN8 Bool)
(declare-var k4 Bool)
(declare-var LN75 Bool)
(declare-var LN222 Bool)
(declare-var LN226 Bool)
(constraint (= (origCir LN8 k4 LN75 LN222 LN226) (skel LN8 k4 LN75 LN222 LN226)))

(check-synth)

