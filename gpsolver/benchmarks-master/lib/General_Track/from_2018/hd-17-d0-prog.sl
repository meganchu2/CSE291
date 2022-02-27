(set-logic BV)

(define-fun hd17 ((x (_ BitVec 32))) (_ BitVec 32)
    (bvand (bvadd (bvor x (bvsub x #x00000001)) #x00000001) x))
(synth-fun f ((x (_ BitVec 32))) (_ BitVec 32)
    ((Start (_ BitVec 32)))
    ((Start (_ BitVec 32) ((bvand Start Start) (bvadd Start Start) (bvsub Start Start) (bvor Start Start) x #x00000001))))

(declare-var x (_ BitVec 32))
(constraint (= (hd17 x) (f x)))

(check-synth)

