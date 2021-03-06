(set-logic BV)

(define-fun hd05 ((x (_ BitVec 32))) (_ BitVec 32)
    (bvor x (bvsub x #x00000001)))
(synth-fun f ((x (_ BitVec 32))) (_ BitVec 32)
    ((Start (_ BitVec 32)) (NT1 (_ BitVec 32)))
    ((Start (_ BitVec 32) (#x00000001 #x00000000 #xffffffff x (bvnot NT1) (bvneg NT1) (bvxor NT1 NT1) (bvand NT1 NT1) (bvor NT1 NT1) (bvsdiv NT1 NT1) (bvadd NT1 NT1) (bvmul NT1 NT1) (bvudiv NT1 NT1) (bvurem NT1 NT1) (bvlshr NT1 NT1) (bvashr NT1 NT1) (bvshl NT1 NT1) (bvsub NT1 NT1) (bvsrem NT1 NT1)))
    (NT1 (_ BitVec 32) (#x00000001 #x00000000 #xffffffff x))))

(declare-var x (_ BitVec 32))
(constraint (= (hd05 x) (f x)))

(check-synth)

