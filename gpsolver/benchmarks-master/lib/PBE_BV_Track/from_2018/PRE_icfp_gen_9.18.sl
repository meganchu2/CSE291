(set-logic BV)

(define-fun ehad ((x (_ BitVec 64))) (_ BitVec 64)
    (bvlshr x #x0000000000000001))
(define-fun arba ((x (_ BitVec 64))) (_ BitVec 64)
    (bvlshr x #x0000000000000004))
(define-fun shesh ((x (_ BitVec 64))) (_ BitVec 64)
    (bvlshr x #x0000000000000010))
(define-fun smol ((x (_ BitVec 64))) (_ BitVec 64)
    (bvshl x #x0000000000000001))
(define-fun im ((x (_ BitVec 64)) (y (_ BitVec 64)) (z (_ BitVec 64))) (_ BitVec 64)
    (ite (= x #x0000000000000001) y z))
(synth-fun f ((x (_ BitVec 64))) (_ BitVec 64)
    ((Start (_ BitVec 64)))
    ((Start (_ BitVec 64) (#x0000000000000000 #x0000000000000001 x (bvnot Start) (smol Start) (ehad Start) (arba Start) (shesh Start) (bvand Start Start) (bvor Start Start) (bvxor Start Start) (bvadd Start Start) (im Start Start Start)))))

(constraint (= (f #xCEC82F2E96E668B2) #x0000000000000001))
(constraint (= (f #xBCC0BC6852185AE6) #x0000000000000001))
(constraint (= (f #x30B43615AA983D54) #x0000000000000001))
(constraint (= (f #x40E2B4CB5A9E54F6) #x0000000000000001))
(constraint (= (f #x52555BEDD494A3EE) #x0000000000000001))
(constraint (= (f #xFFFFFFFF80000002) #x0000000000000000))
(constraint (= (f #x0000000000000001) #x0000000000000000))
(constraint (= (f #x7475A26A39213808) #x0001D1D689A8E484))
(constraint (= (f #x8057EB804CF5E077) #x0002015FAE0133D4))
(constraint (= (f #xBFF0F265EAF4FFFF) #x0002FFC3C997ABD0))
(constraint (= (f #x2F0600431013128D) #x0000BC18010C404C))
(constraint (= (f #x1CD164D410AEFFFF) #x00007345935042B8))
(constraint (= (f #x00000000001026E1) #x0000000000000000))
(constraint (= (f #x00000000001CC6B2) #x0000000000000001))
(constraint (= (f #x0000000000124A1E) #x0000000000000001))
(constraint (= (f #x0000000000186AB0) #x0000000000000001))
(constraint (= (f #x00000000001E43E5) #x0000000000000000))
(constraint (= (f #x0000000000150EE4) #x0000000000000001))
(constraint (= (f #x00000000001AFFFF) #x0000000000000000))
(constraint (= (f #x00000000001181FB) #x0000000000000000))
(constraint (= (f #x00000000001777FA) #x0000000000000001))
(constraint (= (f #x000000000018FFFF) #x0000000000000000))

(check-synth)

