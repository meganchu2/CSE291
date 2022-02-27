(set-logic BV)

(define-fun ehad ((x (BitVec 64))) (BitVec 64)
    (bvlshr x #x0000000000000001))
(define-fun arba ((x (BitVec 64))) (BitVec 64)
    (bvlshr x #x0000000000000004))
(define-fun shesh ((x (BitVec 64))) (BitVec 64)
    (bvlshr x #x0000000000000010))
(define-fun smol ((x (BitVec 64))) (BitVec 64)
    (bvshl x #x0000000000000001))
(define-fun im ((x (BitVec 64)) (y (BitVec 64)) (z (BitVec 64))) (BitVec 64)
    (ite (= x #x0000000000000001) y z))
(synth-fun f ((x (BitVec 64))) (BitVec 64)
    ((Start (BitVec 64) (#x0000000000000000 #x0000000000000001 x (bvnot Start) (smol Start) (ehad Start) (arba Start) (shesh Start) (bvand Start Start) (bvor Start Start) (bvxor Start Start) (bvadd Start Start) (im Start Start Start)))))

(constraint (= (f #x44BDEA77706EF696) #x44BDEA77706EF696))
(constraint (= (f #x378FE2296B844650) #x378FE2296B844650))
(constraint (= (f #x48A7DB48F96DF2C6) #x48A7DB48F96DF2C6))
(constraint (= (f #x9D4E550A2D0B80A0) #x9D4E550A2D0B80A0))
(constraint (= (f #x619584AFE2CEF05E) #x619584AFE2CEF05E))
(constraint (= (f #x000000000000987D) #x000000000000987D))
(constraint (= (f #x000000000000D48A) #x000000000000D48A))
(constraint (= (f #x0000000000017BF1) #x0000000000017BF1))
(constraint (= (f #x000000000001FE42) #x000000000001FE42))
(constraint (= (f #x0000000000014A4D) #x0000000000014A4D))
(constraint (= (f #x6986047E857AFF17) #x06986047E857AFF1))
(constraint (= (f #x355AB728FF7EFBBB) #x0355AB728FF7EFBB))
(constraint (= (f #x7CBB16A4867B4B1C) #x07CBB16A4867B4B1))
(constraint (= (f #xDD80A894275C97F0) #x0DD80A894275C97F))
(constraint (= (f #x8358AF3AC699C18C) #x08358AF3AC699C18))
(constraint (= (f #x0000000000000000) #x0000000000000000))

(check-synth)

